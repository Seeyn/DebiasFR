import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
from torchvision.transforms import Grayscale
from dbfr.Degradations import Degradation
import pickle

@DATASET_REGISTRY.register()
class FFHQDegradationDataset(data.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(FFHQDegradationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions

        if self.crop_components:
            # load component list from a pre-process pth files
            self.components_list = torch.load(opt.get('component_path'))

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend: scan file list from a folder
            self.paths = paths_from_folder(self.gt_folder)

        # degradation configurations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # to gray
        self.gray_prob = opt.get('gray_prob')

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')

        with open(opt['age_labels_path'],'rb') as f:
            self.ages = pickle.load(f)
        with open(opt['gender_labels_path'],'rb') as f:
            self.genders = pickle.load(f)

        self.group ={
        '0-2':0,
        '3-6':1,
        '7-9':2,
        '10-14':3,
        '15-19':4,
        '20-29':5,
        '30-39':6,
        '40-49':7,
        '50-69':8,
        '70-120':9
        }
        self.Degrader = Degradation(differentiable=False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        age = self.ages[int(gt_path.split('/')[-1].split('.')[0])]
        # age_vector = (0.2 * torch.randn(600)).cuda()
        # age_vector[self.group[age]*10:(self.group[age]+1)*10] += 1
        # if self.genders[int(gt_path.split('/')[-1].split('.')[0])] == 1:
        #     age_vector[500:550] += 1
        # else:
        #     age_vector[550:] += 1
        age_gt = self.group[age]
        gender_gt = self.genders[int(gt_path.split('/')[-1].split('.')[0])]
        #print(age_gt,gender_gt)
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        img_gt = cv2.resize(img_gt, (512,512), interpolation=cv2.INTER_LINEAR)
        h, w, _ = img_gt.shape
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        sigma = np.random.uniform(self.noise_range[0], self.noise_range[1])
        noise = np.float32(np.random.randn(*([int(w // scale), int(h // scale),3]))) * sigma / 255.
        noise_pass = noise.flatten()
        length = noise_pass.shape[0]
        noise_new = np.zeros(513*513*3)
        noise_new[:length] = noise_pass
        noise_new[-1] = length


        # jpeg compression
        quality = int(np.random.uniform(self.jpeg_range[0], self.jpeg_range[1]))
        # print('degrade')
        img_lq = self.Degrader(img_gt.unsqueeze(0),kernel,scale,noise,quality)
        # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        # print('fail')
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = Grayscale(3)(img_lq)


        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        
        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)
    
        #[kernel,scale,noise,quality]
        return {'lq': img_lq.squeeze(0).detach() , 'gt': img_gt, 'gt_path': gt_path,'age': torch.zeros(1),'age_gt':torch.tensor(age_gt),'gender_gt':torch.tensor(gender_gt),'kernel':kernel,'scale':scale,'noise':noise_new,'quality':quality}

    def __len__(self):
        return len(self.paths)
