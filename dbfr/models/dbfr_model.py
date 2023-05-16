import math
import os.path as osp
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm
import clip
import numpy as np
from .classifier_arch import AttributeClassifier
from .encoder_arch import ImageClassifier
from dbfr.Degradations import Degradation

@MODEL_REGISTRY.register()
class DebiasFR(BaseModel):

    def __init__(self, opt):
        super(DebiasFR, self).__init__(opt)
        self.idx = 0  # it is used for saving data for check

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            print( self.opt['path'].get('strict_load_g', False))
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', False), param_key)

        self.log_size = int(math.log(self.opt['network_g']['out_size'], 2))

        if self.is_train:
            self.init_training_settings()
            
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = AttributeClassifier()
        self.clip_model.load_state_dict(torch.load('./pretrained_models/net_best.pth')['model_state_dict'])
        self.clip_model.eval()
        # LR predictor

        self.LRpredictor = ImageClassifier().to(device)
        self.LRpredictor.load_state_dict(torch.load('./pretrained_models/net_best_tuned.pth')['model_state_dict'])
        self.LRpredictor.eval()


        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        for name, param in self.LRpredictor.named_parameters():
            param.requires_grad = False

        self.Degrader = Degradation(differentiable=True).to(device)

        self.criterion = torch.nn.CrossEntropyLoss(reduce='mean')
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=512 // 32)
    
    def init_training_settings(self):
        train_opt = self.opt['train']

        # ----------- define net_d ----------- #
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        # ----------- define net_g with Exponential Moving Average (EMA) ----------- #
        # net_g_ema only used for testing on one GPU and saving. There is no need to wrap with DistributedDataParallel
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g.train()
        self.net_d.train()
        self.net_g_ema.eval()
       
        # ----------- define losses ----------- #
        # pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # L1 loss is used in pyramid loss, component style loss and identity loss
        self.cri_l1 = build_loss(train_opt['L1_opt']).to(self.device)

        # gan loss (wgan)
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # regularization weights
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_d_reg_every = train_opt['net_d_reg_every']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.net_g.named_parameters():
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

        # ----------- optimizer d ----------- #
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        normal_params = []
        for _, param in self.net_d.named_parameters():
            normal_params.append(param)
        optim_params_d = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_d']['lr']
        }]
        optim_type = train_opt['optim_d'].pop('type')
        lr = train_opt['optim_d']['lr'] * net_d_reg_ratio
        betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
        self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, lr, betas=betas)
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.age = data['age'].to(self.device)
        self.age_gt = data['age_gt'].to(self.device)
        self.gender_gt = data['gender_gt'].to(self.device)
        
        if 'kernel' in data:
            self.degradation_args = [data['kernel'].to(self.device),data['scale'].to(self.device),data['noise'].to(self.device),data['quality'].to(self.device)]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def construct_img_pyramid(self):
        """Construct image pyramid for intermediate restoration loss"""
        pyramid_gt = [self.gt]
        down_img = self.gt
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(down_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid_gt.insert(0, down_img)
        return pyramid_gt

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    
    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        
        input_age =  self.opt['train'].get('input_age', False)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            
            if np.random.uniform() < self.opt['train'].get('pseudo_prob', 0):
                lr_age_pre,lr_gender_pre = self.LRpredictor(torch.nn.functional.interpolate(self.lq,(256,256)))
                value,index = torch.topk(lr_age_pre[0],2)
                value = torch.nn.functional.softmax(value)
                age_vector = torch.zeros((1,10)).cuda()
                gender_vector = torch.zeros((1,2)).cuda()
                age_vector[0,index] = value
                value,index = torch.topk(lr_gender_pre[0],2)
                value = torch.nn.functional.softmax(value)
                gender_vector[0,index] = value

                pseudo_output, _,_ = self.net_g(self.lq,age_vector,gender_vector,self.gt, input_age,return_latents=True, return_rgb=True)                
                pseudo_age_pre,pseudo_gender_pre,vector = self.clip_model(self.avg_pool(self.upsample(pseudo_output)))
                #Classification loss
                pseudo_age_loss = self.criterion(pseudo_age_pre,age_vector)
                pseudo_gender_loss = self.criterion(pseudo_gender_pre,gender_vector)
                self.output = pseudo_output 
                #LR pixel-wise loss（pseudo pair）
                pseudo_tmp = []
                for index,degradation_arg in enumerate(zip(*self.degradation_args)):
                    kernel,scale,noise,quality = degradation_arg
                    noise_new = noise[:int(noise[-1].item())]
                    original_length = int((noise[-1]/3).sqrt().item())
                    noise_new = noise_new.view(original_length,original_length,3)
                    pseudo_tmp.append(self.Degrader(pseudo_output[index].unsqueeze(0),kernel,scale,noise_new,quality).squeeze(0))
                pseudo_lq = torch.stack(pseudo_tmp) 
                pseudo_lq_pix = self.cri_pix(pseudo_lq,self.lq)
                pseudo_class_weight =  self.opt['train'].get('pseudo_class_weight', 0)
                degradation_weight =  self.opt['train'].get('degradation_weight', 0)
          
                l_g_total += pseudo_class_weight*(pseudo_age_loss + pseudo_gender_loss) + degradation_weight*(pseudo_lq_pix)

                loss_dict['pseudo_age_loss'] = pseudo_age_loss 
                loss_dict['pseudo_gender_loss'] = pseudo_gender_loss 
                loss_dict['pseudo_lq_pix'] = pseudo_lq_pix 
        
            # image pyramid loss weight
            pyramid_loss_weight = self.opt['train'].get('pyramid_loss_weight', 0)
                
            if pyramid_loss_weight > 0 and current_iter > self.opt['train'].get('remove_pyramid_loss', float('inf')):
                pyramid_loss_weight = 1e-12  # very small weight to avoid unused param error
            if pyramid_loss_weight > 0:
                b = self.age_gt.shape[0]
                age_vector = torch.zeros((b,10)).cuda()
                for i in range(b):
                    age_vector[i][self.age_gt[i]] = 1
                gender_vector = torch.zeros((b,2)).cuda()
                for i in range(b):
                    gender_vector[i][self.gender_gt[i]] = 1
                self.output, out_rgbs,latent = self.net_g(self.lq,age_vector,gender_vector,self.gt, input_age,return_latents=True, return_rgb=True)
                pyramid_gt = self.construct_img_pyramid()
            else:
                self.output, out_rgbs = self.net_g(self.lq, return_rgb=False)
            #clip loss
            # resize
            out_c = self.avg_pool(self.upsample(self.output))
            age_pre,gender_pre,vector = self.clip_model(out_c)

            loss1 = self.criterion(age_pre,self.age_gt) 
            loss2 = self.criterion(gender_pre,self.gender_gt)
                
            l_class = (loss1+loss2) * self.opt['train'].get('class_weight', 0)
            l_g_total += l_class
            loss_dict['l_class'] = l_class

            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # image pyramid loss
            if pyramid_loss_weight > 0:
                for i in range(0, self.log_size - 2):
                    l_pyramid = self.cri_l1(out_rgbs[i], pyramid_gt[i]) * pyramid_loss_weight
                    l_g_total += l_pyramid
                    loss_dict[f'l_p_{2**(i+3)}'] = l_pyramid

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # prevent from crush
            if not (torch.any(torch.isnan(l_g_total)) or torch.any(torch.isinf(l_g_total))):
                l_g_total.backward()
                self.optimizer_g.step()

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

        # ----------- optimize net_d ----------- #
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()
        

        fake_d_pred = self.net_d(self.output.detach())
        real_d_pred = self.net_d(self.gt)
        l_d = self.cri_gan(real_d_pred, True, is_disc=True) + self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d'] = l_d
        # In WGAN, real_score should be positive and fake_score should be negative
        loss_dict['real_score'] = real_d_pred.detach().mean()
        loss_dict['fake_score'] = fake_d_pred.detach().mean()
        #l_d.backward()

        if not (torch.any(torch.isnan(l_g_total)) or torch.any(torch.isinf(l_g_total))):
            l_d.backward()

        # regularization loss
        if current_iter % self.net_d_reg_every == 0:
            self.gt.requires_grad = True
            real_pred = self.net_d(self.gt)
            l_d_r1 = r1_penalty(real_pred, self.gt)
            l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
            loss_dict['l_d_r1'] = l_d_r1.detach().mean()
            l_d_r1.backward()

        if not (torch.any(torch.isnan(l_g_total)) or torch.any(torch.isinf(l_g_total))):
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                input_age =  self.opt['train'].get('input_age', False)
                b = self.age_gt.shape[0]
                age_vector = torch.zeros((b,10)).cuda()
                for i in range(b):
                    age_vector[i][self.age_gt[i]] = 1
                gender_vector = torch.zeros((b,2)).cuda()
                for i in range(b):
                    gender_vector[i][self.gender_gt[i]] = 1
                self.output, out_rgbs,latent = self.net_g_ema(self.lq,age_vector,gender_vector,self.gt, input_age,return_latents=True, return_rgb=True)
                # self.output, _ = self.net_g_ema(self.lq)
                out_c = self.avg_pool(self.upsample(self.output))
                age_pre,gender_pre,vector = self.clip_model(out_c)
                self.acc1 = int(age_pre[0].argmax() == self.age_gt[0])
                self.acc2 = int(gender_pre[0].argmax() == self.gender_gt[0])
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                self.output, _ = self.net_g(self.lq)
                self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.best_metric_results[dataset_name]['age_acc'] = dict(better=True, val=float('-inf'), iter=-1)
            self.best_metric_results[dataset_name]['gender_acc'] = dict(better=True, val=float('-inf'), iter=-1)
        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        self.metric_results['age_acc'] = 0
        self.metric_results['gender_acc'] = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            sr_img = tensor2img(self.output.detach().cpu(), min_max=(-1, 1))
            metric_data['img'] = sr_img
            if hasattr(self, 'gt'):
                gt_img = tensor2img(self.gt.detach().cpu(), min_max=(-1, 1))
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                    self.metric_results['age_acc'] += self.acc1
                    self.metric_results['gender_acc'] += self.acc2
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)


            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        # save net_g and net_d
        self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        self.save_network(self.net_d, 'net_d', current_iter)       
        # save training state
        self.save_training_state(epoch, current_iter)
