import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
import pickle
from dbfr import DBFRer
import sys
from dbfr.models import encoder_arch


def main():
    """Inference demo for DBFR (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='inputs/whole_imgs',
        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument('-p', '--pattern', type=str,default='LRlabel')
    # parser.add_argument('--attribute', '--pattern', type=str,default='LRlabel')
    parser.add_argument('--attribute', nargs='+', type=int, default=None)
    
    
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up DBFR restorer ------------------------
  

    model_path = './pretrained_models/DebiasFR_v1.pth'
    restorer = DBFRer(
        model_path=model_path,
        upscale=2,
        channel_multiplier=2,
        bg_upsampler=None)
    LRpredictor = encoder_arch.ImageClassifier()
    LRpredictor.load_state_dict(torch.load('./pretrained_models/Attribute_predictor.pth')['model_state_dict'])
    LRpredictor.eval()
    LRpredictor.cuda()

    # ages = np.load('celeba_age_gt.npy',allow_pickle=True,encoding='bytes').item()
    # genders = np.load('celeba_gender_gt.npy',allow_pickle=True,encoding='bytes').item()
    
    group ={
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
    # ------------------------ restore ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        key = img_path.split('/')[-1].split('.')[0].encode()
        pre_img = torch.from_numpy(input_img).float()/255.
        pre_img = torch.nn.functional.interpolate(pre_img.permute(2,0,1).unsqueeze(0),(256,256),mode='bilinear')
        pre_img = 2*((pre_img)-0.5)
        pre_img = pre_img[:,[2,1,0],:,:]
        lr_age_pre,lr_gender_pre = LRpredictor(pre_img.cuda())
        if args.attribute is None:
            if args.pattern=='Top2':
                value,index = torch.topk(lr_age_pre[0],2)
                value = torch.nn.functional.softmax(value)
                age_vector = torch.zeros((1,10)).cuda()
                gender_vector = torch.zeros((1,2)).cuda()
                age_vector[0,index] = value
                value,index = torch.topk(lr_gender_pre[0],2)
                value = torch.nn.functional.softmax(value)
                gender_vector[0,index] = value
            elif args.pattern=='LRlabel':
                tmp_age = lr_age_pre[0].argmax()
                tmp_gender = lr_gender_pre[0].argmax()
                age_vector = torch.zeros((1,10)).cuda()
                gender_vector = torch.zeros((1,2)).cuda()
                age_vector[0,tmp_age]=1
                gender_vector[0,tmp_gender] = 1
            elif args.pattern=='GTlabel':
                age_vector = torch.zeros((1,10)).cuda()
                gender_vector = torch.zeros((1,2)).cuda()
                age_vector[0,group[str(ages[key],'utf-8')]]=1
                gender_vector[0,genders[key]] = 1
        else:
            tmp_age, tmp_gender = args.attribute
            age_vector = torch.zeros((1,10)).cuda()
            gender_vector = torch.zeros((1,2)).cuda()
            age_vector[0,tmp_age]=1
            gender_vector[0,tmp_gender] = 1



        cropped_faces, restored_faces ,style_code= restorer.enhance(
                input_img,age_vector,gender_vector, has_aligned=True, paste_back=True)
     
        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):

            save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))


    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()
