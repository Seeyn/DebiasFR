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
# sys.path.append('/home/zelin/AttributeEncode/DBFR_modified')

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
  

    
    model_path = './dbfr_g_60000.pth'
    restorer = DBFRer(
        model_path=model_path,
        upscale=2,
        channel_multiplier=2,
        bg_upsampler=None)
    with open('./name2age.txt','rb') as f:
        ages = pickle.load(f)
    with open('./name2gender.txt','rb') as f:
        genders = pickle.load(f)
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
        age = ages[int(img_path.split('/')[-1].split('.')[0])]
        # age = '0-2'
        age_vector = (0.2 * torch.randn(600)).cuda()
        age_vector[group[age]*10:(group[age]+1)*10] += 1
        # age_vector[550:] += 1
        if genders[int(img_path.split('/')[-1].split('.')[0])] == 1:
            age_vector[500:550] += 1
        else:
            age_vector[550:] += 1
        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,age_vector, has_aligned=True, paste_back=True)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        
            # save restored face
            # if args.suffix is not None:
            #     save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
            # else:
            save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))


    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()
