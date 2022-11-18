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


def main():
    """Inference demo for GFPGAN (for users).
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

    # ------------------------ set up background upsampler ------------------------


    model_path = './dbfr_g_60000.pth'
    restorer = DBFRer(
        model_path=model_path,
        upscale=2,
        channel_multiplier=2,
        bg_upsampler=None)


    # ------------------------ restore ------------------------
    styles = []
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        for age in np.arange(10):
            # age = ages[int(img_path.split('/')[-1].split('.')[0])]
            # age = '0-2'
            age_vector = (0.2 * torch.randn(600)).cuda()
            age_vector[age*10:(age+1)*10] += 1

            age_vector[550:] += 1
            # restore faces and background if necessary
            cropped_faces, restored_faces_f ,style_code= restorer.enhance(
                input_img,age_vector, has_aligned=True, paste_back=True)
            styles.append(style_code)
            age_vector[550:] -= 1
            age_vector[500:550] += 1
            cropped_faces, restored_faces_m ,style_code= restorer.enhance(
                input_img,age_vector, has_aligned=True,  paste_back=True)
            styles.append(style_code)
            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces_f)):
                # save restored face
                save_face_name = f'{basename}_{age:02d}.png'
                save_restore_path = os.path.join(args.output, 'restored_faces_f', save_face_name)
                imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{age:02d}.png'))
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces_m)):
                # save restored face
               
                save_face_name = f'{basename}_{age:02d}.png'
                save_restore_path = os.path.join(args.output, 'restored_faces_m', save_face_name)
                imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{age:02d}.png'))


        print(f'Results are in the [{args.output}] folder.')



if __name__ == '__main__':
    main()
