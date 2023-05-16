# Pytorch
import torch
import torch.nn as nn
# Local
from .modules import compress_jpeg, decompress_jpeg
from .utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg( rounding=rounding, factor=factor)

    def forward(self, x):

        height, width = x.shape[2:]
        orig_height, orig_width = height, width
        if height % 16 != 0 or width % 16 != 0:
            # Round up to next multiple of 16
            height = ((height - 1) // 16 + 1) * 16
            width = ((width - 1) // 16 + 1) * 16

            vpad = height - orig_height
            wpad = width - orig_width
            top = vpad // 2
            bottom = vpad - top
            left = wpad // 2
            right = wpad - left
            
            x = torch.nn.functional.pad(x, [left, right,top, bottom], 'replicate')
        
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr,height,width)
        if orig_height != height or orig_width != width:
            recovered = recovered[:,:, top:-bottom, left:-right]
        return recovered/255.

    def set_quality(self, quality):
        factor = quality_to_factor(quality)
        self.compress.factor = factor
        self.decompress.factor = factor


if __name__ == '__main__':
    with torch.no_grad():
        import cv2
        import numpy as np

        img = cv2.imread("Lena.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inputs = np.transpose(img, (2, 0, 1))
        inputs = inputs[np.newaxis, ...]

        tensor = torch.FloatTensor(inputs)
        jpeg = DiffJPEG(512, 512, differentiable=True)

        quality = 80
        jpeg.set_quality(quality)

        outputs = jpeg(tensor)
        outputs = outputs.detach().numpy()
        outputs = np.transpose(outputs[0], (1, 2, 0))

        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)

        cv2.imshow("QF:"+str(quality), outputs / 255.)
        cv2.waitKey()

        from skimage.metrics import peak_signal_noise_ratio as PSNR
        img = cv2.imread("Lena.png")
        print(PSNR(np.uint8(outputs), np.uint8(img)))
