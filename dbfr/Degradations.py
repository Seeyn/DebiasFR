import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from  .DiffJPEG import DiffJPEG 
class GaussianBlurConv(nn.Module):
    def __init__(self):
        super(GaussianBlurConv, self).__init__()
        self.channels = 3
    def __call__(self, x,kernel):
        if type(kernel)==np.ndarray:
            kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
            kernel = np.repeat(kernel, self.channels, axis=0)
        else:
            kernel = kernel.repeat(self.channels,1,1,1).float()

        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        x = F.pad(x,(20,20,20,20),'reflect')
        x = F.conv2d(x, self.weight , groups=self.channels)
        return x


class Degradation(nn.Module):
    def __init__(self,differentiable=False):
        super(Degradation, self).__init__()
        self.blur = GaussianBlurConv()
        self.jpeg = DiffJPEG(differentiable)

    def forward(self,x,blur_kernel,scale,noise,quality):
        x = self.blur(x,blur_kernel)        
        w,h = x.shape[2:]
        x = torch.nn.functional.interpolate(x , size=(int(w // scale), int(h // scale)),mode='bilinear')
        if type(blur_kernel)==np.ndarray:
            noise = torch.from_numpy(noise).permute(2,0,1).unsqueeze(0)
        else:
            noise = noise.permute(2,0,1).unsqueeze(0).float()
        x = (x+noise).clip(0,1)
        self.jpeg.set_quality(quality)
        x = self.jpeg((x*255.))
        x = torch.nn.functional.interpolate(x , size=(512,512),mode='bilinear')
        return x
