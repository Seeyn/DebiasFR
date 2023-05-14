from .Degradations import Degradation
import torch
import torch.nn as nn
from torch.autograd import Variable

class DegradationProcess(nn.Module):
    def __init__(self):
        super(DegradationProcess, self).__init__()
        self.Degrader = Degradation(differentiable=True)
        
    def forward(self, out, degradation_args):
        img_lq = []
        for img,degradation_arg in zip(out,degradation_args):
            kernel,scale,noise,quality = degradation_arg
            img_lq.append(self.Degrader(img,kernel,scale,noise,quality))
        
        return torch.stack(img_lq)