import math
import random
from matplotlib import style
import torch
import clip
from torch import nn
from torch.nn import functional as F
class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, fc_dim, n_fc,
                 weight_norm=False, activation='lrelu', normalize_mlp=False):#, pixel_norm=False):
        super(MLP, self).__init__()
        if weight_norm:
            linear = EqualLinear
        else:
            linear = nn.Linear

        if activation == 'lrelu':
            actvn = nn.LeakyReLU(0.2,True)
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        self.model = []

        # normalize input
        if normalize_mlp:
            self.model += [PixelNorm()]

         # set the first layer
        self.model += [linear(input_dim, fc_dim),
                       actvn]
        if normalize_mlp:
            self.model += [PixelNorm()]

        # set the inner layers
        for i in range(n_fc - 2):
            self.model += [linear(fc_dim, fc_dim),
                           actvn]
            if normalize_mlp:
                self.model += [PixelNorm()]

        # set the last layer
        self.model += [linear(fc_dim, out_dim)] # no output activations

        # normalize output
        if normalize_mlp:
            self.model += [PixelNorm()]

        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        out = self.model(input)
        return out


class AttributeClassifier(nn.Module):
    def __init__(self):
        super(AttributeClassifier,self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.preprocess = preprocess
        self.clip_model = model
        self.clip_model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
        self.ageclassifier = MLP(512,10,256,8,weight_norm=False,normalize_mlp=False).to(device)
        self.genderclassifier =  MLP(512,2,256,8,weight_norm=False,normalize_mlp=False).to(device)
    
    def forward(self,input,out_represent=True):

        x = self.clip_model.encode_image(input).float()
        x = x/(torch.norm(x,dim=1).unsqueeze(1))
        age_pre = self.ageclassifier(x)
        gender_pre = self.genderclassifier(x)
        if out_represent:
            return age_pre, gender_pre,x
        else:
            return age_pre, gender_pre
