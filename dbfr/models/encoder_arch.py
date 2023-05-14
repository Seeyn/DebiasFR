import math
import random
from matplotlib import style
import torch
import clip
from torch import nn
from torch.nn import functional as F
from math import sqrt

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class AgeEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=4, style_dim=50, padding_type='reflect',
                 conv_weight_norm=False, actvn='lrelu'):
        super(AgeEncoder, self).__init__()

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        encoder = [padding_layer(3), conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        activation]

        encoder += [conv2d(ngf * mult * 2, style_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        features = self.encoder(input)
        latent = features.mean(dim=3).mean(dim=2)
        return latent

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier,self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.encoder = ResNet(ResidualBlock, [3, 4, 6, 3])
        self.ageclassifier = MLP(2048,10,256,2,weight_norm=False,normalize_mlp=False).to(device)
        self.genderclassifier =  MLP(2048,2,256,2,weight_norm=False,normalize_mlp=False).to(device)

        #self.encoder = AgeEncoder(3,style_dim=512,conv_weight_norm=True).to(device)
        #self.ageclassifier = MLP(512,10,256,8,weight_norm=False,normalize_mlp=False).to(device)
        #self.genderclassifier =  MLP(512,2,256,8,weight_norm=False,normalize_mlp=False).to(device)
    
    def forward(self,input):

        x = self.encoder(input)
        # print(x.shape)
        age_pre = self.ageclassifier(x)
        gender_pre = self.genderclassifier(x)
        
        return age_pre, gender_pre
