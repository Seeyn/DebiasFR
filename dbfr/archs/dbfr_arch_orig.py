import math
import random
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F
from .stylegan2_bilinear_arch import (ConvLayer,ResBlock, EqualConv2d, EqualLinear, ScaledLeakyReLU,
                                      StyleGAN2GeneratorBilinear)

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, device='cpu'):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale
        self.device = device

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale, self.device)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, device='cpu'):
    return scale * F.leaky_relu(input + bias.view((1, -1)+(1,)*(len(input.shape)-2)), negative_slope=negative_slope)




class ConvUpLayer(nn.Module):
    """Convolutional upsampling layer. It uses bilinear upsampler + Conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.scale is used to scale the convolution weights, which is related to the common initializations.
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        # activation
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Module):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class StyleGAN2GeneratorFusion(StyleGAN2GeneratorBilinear):
    """

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 lr_mlp=0.01,
                 narrow=1,
                 sft_half=False):
        super(StyleGAN2GeneratorFusion, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            lr_mlp=lr_mlp,
            narrow=narrow)
        self.sft_half = sft_half
        self.calayer =  nn.ModuleList()
        self.ad_gen = nn.ModuleList()
        self.ad_enc = nn.ModuleList()
        print(self.log_size)
        for i in range(self.log_size,2,-1):
            out_channels = self.channels[f'{2**i}']
            self.calayer.insert(0,nn.Sequential(EqualConv2d(out_channels*2,out_channels,1,padding=0, bias=True, bias_init_val=0),nn.Sigmoid()))
            self.calayer.insert(0,nn.Sequential(EqualConv2d(out_channels*2,out_channels,1,padding=0, bias=True, bias_init_val=0),nn.Sigmoid()))
            self.ad_gen.insert(0,nn.Sequential(EqualConv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=True, bias_init_val=0),ScaledLeakyReLU(0.2)))
            self.ad_enc.insert(0,nn.Sequential(EqualConv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=True, bias_init_val=0),ScaledLeakyReLU(0.2)))
            self.ad_gen.insert(0,nn.Sequential(EqualConv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=True, bias_init_val=0),ScaledLeakyReLU(0.2)))
            self.ad_enc.insert(0,nn.Sequential(EqualConv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=True, bias_init_val=0),ScaledLeakyReLU(0.2)))

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False,
                style_codes= None,
                input_is_style=False):
        """

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        # print(out)
        if input_is_style:
            out , return_style= self.style_conv1(out, style_codes[0], noise=noise[0],
            input_is_style=input_is_style)
            print(out)
        else:
            out , return_style= self.style_conv1(out, latent[:, 0], noise=noise[0],
            input_is_style=input_is_style)
        # print(return_style.shape)
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            if input_is_style:
                out,s = conv1(out, style_codes[i], noise=noise1,input_is_style=input_is_style)
            else:
                out,s = conv1(out, latent[:, i], noise=noise1,input_is_style=input_is_style)
            return_style = torch.cat(( return_style ,s),dim=1)
            # print(s.shape)
            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    #out = out * conditions[i - 1] + conditions[i]
                    out_ = torch.cat([self.ad_gen[i-1](out),self.ad_enc[i-1](conditions[i-1])],dim=1)
                    out_ = self.calayer[i-1](out_)
                    out = out + conditions[i-1]*out_
                    out = F.leaky_relu(out, negative_slope=0.2)* math.sqrt(2)

            if input_is_style:
                out,s = conv2(out, style_codes[i+1], noise=noise2,input_is_style=input_is_style)
            else:
                out,s = conv2(out, latent[:, i + 1], noise=noise2,input_is_style=input_is_style)
            return_style = torch.cat(( return_style ,s),dim=1)
            # print(s.shape)
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    #out = out * conditions[i - 1] + conditions[i]
                    out_ = torch.cat([self.ad_gen[i](out),self.ad_enc[i](conditions[i])],dim=1)
                    out_ = self.calayer[i](out_)
                    out = out + conditions[i]*out_
                    out = F.leaky_relu(out, negative_slope=0.2)* math.sqrt(2)

            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, return_style #latent
        else:
            return image, None
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


  
class Bottleneck(nn.Module):
    

    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential( 
                    EqualConv2d(in_channels,in_channels//2, 1, stride=1, padding=0, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2))
        self.conv2 = nn.Sequential( 
                    EqualConv2d(in_channels//2, in_channels//2, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2))
        self.conv3 =  EqualConv2d(in_channels//2, out_channels, 1, stride=1, padding=0, bias=True, bias_init_val=0)             
        self.skip = EqualConv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=True, bias_init_val=0)
        self.activation =  ScaledLeakyReLU(0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        skip = self.skip(x)
        out = self.activation(out + skip) 
        return out

@ARCH_REGISTRY.register()
class DBFR(nn.Module):
    """

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        decoder_load_path (str): The path to the pre-trained decoder model (usually, the StyleGAN2). Default: None.
        fix_decoder (bool): Whether to fix the decoder. Default: True.

        num_mlp (int): Layer number of MLP style layers. Default: 8.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Whether to use different latent w for different layers. Default: False.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(
            self,
            out_size,
            num_style_feat=512,
            channel_multiplier=1,
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):

        
        super(DBFR, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = ConvLayer(3, channels[f'{first_out_size}'], 1, bias=True, activate=True)
        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.final_conv = ConvLayer(in_channels, channels['4'], 3, bias=True, activate=True)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(EqualConv2d(channels[f'{2**i}'], 3, 1, stride=1, padding=0, bias=True, bias_init_val=0))
            #self.toRGB.append(nn.Conv2d(channels[f'{2**i}'], 3, 1))
        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = EqualLinear(
            channels['4'] * 4 * 4, linear_out_channel, bias=True, bias_init_val=0, lr_mul=1, activation=None)
        # self.final_linear = nn.Linear(channels['4'] * 4 * 4, linear_out_channel)
        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorFusion(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            lr_mlp=lr_mlp,
            narrow=narrow,
            sft_half=sft_half)

        # load pre-trained stylegan2 model if necessary
        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(decoder_load_path, map_location=lambda storage, loc: storage)['params_ema'],strict=False)
        # fix decoder without updating params
        if fix_decoder:
        #    for _, param in self.stylegan_decoder.named_parameters():
        #        param.requires_grad = False
            num = 0
            for _, param in self.stylegan_decoder.named_parameters():
                if 'modulated_conv.modulation'  not in _ or 'style'  not in _ :
                    param.requires_grad = False
                    if 'cal'  in _ or 'ad'  in _:
                        param.requires_grad = True
                        num += 1
                        print(_)
                else:
                    num += 1
                    print(_)
            print(num)
        # for SFT modulations (scale and shift)
        self.condition_in1 = nn.ModuleList()
        self.condition_in2 = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            
            self.condition_in1.append(Bottleneck(out_channels,sft_out_channels))
            self.condition_in2.append(Bottleneck(out_channels,sft_out_channels))
            '''
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),ScaledLeakyReLU(0.2)))
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),ScaledLeakyReLU(0.2)))
            '''
        self.mlp = MLP(600,512,256,8,weight_norm=True,normalize_mlp=True)
        '''
        for _, param in self.condition_scale.named_parameters():
            param.requires_grad = False
        for _, param in self.condition_shift.named_parameters():
            param.requires_grad = False
        for _, param in self.conv_body_first.named_parameters():
            param.requires_grad = False
        for _, param in self.conv_body_down.named_parameters():
            param.requires_grad = False
        for _, param in self.condition_scale.named_parameters():
            param.requires_grad = False
        '''





    def forward(self, x,age,y, input_age=True,return_latents=False, return_rgb=True, randomize_noise=True,style_codes=None,input_is_style=False):
        """Forward function for GFPGANBilinear.

        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        unet_skips = []
        out_rgbs = []

        # encoder
        feat = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)

        feat = self.final_conv(feat)

        if not input_age:
            #original version
            # style code
            style_code = self.final_linear(feat.view(feat.size(0), -1))

            if self.different_w:
                style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)
        else:
            style_code = self.mlp(age)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            in1 = self.condition_in1[i](feat)
            conditions.append(in1.clone())
            in2 = self.condition_in2[i](feat)
            conditions.append(in2.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))
 
        # decoder
        image, return_style = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise,
                                         style_codes = style_codes,input_is_style=input_is_style)

        return image, out_rgbs, return_style

