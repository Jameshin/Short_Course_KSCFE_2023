import torch
import torch.nn as nn
from ops import *


class Generator(nn.Module):
    
    def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16, act = nn.PReLU(), scale=4):
        super(Generator, self).__init__()
        
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 9, BN = False, act = act)
        
        resblocks = [ResBlock(channels = n_feats, kernel_size = 3, act = act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = True, act = None)
        
        if(scale == 4):
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = 2, act = act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = scale, act = act)]

        self.tail = nn.Sequential(*upsample_blocks)
        
        self.last_conv = conv(in_channel = n_feats, out_channel = img_feat, kernel_size = 3, BN = False, act = nn.Tanh())
        
    def forward(self, x):
        
        x = self.conv01(x)
        _skip_connection = x
        
        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection
        
        x = self.tail(feat)
        x = self.last_conv(x)
        
        return x, feat



class GeneratorS(nn.Module):
    
    def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 4, act = nn.PReLU(), scale=4):
        super(GeneratorS, self).__init__()
        
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 9, BN = False, act = act)
        
        resblocks = [ResBlock(channels = n_feats, kernel_size = 3, act = act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = True, act = None)
        
        if(scale == 4):
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = 2, act = act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = scale, act = act)]

        self.tail = nn.Sequential(*upsample_blocks)
        
        self.last_conv = conv(in_channel = n_feats, out_channel = img_feat, kernel_size = 3, BN = False, act = nn.Tanh())
        
    def forward(self, x):
        
        x = self.conv01(x)
        _skip_connection = x
        
        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection
        
        x = self.tail(feat)
        x = self.last_conv(x)
        
        return x, feat
        

class GeneratorSF(nn.Module):
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, num_block=4, act=nn.PReLU(), scale=4):
        super(GeneratorSF, self).__init__()

        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=9, BN=False, act=act)

        resblocks = []
        for i in range(num_block):
            if i == 1 or i == 2:
                resblocks.append(ResBlockF(channels=n_feats, kernel_size=3, act=act, bias=True, out_channel=n_feats//2, first_out_channel=n_feats//2))
            else:
                resblocks.append(ResBlockF(channels=n_feats, kernel_size=3, act=act, bias=True))
        self.body = nn.Sequential(*resblocks)

        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=True, act=None)

        if scale == 4:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)]

        self.tail = nn.Sequential(*upsample_blocks)

        self.last_conv = conv(in_channel=n_feats, out_channel=img_feat, kernel_size=3, BN=False, act=nn.Tanh())

    def forward(self, x):
        x = self.conv01(x)
        _skip_connection = x

        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection

        x = self.tail(feat)
        x = self.last_conv(x)

        return x, feat
        
'''
# for 32 filters in all residual blocks
class GeneratorSF(nn.Module):

    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, num_block=4, act=nn.PReLU(), scale=4):
        super(GeneratorSF, self).__init__()

        self.conv01 = nn.Conv2d(in_channels=img_feat, out_channels=n_feats, kernel_size=9, padding=4)

        resblocks = [ResBlockF(channels=n_feats, kernel_size=3, act=act, filter_size=32) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)

        self.conv02 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=n_feats)

        if scale == 4:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act),
                              Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act)]
        else:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)]

        self.tail = nn.Sequential(*upsample_blocks)

        self.last_conv = nn.Conv2d(in_channels=n_feats, out_channels=img_feat, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.conv01(x)
        _skip_connection = x

        x = self.body(x)
        x = self.conv02(x)
        x = self.bn(x)
        feat = x + _skip_connection

        x = self.tail(feat)
        x = self.last_conv(x)
        x = self.tanh(x)

        return x, feat

'''   
class Discriminator(nn.Module):
    
    def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, act = nn.LeakyReLU(inplace = True), num_of_block = 3, patch_size = 96):
        super(Discriminator, self).__init__()
        self.act = act
        
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act)
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act, stride = 2)
        
        body = [discrim_block(in_feats = n_feats * (2 ** i), out_feats = n_feats * (2 ** (i + 1)), kernel_size = 3, act = self.act) for i in range(num_of_block)]    
        self.body = nn.Sequential(*body)
        
        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))
        
        tail = []
        
        tail.append(nn.Linear(self.linear_size, 1024))
        tail.append(self.act)
        tail.append(nn.Linear(1024, 1))
        tail.append(nn.Sigmoid())
        
        self.tail = nn.Sequential(*tail)
        
        
    def forward(self, x):
        
        x = self.conv01(x)
        x = self.conv02(x)
        x = self.body(x)        
        x = x.view(-1, self.linear_size)
        x = self.tail(x)
        
        return x

