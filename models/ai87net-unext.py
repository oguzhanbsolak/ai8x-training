import torch
import torch.nn as nn
import torch.nn.functional as F
import ai8x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=False):
        super().__init__()
        if max_pool:
            self.depthwise = ai8x.FusedMaxPoolDepthwiseConv2dBN(in_channels, in_channels, kernel_size=3, padding=1)
        else:
            self.depthwise = ai8x.FusedDepthwiseConv2dBN(in_channels, in_channels, kernel_size=3, padding=1)
        self.pointwise = ai8x.FusedConv2dBNReLU(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=False):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, max_pool)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.layer1 = ConvBlock(in_channels, embed_dims[0])
        self.layer2 = ConvBlock(embed_dims[0], embed_dims[1], max_pool=True)
        self.layer3 = ConvBlock(embed_dims[1], embed_dims[2], max_pool=True)
        self.layer4 = ConvBlock(embed_dims[2], embed_dims[3], max_pool=True)
        self.layer5 = ConvBlock(embed_dims[3], embed_dims[4], max_pool=True)  # Additional Layer

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)  # Additional Layer
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, num_classes, embed_dims):
        super().__init__()
        self.layer1 = ConvBlock(embed_dims[4], embed_dims[4])
        self.up1 = ai8x.ConvTranspose2d(embed_dims[4], embed_dims[3], kernel_size=3, stride=2, padding=1)
        self.match_x4 = ai8x.Conv2d(embed_dims[3], embed_dims[3], kernel_size=1)

        self.layer2 = ConvBlock(embed_dims[3], embed_dims[3])
        self.up2 = ai8x.ConvTranspose2d(embed_dims[3], embed_dims[2], kernel_size=3, stride=2, padding=1)
        self.match_x3 = ai8x.Conv2d(embed_dims[2], embed_dims[2], kernel_size=1)

        self.layer3 = ConvBlock(embed_dims[2], embed_dims[2])
        self.up3 = ai8x.ConvTranspose2d(embed_dims[2], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.match_x2 = ai8x.Conv2d(embed_dims[1], embed_dims[1], kernel_size=1)

        self.layer4 = ConvBlock(embed_dims[1], embed_dims[1])
        self.up4 = ai8x.ConvTranspose2d(embed_dims[1], embed_dims[0], kernel_size=3, stride=2, padding=1)
        self.match_x1 = ai8x.Conv2d(embed_dims[0], embed_dims[0], kernel_size=1)

        self.final_conv = ai8x.Conv2d(embed_dims[0], num_classes, kernel_size=1)

        self.resid = ai8x.Add()

    def forward(self, x):
        x1, x2, x3, x4, x5 = x

        x = self.layer1(x5)
        x = self.up1(x)
        x4 = self.match_x4(x4)
        x = self.resid(x, x4)
        
        x = self.layer2(x)
        x = self.up2(x)
        x3 = self.match_x3(x3)
        x = self.resid(x, x3)
        
        x = self.layer3(x)
        x = self.up3(x)
        x2 = self.match_x2(x2)
        x = self.resid(x, x2)
        
        x = self.layer4(x)
        x = self.up4(x)
        x1 = self.match_x1(x1)
        x = self.resid(x, x1)
        
        x = self.final_conv(x)

        return x


class CNNMidModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.layer1 = ai8x.FusedConv2dBNReLU(in_channels=in_channels, out_channels=2*in_channels, kernel_size=1, stride=1, padding=0)
        self.layer2 = ai8x.FusedConv2dBNReLU(in_channels=2*in_channels, out_channels=2*in_channels, kernel_size=1, stride=1, padding=0)
        self.layer3 = ai8x.FusedConv2dBNReLU(in_channels=2*in_channels, out_channels=2*in_channels, kernel_size=1, stride=1, padding=0)
        self.layer4 = ai8x.FusedConv2dBN(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class LightAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = ai8x.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attn = F.sigmoid(self.conv(x))
        return x * attn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.fc1 = ai8x.FusedAvgPoolConv2dBNReLU(in_channels, in_channels // reduction, kernel_size=1,
                                                 padding=0, pool_size=8, pool_stride=8)
        self.fc2 = ai8x.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        se = self.fc1(x)
        se = F.sigmoid(self.fc2(se))
        return x * se


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = ai8x.FusedLinearReLU(in_features, hidden_features)
        self.fc2 = ai8x.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class AttentionMidModule(nn.Module):
    def __init__(self, in_channels, mlp_ratio):
        super().__init__()
        
        self.attn_block = LightAttention(in_channels)
        self.se_block = SEBlock(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=in_channels*mlp_ratio)

    def forward(self, x):
        attn_features = self.attn_block(x)
        se_features = self.se_block(attn_features)
        batch_size, channels, height, width = se_features.shape
        se_features_flat = se_features.view(batch_size, channels, -1).permute(0, 2, 1).contiguous()
        mlp_features = self.mlp(se_features_flat)
        mlp_features = mlp_features.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)
        
        return mlp_features


class UNext_NoMidModule(nn.Module):
    def __init__(self, num_classes=4, num_channels=48, dimensions=(128, 128), bias=True, fold_ratio=4,
                 embed_dims=[32, 64, 128, 256, 256], **kwargs): 
        super().__init__()
        self.num_final_channels = num_classes * fold_ratio * fold_ratio

        self.encoder = Encoder(num_channels, embed_dims)
        self.decoder = Decoder(self.num_final_channels, embed_dims)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class UNext_CNNMidModule(nn.Module):
    def __init__(self, num_classes=4, num_channels=48, dimensions=(128, 128), bias=True, fold_ratio=4,
                 embed_dims=[32, 64, 128, 256, 256], **kwargs):         
        super().__init__()
        self.num_final_channels = num_classes * fold_ratio * fold_ratio

        self.encoder = Encoder(num_channels, embed_dims)
        self.mid_module = CNNMidModule(embed_dims[-1])
        self.decoder = Decoder(self.num_final_channels, embed_dims)

    def forward(self, x):
        x = self.encoder(x)
        x[-1] = self.mid_module(x[-1])
        x = self.decoder(x)
        return x


class UNext_AttentionMidModule(nn.Module):
    def __init__(self, num_classes=4, num_channels=48, dimensions=(128, 128), bias=True, fold_ratio=4,
                 embed_dims=[32, 64, 128, 256, 256], mlp_ratio=2, **kwargs):
        super().__init__()
        self.num_final_channels = num_classes * fold_ratio * fold_ratio

        self.encoder = Encoder(num_channels, embed_dims)
        self.mid_module = AttentionMidModule(embed_dims[-1], mlp_ratio)
        self.decoder = Decoder(self.num_final_channels, embed_dims)

    def forward(self, x):
        x = self.encoder(x)
        x[-1] = self.mid_module(x[-1])
        x = self.decoder(x)
        return x


def ai87unextnomidmodule(pretrained=False, **kwargs):
    """
    Constructs a large unet (unet_v7) model.
    """
    assert not pretrained
    return UNext_NoMidModule(**kwargs)


def ai87unextcnnmidmodule(pretrained=False, **kwargs):
    """
    Constructs a large unet (unet_v7) model.
    """
    assert not pretrained
    return UNext_CNNMidModule(**kwargs)


def ai87unextattentionmidmodule(pretrained=False, **kwargs):
    """
    Constructs a large unet (unet_v7) model.
    """
    assert not pretrained
    return UNext_AttentionMidModule(**kwargs)


models = [
    {
        'name': 'ai87unextnomidmodule',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai87unextcnnmidmodule',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai87unextattentionmidmodule',
        'min_input': 1,
        'dim': 2,
    },
]