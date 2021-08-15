import torch
import torch.nn as nn
from .equiconv import DeformConv2d_plus_Offset


def convrelu(in_channels, out_channels, kernel, padding, layerdict, offsetdict, layer_number):
    return nn.Sequential(
        DeformConv2d_plus_Offset(in_channels, out_channels, kernel, padding=padding,
        offset_input=offsetdict[layerdict[layer_number]]),
        nn.ReLU(inplace=True),
    )


def equiconvrelu(in_channels, out_channels, kernel, padding, layerdict, offsetdict, layer_number):
    return nn.Sequential(
        DeformConv2d_plus_Offset(in_channels, out_channels, kernel, padding=padding,
        offset_input=offsetdict[layerdict[layer_number]]),
        nn.ReLU(inplace=True),
    )


def base_layer_0_3(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False,
        offset_input=offsetdict[layerdict[2]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
    )


def base_layer_3_5(layerdict, offsetdict):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[3]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[4]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[5]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[6]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )


def base_layer_5(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[7]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[8]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        DeformConv2d_plus_Offset(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[9]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[10]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
   )


def base_layer_6(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[11]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[12]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        DeformConv2d_plus_Offset(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[13]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[14]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )


def base_layer_7(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[15]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[16]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        DeformConv2d_plus_Offset(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[17]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[18]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )


class UNetEquiconv(nn.Module):
    """UNet-equiconv"""
    def __init__(self, n_class, layer_dict=None, offset_dict=None):
        super(UNetEquiconv, self).__init__()

        self.layerdict = layer_dict
        self.offsetdict = offset_dict

        self.layer0 = base_layer_0_3(self.layerdict, self.offsetdict)
        self.layer0_1x1 = convrelu(64, 64, 1, 0, self.layerdict, self.offsetdict, layer_number=26)
        self.layer1 = base_layer_3_5(self.layerdict, self.offsetdict)
        self.layer1_1x1 = convrelu(64, 64, 1, 0, self.layerdict, self.offsetdict, layer_number=24)
        self.layer2 = base_layer_5(self.layerdict, self.offsetdict)
        self.layer2_1x1 = convrelu(128, 128, 1, 0, self.layerdict, self.offsetdict, layer_number=22)
        self.layer3 = base_layer_6(self.layerdict, self.offsetdict)
        self.layer3_1x1 = convrelu(256, 256, 1, 0, self.layerdict, self.offsetdict, layer_number=20)
        self.layer4 = base_layer_7(self.layerdict, self.offsetdict)
        self.layer4_1x1 = convrelu(512, 512, 1, 0, self.layerdict, self.offsetdict, layer_number=19)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = equiconvrelu(256 + 512, 512, 3, 1, self.layerdict, self.offsetdict, layer_number=21)
        self.conv_up2 = equiconvrelu(128 + 512, 256, 3, 1, self.layerdict, self.offsetdict, layer_number=23)
        self.conv_up1 = equiconvrelu(64 + 256, 256, 3, 1, self.layerdict, self.offsetdict, layer_number=25)
        self.conv_up0 = equiconvrelu(64 + 256, 128, 3, 1, self.layerdict, self.offsetdict, layer_number=27)
        self.conv_original_size0 = equiconvrelu(3, 64, 3, 1, self.layerdict, self.offsetdict, layer_number=0)
        self.conv_original_size1 = equiconvrelu(64, 64, 3, 1, self.layerdict, self.offsetdict, layer_number=1)
        self.conv_original_size2 = equiconvrelu(64 + 128, 64, 3, 1, self.layerdict, self.offsetdict, layer_number=28)
        self.conv_last_layer = DeformConv2d_plus_Offset(64, n_class, 1, offset_input=self.offsetdict[self.layerdict[29]])

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        out = self.conv_last_layer(x)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layerdict, offsetdict = torch.load('./layer_256x512.pt'), torch.load('./offset_256x512.pt')

    model = UNetEquiconv(8, layer_dict=layerdict, offset_dict=offsetdict)
    model = model.to(device)
    input0 = torch.randn(2, 3, 256, 512)
    input0 = input0.to(device)
    output0 = model(input0)
    print(f"Output shape: {output0.shape}")
