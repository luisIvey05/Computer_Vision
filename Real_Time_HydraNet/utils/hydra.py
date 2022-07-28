import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    """
    conv3x3: A nn.Conv2d used for 3x3 convolution where kernel size is 3.
    Note: When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is also known as a "Depthwise Convolution".

    :param in_channels: Number of channels in the input image
    :param out_channels: Number of channels produced by the convolution
    :param stride: Stride of the convolution. Default=1
    :param dilation: Spacing between kernel elements. Default=1
    :param groups: Number of blocked connections form input channel to output channels. Default=1
    :param bias: If True, adds a learnable bias to the output. Default=False

    :return: A conv3x3 layer.

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias,
                     groups=groups)


def batchnorm(num_features):
    """
    batchnorm Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
    as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
    Shift.
    :param num_features: C from an expected input of size (N,C,H,W)
    :return: A batchnorm layer.
    """
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """
        conv1x1: A nn.Conv2d used for 1x1 pointwise convolution where kernel size is 1.

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param stride: Stride of the convolution. Default=1
        :param groups: Number of blocked connections form input channel to output channels. Default=1
        :param bias: If True, adds a learnable bias to the output. Default=False

        :return: A conv1x1 layer.

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)


def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    """
        convbnrelu: Convolution -> Batch Normalization -> Relu

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolution.
        :param stride: Stride of the convolution. Default=1
        :param groups: Number of blocked connections form input channel to output channels. Default=1
        :param act: If True, activation is applied to the output. Default=True

        :return: A Convolution/BatchNormalization/Relu layer.

        """
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels))


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)  # Boolean/Conditional
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        # residual = x
        out = self.output(x)
        if self.residual:
            return (out + x)  # +residual
        else:
            return out


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes, out_planes, stride=1, bias=False,
                            groups=in_planes if groups else 1))  # setattr(object, name, value)
            self.stride = 1
            self.n_stages = n_stages
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)  # getattr(object, num[, default])
            x = top + x
        return x


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tasks = 2
        self.num_classes = 6

    def define_mobilenet(self):
        mobilenet_config = [[1, 16, 1, 1],  # L1
                            [6, 24, 2, 2],  # L2
                            [6, 32, 3, 2],  # L3
                            [6, 64, 4, 2],  # L4
                            [6, 96, 3, 1],  # L5
                            [6, 160, 3, 2],  # L6
                            [6, 320, 1, 1],  # L7
                            ]
        self.in_channels = 32  # Number of input channels
        self.num_layers = len(mobilenet_config)
        self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)
        c_layer = 2
        for t, c, n, s in mobilenet_config:
            layers = []
            for idx in range(n):
                print("[INFO] IN_CHANNELS {}".format(self.in_channels))
                layers.append(InvertedResidualBlock(self.in_channels, c, expansion_factor=t, stride=s if
                idx == 0 else 1))
                self.in_channels = c
                setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))  # setattr(object, name, value)
                c_layer += 1  # stores mobilenet layers in hydranet

    def _make_crp(self, in_planes, out_planes, stages, groups=False):
        layers = [CRPBlock(in_planes, out_planes, stages, groups=groups)]  # Call a CRP Blocks in layers
        return nn.Sequential(*layers)

    def define_lightweight_refinenet(self):
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)

        self.crp4 = self._make_crp(256, 256, 4, groups=False)
        self.crp3 = self._make_crp(256, 256, 4, groups=False)
        self.crp2 = self._make_crp(256, 256, 4, groups=False)
        self.crp1 = self._make_crp(256, 256, 4, groups=True)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.pre_depth = conv1x1(256, 256, groups=256, bias=False)  # Defines the Purple Pre-Head for Depth
        self.depth = conv3x3(256, 1, bias=True)  # Defines the Final Layer of Depth
        self.pre_segm = conv1x1(256, 256, groups=256, bias=False)  # Defines the Purple Pre-Head for Segm
        self.segm = conv3x3(256, self.num_classes, bias=True)  # Defines the Final layer of Segmentation

        self.relu = nn.ReLU6(inplace=True)  # Defines a ReLU6 Operation inplace=if you want to modify the values or not

        if self.num_tasks == 3:
            # Define a Normal Head
            self.pre_normal = conv1x1(256, 256, groups=256, bias=False)
            self.normal = conv3x3(256, 3, bias=True)

    def forward(self, x):
        # MOBILENET V2
        x = self.layer1(x)
        x = self.layer2(x)  # x/2
        l3 = self.layer3(x)  # 24, x/4
        l4 = self.layer4(l3)  # 32, x/8
        l5 = self.layer5(l4)  # 64, x/16
        l6 = self.layer6(l5)  # 96, x/16
        l7 = self.layer7(l6)  # 160, x/32
        l8 = self.layer8(l7)  # 320, x/32

        # LIGHT-WEIGHT REFINENET
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # HEADS
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)

        out_d = self.pre_depth(l3)
        out_d = self.relu(out_d)
        out_d = self.depth(out_d)

        if self.num_tasks == 3:
            out_n = self.pre_normal(l3)
            out_n = self.relu(out_n)
            out_n = self.normal(out_n)
            return out_segm, out_d, out_n
        else:
            return out_segm, out_d
