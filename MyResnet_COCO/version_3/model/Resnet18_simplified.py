import torch
from torch import Tensor
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#这个实现的是两层的残差块，用于resnet18/34
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  #当连接的维度不同时，使用1*1的卷积核将低维转成高维，然后才能进行相加

        out += identity                     #实现H(x)=F(x)+x或H(x)=F(x)+Wx
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    #参数block指明残差块是两层或三层，参数layers指明每个卷积层需要的残差块数量，
    # num_classes指明分类数，zero_init_residual是否初始化为0
    def __init__(
        self,
        #block: Type[Union[BasicBlock, Bottleneck]],
        block,
        layers,
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64, 
        #replace_stride_with_dilation: Optional[List[bool]] = None,
        replace_stride_with_dilation = None,
        #norm_layer: Optional[Callable[..., nn.Module]] = None
        norm_layer = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  #一开始先使用64*7*7的卷积核,stride=2, padding=3
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                 #kaiming高斯初始化，目的是使得Conv2d卷积层反向传播的输出的方差都为1
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                 #初始化m.weight，即gamma的值为1；m.bias即beta的值为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 在每个残差分支中初始化最后一个BN，即BatchNorm2d
        # 以便残差分支以零开始，并且每个残差块的行为类似于一个恒等式。
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                #Bottleneck的最后一个BN是m.bn3
                #if isinstance(m, Bottleneck): 
                #    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]  
                #BasicBlock的最后一个BN是m.bn2
                #elif isinstance(m, BasicBlock):
                #    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    #实现一层卷积，block参数指定是两层残差块或三层残差块，planes参数为输入的channel数，blocks说明该卷积有几个残差块
    #def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
    #                stride: int = 1, dilate: bool = False) -> nn.Sequential:
    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate:bool=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        #即如果该层的输入的channel数inplanes和其输出的channel数planes * block.expansion不同，
        #那要使用1*1的卷积核将输入x低维转成高维，然后才能进行相加
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        #只有卷积和卷积直接的连接需要低维转高维
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    #return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
    #               **kwargs)
    return ResNet( BasicBlock, [2, 2, 2, 2],**kwargs)