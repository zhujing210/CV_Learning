import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBasicBlock(nn.Module): 
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResnetBasicBlock, self).__init__()
        # 定义残差块内连续的两个卷积层
        self.residual = nn.Sequential(
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channel)
        )

    def forward(self. x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x+output)


class ResnetDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResnetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(extra_x + output)


class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        self.conv1 = mm.Conv2d(3, 64, kernel_size = 7, stride=2, padding=3) # 卷积下取整，池化上取整
        self.bn1 = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResnetBasicBlock(64,64,1),
                                    ResnetBasicBlock(64, 64, 1))
        
        self.layer2 = nn.Sequential(ResnetDownBlock(64, 128, [2, 1]),
                                    ResnetBasicBlock(128, 128, 1))
        
        self.layer3 = nn.Sequential(ResnetDownBlock(128, 256, [2, 1]),
                                    ResnetBasicBlock(256, 256, 1))
        
        self.layer4 = nn.Sequential(ResnetDownBlock(256, 512, [2, 1]),
                                    ResnetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) #只需要关注输出维度的大小 output_size ，具体的实现过程和参数选择自动帮你确定了

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer3(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = output.reshape(x.shape[0], -1) # tensor行数为shape[0]，列数自定计算。即平铺开来
        output = self.fc(output)
        return output
        