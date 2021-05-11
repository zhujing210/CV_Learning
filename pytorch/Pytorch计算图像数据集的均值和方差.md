[原文](https://www.cnblogs.com/wanghui-garcia/p/11448460.html)，防止以后找不到特做一个保存

在使用 torchvision.transforms进行数据处理时我们经常进行的操作是：

```
transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
```

前面的(0.485,0.456,0.406)表示均值，分别对应的是RGB三个通道；后面的(0.229,0.224,0.225)则表示的是标准差

这上面的均值和标准差的值是ImageNet数据集计算出来的，所以很多人都使用它们

但是如果你想要计算自己的数据集的均值和标准差，让其作为你的transforms.Normalize函数的参数的话可以进行下面的操作

代码get_mean_std.py：

```python
# coding:utf-8
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from dataloader import Dataloader
from options import options
import pickle
"""
    在训练前先运行该函数获得数据的均值和标准差
"""

class Dataloader():
    def __init__(self, opt):
        # 训练，验证，测试数据集文件夹名
        self.opt = opt
        self.dirs = ['train', 'test', 'testing']

        self.means = [0, 0, 0]
        self.stdevs = [0, 0, 0]

        self.transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),#数据值从[0,255]范围转为[0,1]，相当于除以255操作
                                        # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
                                        ])

        # 因为这里使用的是ImageFolder，按文件夹给数据分类，一个文件夹为一类，label会自动标注好
        self.dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), self.transform) for x in self.dirs}


    def get_mean_std(self, type, mean_std_path):
        """
        计算数据集的均值和标准差
        :param type: 使用的是那个数据集的数据，有'train', 'test', 'testing'
        :param mean_std_path: 计算出来的均值和标准差存储的文件
        :return: 
        """
        num_imgs = len(self.dataset[type])
        for data in self.dataset[type]:
            img = data[0]
            for i in range(3):
                # 一个通道的均值和标准差
                self.means[i] += img[i, :, :].mean()
                self.stdevs[i] += img[i, :, :].std()


        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        print("{} : normMean = {}".format(type, self.means))
        print("{} : normstdevs = {}".format(type, self.stdevs))
        
        # 将得到的均值和标准差写到文件中，之后就能够从中读取
        with open(mean_std_path, 'wb') as f:
            pickle.dump(self.means, f)
            pickle.dump(self.stdevs, f)
            print('pickle done')

if __name__ == '__main__':
    opt = options().parse()
    dataloader = Dataloader(opt)
    for x in dataloader.dirs:
        mean_std_path = 'mean_std_value_' + x + '.pkl'
        dataloader.get_mean_std(x, mean_std_path)
```

然后再从相应的文件读取均值和标准差放到dataloader.py的transforms.Normalize函数中即可：

```python
# coding:utf-8
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pickle


"""
    用于加载训练train、验证test和测试数据testing
"""

class Dataloader():
    def __init__(self, opt):
        # 训练，验证，测试数据集文件夹名
        self.opt = opt
        self.dirs = ['train', 'test', 'testing']
        # 均值和标准差存储的文件路径
        self.mean_std_path = {x: 'mean_std_value_' + x + '.pkl' for x in self.dirs}

        # 初始化为0
        self.means = {x: [0, 0, 0] for x in self.dirs}
        self.stdevs = {x: [0, 0, 0] for x in self.dirs}
        print(type(self.means['train']))
        print(self.means)
        print(self.stdevs)

        for x in self.dirs:
            #如果存在则说明之前有获取过均值和标准差
            if os.path.exists(self.mean_std_path[x]):
                with open(self.mean_std_path[x], 'rb') as f:
                    self.means[x] = pickle.load(f)
                    self.stdevs[x] = pickle.load(f)
                    print('pickle load done')

        print(self.means)
        print(self.stdevs)
        # 将相应的均值和标准差设置到transforms.Normalize函数中
        self.transform = {x: transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.means[x], self.stdevs[x]),
                                        ]) for x in self.dirs}
...
```

