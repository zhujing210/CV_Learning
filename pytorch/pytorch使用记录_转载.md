# pytorch使用记录

2020-03-19 

- [learning](http://fenggangliu.top/categories/learning/)

pytorch相关问题



# 预训练模型

## 下载

[官方ImageNet预训练模型说明(有对应top1&top5指标)](https://pytorch.org/docs/stable/torchvision/models.html)

[对应模型找url下载云训练模型](https://github.com/pytorch/vision/tree/master/torchvision/models)

## 加载

```
pretrained_dict = torch.load(os.path.expanduser(model_paths[arch]),
map_location=lambda storage, loc: storage)  # os.path.expanduser把~/变成具体地址
model_dict = model.state_dict()  # 获取模型的参数字典
pretrained_dict = {k: v for k, v in pretrained_dict.items() if
not k.count("fc")}  # 可遍历字典, 做任意修改(比如k.repalce()换key的名称). 也可以判断model_dict是否有
model_dict.update(pretrained_dict) # pretrained_dict与model_dict相同key的value的维度必须相同
model.load_state_dict(model_dict)  # 更新模型权重
```

## 训练部分参数

[参考](https://www.jianshu.com/p/d67d62982a24)

### 固定部分参数

```
for k,v in model.named_parameters():
    if k!='xxx.weight':
        v.requires_grad=False#固定参数
```

也可通过查询`v.requires_grad`, 看是否被固定.

如果参数被固定, 则计算出的梯度为`None`.

> 查看梯度
>
> ```
> x.grad for x in self.optimizer.param_groups[0]['params']
> ```

### optimize部分参数

```
#将要训练的参数放入优化器
optimizer=torch.optim.Adam(params=[model.xxx.weight],lr=lr)
```

# checkpoint

pytorch 模型 .pt, .pth, .pkl都一样, 只是后缀不同

## save

```
def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:  # 单独保存best_model
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
# 调用函数, 定义state
save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'best_acc_top5': best_acc_top5,
      'optimizer' : optimizer.state_dict(),
      }, is_best, args.save)
```

## load

直接load后覆盖当前的值

```
checkpoint = torch.load(args.resume, map_location='cpu')
args.start_epoch = checkpoint['epoch']
best_acc_top1 = checkpoint['best_acc_top1']
best_acc_top5 = checkpoint['best_acc_top5']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])  # 加载optimizer参数
```

## 每一个param

```
# 打印模型的 state_dict
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
# 打印优化器的 state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

## 跨设备保存与加载模型

### GPU上与CPU相互加载

#### GPU保存, CPU加载

```
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))
```

如果模型在多个GPU上训练，那么在CPU上加载时，会报错`unexpected key “module.conv1.weight”`. 因为多GPU训练并保存模型时，模型的参数名都带上了`module`前缀，因此可以在加载模型时，把key中的这个前缀去掉

```
state_dict = torch.load('myfile.pth.tar')
# 创建一个不包含`module.`的新OrderedDict
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # 去掉 `module.`
    new_state_dict[name] = v
# 加载参数
model.load_state_dict(new_state_dict)
```

#### CPU保存, GPU加载

```
device = torch.device("cuda")
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # 选择希望使用的GPU
# 需要使用model.to(torch.devie('cuda'))将初始化的模型转换为CUDA优化模型
model.to(device)
```

### 单卡保存, 多卡加载

和上面问题相同, 需要手动把key的`module`前缀去掉

# 加速数据加载

## prefetch_generator

**安装：**

```
pip install prefetch_generator
```

**使用：**

```
# 新建DataLoaderX类
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
```

然后用 `DataLoaderX` 替换原本的 `DataLoader`。

**提速原因：**

> 原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
> 使用 prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。

## DALI

https://github.com/tanglang96/DataLoaders_DALI/blob/master/imagenet.py

## num_worker

`DataLoader` 的 `num_worker` 如果设置太小，则不能充分利用多线程提速，如果设置太大，会造成线程阻塞，或者撑爆内存，反而导致训练变慢甚至程序崩溃。

他的大小和具体的硬件和软件都有关系，所以没有一个统一的标准，可以通过一些简单的实验来确定。

我的经验是gpu 的数量的4倍比较合适

# 类激活映射（CAM）

https://blog.csdn.net/u014264373/article/details/85415921

# Tensorboard

torch1.1.0后可不用tensorboardX, 直接用tensorboard.

- 需要注意版本的匹配, torch>=1.1，tensorboard >=1.14
- 使用过程中全靠CPU跑,会占用很多内存
- 结果会存在新建的./runs文件夹下

```
pip install tensorboard future
```

## graph与scalar

```
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义网络
class Test_model(nn.Module):
    def __init__(self):
        super(Test_model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.layer(x)

model = Test_model()

writer = SummaryWriter()
writer.add_graph(model, input_to_model=torch.randn((3,3)))
writer.add_scalar(tag="test", scalar_value=torch.tensor(1)
                    , global_step=1)
writer.close()
# 或者用 with SummaryWriter(comment='net') as w: w.add_graph(...)
```

### graph中模块命名

比如ResNet中如果不给残差块命名, 相同stage会自动按顺序命名’0,1,2,3,,’, 但是不同stage间会命名重复, 导致最后graph重叠错乱

```
# 错误示范
layers = []
layers.append(nn.Conv2d(1,20,5))
....
model = nn.Sequential(*layers)
# 正确操作1
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
# 正确操作2
layers = OrderedDict()
layers['conv1'] = nn.Conv2d(1,20,5)
....
model = nn.Sequential(layers)
# 正确操作3
model = nn.Sequential()
model.add_module("conv1",nn.Conv2d(1,20,5))
model.add_module('relu1', nn.ReLU())
```

## 特征图可视化

https://zhuanlan.zhihu.com/p/60753993

## 其他

官网例程: https://github.com/lanpa/tensorboardX

# 搭建网络方法

[源自博客园](https://www.cnblogs.com/denny402/p/7593301.html)

```
import torch
import torch.nn.functional as F
from collections import OrderedDict
```

## 最原始:继承nn.Module类

```
class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

print("Method 1:")
model1 = Net1()
print(model1)
```

[![img](http://fenggangliu.top/2020/03/19/pytorch%E4%BD%BF%E7%94%A8%E8%AE%B0%E5%BD%95/module.png)](http://fenggangliu.top/2020/03/19/pytorch使用记录/module.png)

## Sequential打包

用Sequential()容器进行快速搭建，模型的各层被顺序添加到容器中。缺点是每层的编号是默认的阿拉伯数字，不易区分。

```
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 2:")
model2 = Net2()
print(model2)
```

[![img](http://fenggangliu.top/2020/03/19/pytorch%E4%BD%BF%E7%94%A8%E8%AE%B0%E5%BD%95/sequan.png)](http://fenggangliu.top/2020/03/19/pytorch使用记录/sequan.png)

## add_module+Sequential打包并单独命名

这种方法是对第二种方法的改进：通过add_module()添加每一层，并且为每一层增加了一个单独的名字。

add_module可用来实现nn.Sequential()动态添加方法

```
class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv=torch.nn.Sequential()
        self.conv.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv.add_module("relu1",torch.nn.ReLU())
        self.conv.add_module("pool1",torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential()
        self.dense.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense.add_module("relu2",torch.nn.ReLU())
        self.dense.add_module("dense2",torch.nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 3:")
model3 = Net3()
print(model3)
```

[![img](http://fenggangliu.top/2020/03/19/pytorch%E4%BD%BF%E7%94%A8%E8%AE%B0%E5%BD%95/add.png)](http://fenggangliu.top/2020/03/19/pytorch使用记录/add.png)

## OrderedDict+Sequential打包并单独命名

和add_module效果一样, 通过字典的形式添加每一层，并且设置单独的层名称。

```
class Net4(torch.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", torch.nn.ReLU()),
                    ("pool", torch.nn.MaxPool2d(2))
                ]
            ))

        self.dense = torch.nn.Sequential(
            OrderedDict([
                ("dense1", torch.nn.Linear(32 * 3 * 3, 128)),
                ("relu2", torch.nn.ReLU()),
                ("dense2", torch.nn.Linear(128, 10))
            ])
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 4:")
model4 = Net4()
print(model4)
```

[![img](http://fenggangliu.top/2020/03/19/pytorch%E4%BD%BF%E7%94%A8%E8%AE%B0%E5%BD%95/order.png)](http://fenggangliu.top/2020/03/19/pytorch使用记录/order.png)

## ModuleDict

ModuleDict可以形成字典的module, 从而可以用key调用相应module

```
blockDict = nn.ModuleDict(zip(archStrList, blockList))
```

其中, archStrList是一个字符串列表用来指定blockList里对应顺序的名字, blockList是一个字典module的list

# 基础元素

## Variable

每个Variable都有三个属性:

- data: .data(来访问原始的张量tensor)
- grad: 对应Tensor的梯度.grad(梯度会被累计到 `.grad`上)
- grad_fn: Variable是通过什么方式得到的.grad_fn(除了用户创建的 Variable 外 - 它们的 grad_fn is None)。

## 计算图与动态图机制

## autograd

https://blog.csdn.net/byron123456sfsfsfa/article/details/92210253

# 推荐链接

[PyTorch trick 集锦](https://zhuanlan.zhihu.com/p/76459295)

> Last updated: 2020-05-12 07:59:19
> 转载注明出处，原文地址：[fenggangliu.top/2020/03/19/pytorch%E4%BD%BF%E7%94%A8%E8%AE%B0%E5%BD%95/](http://fenggangliu.top/2020/03/19/pytorch使用记录/)