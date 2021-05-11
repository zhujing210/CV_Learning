# [Pytorch中的Distributed Data Parallel与混合精度训练（Apex）](https://zhuanlan.zhihu.com/p/105755472)

# Pytorch的nn.DataParallel详解

> torch.nn.DataParalle只用于单机多卡

公司配备多卡的GPU服务器，当我们在上面跑程序的时候，当迭代次数或者epoch足够大的时候，我们通常会使用nn.DataParallel函数来用多个GPU来加速训练。一般我们会在代码中加入以下这句：

```text
device_ids = [0, 1]
net = torch.nn.DataParallel(net, device_ids=device_ids)
```

似乎只要加上这一行代码，你在ternimal下执行`watch -n 1 nvidia-smi`后会发现确实会使用多个GPU来并行训练。但是细心点会发现其实第一块卡的显存会占用的更多一些，那么这是什么原因导致的？查阅pytorch官网的nn.DataParrallel相关资料，首先我们来看下其定义如下：

```python
CLASS torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```

其中包含三个主要的参数：module，device_ids和output_device。官方的解释如下：

![img](https://pic3.zhimg.com/80/v2-b0d4a0d85e54644cf10b9eb56c16018a_720w.jpg)

module即表示你定义的模型，device_ids表示你训练的device，output_device这个参数表示输出结果的device，而这最后一个参数output_device一般情况下是省略不写的，那么默认就是在device_ids[0]，也就是第一块卡上，也就解释了为什么第一块卡的显存会占用的比其他卡要更多一些。进一步说也就是当你调用nn.DataParallel的时候，只是在你的input数据是并行的，但是你的output loss却不是这样的，每次都会在第一块GPU相加计算，这就造成了第一块GPU的负载远远大于剩余其他的显卡。

下面来具体讲讲nn.DataParallel中是怎么做的。

首先在前向过程中，你的输入数据会被划分成多个子部分（以下称为副本）送到不同的device中进行计算，而你的模型module是在每个device上进行复制一份，也就是说，输入的batch是会被平均分到每个device中去，但是你的模型module是要拷贝到每个devide中去的，每个模型module只需要处理每个副本即可，当然你要保证你的batch size大于你的gpu个数。然后在反向传播过程中，每个副本的梯度被累加到原始模块中。概括来说就是：**DataParallel 会自动帮我们将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并在原GPU上汇总**。

注意还有一句话，官网中是这样描述的：

> The parallelized `module` must have its parameters and buffers on `device_ids[0]` before running this `DataParallel` module.

意思就是：在运行此DataParallel模块之前，并行化模块必须在device_ids [0]上具有其参数和缓冲区。在执行DataParallel之前，会首先把其模型的参数放在device_ids[0]上，一看好像也没有什么毛病，其实有个小坑。我举个例子，服务器是八卡的服务器，刚好前面序号是0的卡被别人占用着，于是你只能用其他的卡来，比如你用2和3号卡，如果你直接指定device_ids=[2, 3]的话会出现模型初始化错误，类似于module没有复制到在device_ids[0]上去。那么你需要在运行train之前需要添加如下两句话指定程序可见的devices，如下：

```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
```

**当你添加这两行代码后，那么device_ids[0]默认的就是第2号卡**，你的模型也会初始化在第2号卡上了，而不会占用第0号卡了。这里简单说一下设置上面两行代码后，那么对这个程序而言可见的只有2和3号卡，和其他的卡没有关系，这是物理上的号卡，逻辑上来说其实是对应0和1号卡，即device_ids[0]对应的就是第2号卡，device_ids[1]对应的就是第3号卡。（当然你要保证上面这两行代码需要定义在

```python
device_ids = [0, 1]
net = torch.nn.DataParallel(net, device_ids=device_ids)
```

这两行代码之前，一般放在train.py中import一些package之后。）

那么在训练过程中，你的优化器同样可以使用nn.DataParallel，如下两行代码：

```python
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
```

那么使用nn.DataParallel后，事实上DataParallel也是一个Pytorch的nn.Module，那么你的模型和优化器都需要使用.module来得到实际的模型和优化器，如下：

```python
保存模型：
torch.save(net.module.state_dict(), path)
加载模型：
net=nn.DataParallel(Resnet18())
net.load_state_dict(torch.load(path))
net=net.module
优化器使用：
optimizer.step() --> optimizer.module.step()
```

还有一个问题就是，如果直接使用nn.DataParallel的时候，训练采用多卡训练，会出现一个warning：

```
UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; 
will instead unsqueeze and return a vector.
```

首先说明一下：每张卡上的loss都是要汇总到第0张卡上求梯度，更新好以后把权重分发到其余卡。但是为什么会出现这个warning，这其实和nn.DataParallel中最后一个参数dim有关，其表示tensors被分散的维度，默认是0，nn.DataParallel将在dim0（批处理维度）中对数据进行分块，并将每个分块发送到相应的设备。单卡的没有这个warning，多卡的时候采用nn.DataParallel训练会出现这个warning，由于计算loss的时候是分别在多卡计算的，那么返回的也就是多个loss，你使用了多少个gpu，就会返回多少个loss。（有人建议DataParallel类应该有reduce和size_average参数，比如用于聚合输出的不同loss函数，最终返回一个向量，有多少个gpu，返回的向量就有几维。） 	

关于这个问题在pytorch官网的issues上有过讨论，下面简单摘出一些。

[DataParallel does not work with tensors of dimension 0 · Issue #9811 · pytorch/pytorchgithub.com![图标](https://pic4.zhimg.com/v2-2b69048deeb1ce5914dc891d7aa149e7_ipico.jpg)](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/issues/9811)

前期探讨中，有人提出求loss平均的方式会在不同数量的gpu上训练会以微妙的方式影响结果。模块返回该batch中所有损失的平均值，如果在4个gpu上运行，将返回4个平均值的向量。然后取这个向量的平均值。但是，如果在3个GPU或单个GPU上运行，这将不是同一个数字，因为每个GPU处理的batch size不同！举个简单的例子（就直接摘原文出来）：

A batch of 3 would be calculated on a single GPU and results would be [0.3, 0.2, 0.8] and model that returns the loss would return 0.43.

If cast to DataParallel, and calculated on 2 GPUs, [GPU1 - batch 0,1], [GPU2 - batch 2] - return values would be [0.25, 0.8] (0.25 is average between 0.2 and 0.3)- taking the average loss of [0.25, 0.8] is now 0.525!

Calculating on 3 GPUs, one gets [0.3, 0.2, 0.8] as results and average is back to 0.43!

似乎一看，这么求平均loss确实有不合理的地方。那么有什么好的解决办法呢，可以使用size_average=False，reduce=True作为参数。每个GPU上的损失将相加，但不除以GPU上的批大小。然后将所有平行损耗相加，除以整批的大小，那么不管几块GPU最终得到的平均loss都是一样的。

那pytorch贡献者也实现了这个loss求平均的功能，即通过gather的方式来求loss平均：

[Support modules that output scalar in Gather (and data parallel) by SsnL · Pull Request #7973 · pytorch/pytorchgithub.com![图标](https://pic3.zhimg.com/v2-a05709e4e42564048ea856b479b44752_ipico.jpg)](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/pull/7973/commits/c285b3626a7a4dcbbddfba1a6b217a64a3f3f3be)

如果它们在一个有2个GPU的系统上运行，DP将采用多GPU路径，调用gather并返回一个向量。如果运行时有1个GPU可见，DP将采用顺序路径，完全忽略gather，因为这是不必要的，并返回一个标量。

如果它们在一个有2个GPU的系统上运行，DP将采用多GPU路径，调用gather并返回一个向量。如果运行时有1个GPU可见，DP将采用顺序路径，完全忽略gather，因为这是不必要的，并返回一个标量。

![img](https://img-blog.csdnimg.cn/20190213155646420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDA4NzU3OA==,size_16,color_FFFFFF,t_70)

# [Pytorch中的Distributed Data Parallel与混合精度训练（Apex）](https://zhuanlan.zhihu.com/p/105755472)

之前我在并行训练的时候一直用的是DataParallel，而不管是同门师兄弟还是其他大佬一直推荐Distributed DataParallel。前两天改代码的时候我终于碰到坑了，各种原因导致单进程多卡的时候只有一张卡在进行运算。痛定思痛，该学习一下传说中的分布式并行了。

基本上是一篇教程的翻译，原文链接：

[Distributed data parallel training in Pytorchyangkky.github.io![图标](https://pic1.zhimg.com/v2-657a02653526c6d0ffb2b244c0333388_180x120.jpg)](https://link.zhihu.com/?target=https%3A//yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)

后续等我把这些并行计算的内容捋清楚了，会再自己写一份更详细的tutorial~

**注意**：需要在每一个进程设置相同的随机种子，以便所有模型权重都初始化为相同的值。

## 1. **动机**

加速神经网络训练最简单的办法就是上GPU，如果一块GPU还是不够，就多上几块。

事实上，比如[BERT](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.04805)和[GPT-2](https://link.zhihu.com/?target=https%3A//d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)这样的大型语言模型甚至是在上百块GPU上训练的。

为了实现多GPU训练，我们必须想一个办法在多个GPU上分发数据和模型，并且协调训练过程。

## **2. Why Distributed Data Parallel？**

Pytorch兼顾了主要神经网络结构的易用性和可控性。而其提供了两种办法在多GPU上分割数据和模型：即

[nn.DataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23dataparallel) 以及 [nn.DistributedDataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23distributeddataparallel)。

[nn.DataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23dataparallel) 使用起来更加简单（通常只要封装模型然后跑训练代码就ok了）。但是在每个训练批次（batch）中，因为模型的权重都是在 一个进程上先算出来 然后再把他们分发到每个GPU上，**所以网络通信就成为了一个瓶颈，而GPU使用率也通常很低**。

除此之外，[nn.DataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23dataparallel) 需要所有的GPU都在一个节点（一台机器）上，且**并不支持** [Apex](https://link.zhihu.com/?target=https%3A//nvidia.github.io/apex/amp.html) 的 [混合精度训练](https://link.zhihu.com/?target=https%3A//devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/).

## 3. **现有文档的局限性**

总的来说，Pytorch的文档是全面且清晰的，特别是在1.0版本的那些。完全通过文档和教程就可以自学Pytorch，这并不是显示一个人有多大佬，而显然更多地反映了Pytorch的易用性和优秀的文档。

但是好巧不巧的，就是在（Distributed）DataParallel这个系列的文档讲的就不甚清楚，或者干脆没有/不完善/有很多无关内容。以下是一些例子（抱怨）。

- - Pytorch提供了一个使用AWS（亚马逊网络服务）进行分布式训练的[教程](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html)，这个教程在教你如何使用AWS方面很出色，但甚至没提到 [nn.DistributedDataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23distributeddataparallel) 是干什么用的，这导致相关的代码块很难follow。
  - 而[另外一篇Pytorch提供的教程](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/intermediate/dist_tuto.html)又太细了，它对于一个不是很懂Python中MultiProcessing的人（比如我）来说很难读懂。因为它花了大量的篇幅讲 [nn.DistributedDataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23distributeddataparallel) 中的复制功能（数据是怎么复制的）。然而，他并没有在高层逻辑上总结一下都在扯啥，甚至没说这个DistributedDataParallel是*咋用*的？
  - 这里还有一个[Pytorch关于入门分布式数据并行的(Distributed data parallel)教程](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/intermediate/ddp_tutorial.html)。这个教程展示了如何进行一些设置，但并没解释这些设置是干啥用的，之后也展示了一些讲模型分到各个GPU上并执行一个优化步骤（optimization step）。然而，这篇教程里的代码是跑不同的(函数名字都对不上)，也没告诉你*怎么跑这个代码*。和之前的教程一样，他也没给一个逻辑上分布式训练的工作概括。
  - 而官方给的最好的例子，无疑是[ImageNet](https://link.zhihu.com/?target=https%3A//github.com/pytorch/examples/tree/master/imagenet)的训练，然而因为这个例子要 素 过 多，导致也看不出来哪个部分是用于分布式多GPU训练的。
  - Apex提供了他们自己的[ImageNet的训练例](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/apex/tree/master/examples/imagenet)。例子的文档告诉大家他们的 nn.DistributedDataParallel 是自己重写的，但是如果连最初的版本都不会用，更别说重写的了。
  - 而这个[教程](https://link.zhihu.com/?target=http%3A//www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)很好地描述了在底层， nn.DistributedDataParallel 和 nn.DataParallel 到底有什么不同。然而他并没有如何使用 nn.DataParallel 的例程。

## **4. 大纲**

本教程实际上是针对那些**已经熟悉在Pytorch中训练神经网络模型**的人的，本文不会详细介绍这些代码的任何一部分。

本文将首先概述一下总体情况，然后展示一个最小的使用GPU训练MNIST数据集的例程。之后对这个例程进行修改，以便在多个gpu(可能跨多个节点)上进行训练，并逐行解释这些更改。重要的是，本文还将**解释如何运行代码**。

另外，本文还演示了如何使用Apex进行简单的**混合精度分布式训练**。

## **5.大图景（The big picture）**

使用 nn.DistributedDataParallel 进行Multiprocessing可以在多个gpu之间复制该模型，每个gpu由一个进程控制。（如果你想，也可以一个进程控制多个GPU，但这会比控制一个慢得多。也有可能有多个工作进程为每个GPU获取数据，但为了简单起见，本文将省略这一点。）这些GPU可以位于同一个节点上，也可以分布在多个节点上。每个进程都执行相同的任务，并且每个进程与所有其他进程通信。

**只有梯度会在进程/GPU之间传播，这样网络通信就不至于成为一个瓶颈了。**

![img](https://pic1.zhimg.com/80/v2-657a02653526c6d0ffb2b244c0333388_720w.jpg)

训练过程中，每个进程从磁盘加载自己的小批（minibatch）数据，并将它们传递给自己的GPU。每个GPU都做它自己的前向计算，然后梯度在GPU之间全部约简。每个层的梯度不依赖于前一层，因此梯度全约简与并行计算反向传播，进一步缓解网络瓶颈。在反向传播结束时，每个节点都有平均的梯度，确保模型权值保持同步（synchronized）。

上述的步骤要求需要多个进程，甚至可能是不同结点上的多个进程同步和通信。而Pytorch通过它的 [distributed.init_process_group](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/distributed.html%23initialization) 函数实现。这个函数需要知道如何找到进程0（process 0），一边所有的进程都可以同步，也知道了一共要同步多少进程。每个独立的进程也要知道总共的进程数，以及自己在所有进程中的阶序（rank）,当然也要知道自己要用那张GPU。总进程数称之为 world size。最后，每个进程都需要知道要处理的数据的哪一部分，这样批处理就不会重叠。而Pytorch通过 [nn.utils.data.DistributedSampler](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html) 来实现这种效果。

## **6. 最小例程与解释**

为了展示如何做到这些，这里有一个在[MNIST上训练的例子](https://link.zhihu.com/?target=https%3A//github.com/yangkky/distributed_tutorial/blob/master/src/mnist.py)，并且之后把它修改为可以[在多节点多GPU上运行](https://link.zhihu.com/?target=https%3A//github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py)，最终修改的版本还可以[支持混合精度运算](https://link.zhihu.com/?target=https%3A//github.com/yangkky/distributed_tutorial/blob/master/src/mnist-mixed.py)。

首先，我们import所有我们需要的库

```python
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
```

之后，我们训练了一个MNIST分类的简单卷积网络

```python
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```

这个 main() 函数会接受一些参数并运行训练函数。

```python
 def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    train(0, args)
```

而这部分则是训练函数

```python
 def train(gpu, args):
	torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
```

最后，我们要确保 main() 函数会被调用

```python
if __name__ == '__main__':
    main()
```

上述代码中肯定有一些我们还不需要的额外的东西(例如gpu和节点的数量)，但是将整个框架放置到位是很有帮助的。之后在命令行输入

```python
python src/mnist.py -n 1 -g 1 -nr 0
```

就可以在一个结点上的单个GPU上训练啦~

## **7. 加上MultiProcessing**

我们需要一个脚本，用来启动一个进程的每一个GPU。每个进程需要知道使用哪个GPU，以及它在所有正在运行的进程中的阶序（rank）。而且，我们需要**在每个节点上运行脚本**。

现在让我们康康每个函数的变化，这些改变将被单独框出方便查找。

```python
 def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = '10.57.23.164'              #
    os.environ['MASTER_PORT'] = '8888'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################
```

上一节中一些参数在这个地方才需要

- args.nodes 是我们使用的结点数
- args.gpus 是每个结点的GPU数
- args.nr 是当前结点的阶序rank，这个值的取值范围是 0 到 args.nodes - 1.

OK，现在我们一行行看都改了什么

- Line 14：基于结点数以及每个结点的GPU数，我们可以计算 world_size 或者需要运行的总进程数，这和总GPU数相等。
- Line 15：告诉Multiprocessing模块去哪个IP地址找process 0以确保初始同步所有进程。
- Line 16：同样的，这个是process 0所在的端口
- Line 17：现在，我们需要生成 args.gpus 个进程, 每个进程都运行 train(i, args), 其中 i 从 0 到 args.gpus - 1。注意, main() 在每个结点上都运行, 因此总共就有 args.nodes * args.gpus = args.world_size 个进程.

除了14，15行的设置，也可以在终端中运行

```
export MASTER_ADDR=10.57.23.164 和 export MASTER_PORT=8888
```

接下来，需要修改的就是训练函数了，改动的地方依然被框出来啦。

```python
def train(gpu, args):
    ############################################################
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    ############################################################
    
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    ###############################################################

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )                                               
    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    ################################################################

    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=0,
       pin_memory=True,
    #############################
      sampler=train_sampler)    # 
    #############################
    ...
```

为了简单起见，上面的代码去掉了简单循环并用 ... 代替，不过你可以在[这里看到完整脚本](https://link.zhihu.com/?target=https%3A//github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py) 。

- Line3：这里是该进程在所有进程中的全局rank（一个进程对应一个GPU）。这个rank在Line6会用到
- Line4~6：初始化进程并加入其他进程。这就叫做“blocking”，也就是说只有当所有进程都加入了,单个进程才会运行。这里使用了 nccl 后端，因为[Pytorch文档](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/distributed.html)说它是跑得最快的。 init_method 让进程组知道去哪里找到它需要的设置。在这里，它就在寻找名为 MASTER_ADDR 以及 MASTER_PORT 的环境变量，这些环境变量在 main 函数中设置过。当然，本来可以把world_size 设置成一个全局变量，不过本脚本选择把它作为一个关键字参量（和当前进程的全局阶序global rank一样）
- Line23： 将模型封装为一个 [DistributedDataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23distributeddataparallel) 模型。这将把模型复制到GPU上进行处理。
- Line35~39： [nn.utils.data.DistributedSampler](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html) 确保每个进程拿到的都是不同的训练数据切片。
- Line46/Line51：因为用了 [nn.utils.data.DistributedSampler](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html) 所以不能用正常的办法做shuffle。

要在4个节点上运行它(每个节点上有8个gpu)，我们需要4个终端(每个节点上有一个)。在节点0上(由 main 中的第13行设置)：

python src/mnist-distributed.py -n 4 -g 8 -nr 0

而在其他的节点上：

python src/mnist-distributed.py -n 4 -g 8 -nr i

其中 i∈1,2,3. 换句话说，我们要把这个脚本在每个结点上运行脚本，让脚本运行 args.gpus 个进程以在训练开始之前同步每个进程。

注意，脚本中的batchsize设置的是每个GPU的batchsize，因此实际的batchsize要乘上总共的GPU数目（worldsize）。

## **8. 使用Apex进行混合混合精度训练**

混合精度训练，即组合浮点数 (FP32)和半精度浮点数 (FP16)进行训练，允许我们使用更大的batchsize，并利用[NVIDIA张量核](https://link.zhihu.com/?target=https%3A//www.nvidia.com/en-us/data-center/tensorcore/)进行更快的计算。AWS [p3](https://link.zhihu.com/?target=https%3A//aws.amazon.com/cn/ec2/instance-types/p3/)实例使用了8块带张量核的NVIDIA Tesla V100 GPU。

我们只需要修改 train 函数即可，为了简便表示，下面已经从示例中剔除了数据加载代码和反向传播之后的代码，并将它们替换为 ... ，不过你可以在[这看到完整脚本](https://link.zhihu.com/?target=https%3A//github.com/yangkky/distributed_tutorial/blob/master/src/mnist-mixed.py)。

```python
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)
        
	torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    ##############################################################
    model, optimizer = amp.initialize(model, optimizer, 
                                      opt_level='O2')
    model = DDP(model)
    ##############################################################
    # Data loading code
	...
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
    ##############################################################
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
    ##############################################################
            optimizer.step()
     ...
```

- Line18： [amp.initialize](https://link.zhihu.com/?target=https%3A//nvidia.github.io/apex/amp.html%23unified-api) 将模型和优化器为了进行后续混合精度训练而进行封装。注意，在调用 [amp.initialize](https://link.zhihu.com/?target=https%3A//nvidia.github.io/apex/amp.html%23unified-api) 之前，模型必须已经部署在GPU上。 opt_level 从 O0 （全部使用浮点数）一直到 O3 （全部使用半精度浮点数）。而 O1 和 O2 属于不同的混合精度程度，具体可以参阅[APEX的官方文档](https://link.zhihu.com/?target=https%3A//nvidia.github.io/apex/amp.html%23opt-levels-and-properties)。注意之前数字前面的是大写字母O。
- Line20：[apex.parallel.DistributedDataParallel](https://link.zhihu.com/?target=https%3A//nvidia.github.io/apex/parallel.html) 是一个 nn.DistributedDataParallel 的替换版本。我们不需要指定GPU，因为Apex在一个进程中只允许用一个GPU。且它也假设程序在把模型搬到GPU之前已经调用了 torch.cuda.set_device(local_rank)(line 10) .
- Line37-38：混合精度训练需要[缩放](https://link.zhihu.com/?target=https%3A//devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)损失函数以阻止梯度出现下溢。不过Apex会自动进行这些工作。

这个脚本和之前的分布式训练脚本的运行方式相同。



# [Pytorch自动混合精度(AMP)介绍与使用](https://www.cnblogs.com/jimchen1218/p/14315008.html)

背景：

pytorch从1.6版本开始，已经内置了torch.cuda.amp，采用自动混合精度训练就不需要加载第三方NVIDIA的apex库了。本文主要从三个方面来介绍AMP：

一．什么是AMP?

二．为什么要使用AMP？

三．如何使用AMP?

四. 注意事项

 

正文：

　**一．什么是AMP?**

   默认情况下，大多数深度学习框架都采用32位浮点算法进行训练。２０１７年，NVIDIA研究了一种用于混合精度训练的方法，该方法在训练网络时将单精度（FP32）与半精度(FP16)结合在一起，并使用相同的超参数实现了与FP32几乎相同的精度。

在介绍AMP之前，先来理解下FP16与FP32，FP16也即半精度是一种计算机使用的二进制浮点数据类型，使用2字节存储。而FLOAT就是FP32。

![img](https://img2020.cnblogs.com/blog/1781642/202101/1781642-20210122192609433-1423562086.png)

其中，sign位表示正负，exponent位表示指数2^(n-15+1(n=0))，fraction位表示分数(m/1024)。

![img](https://img2020.cnblogs.com/blog/1781642/202101/1781642-20210122194232572-758719678.png)

一般情况下，我们在pytorch中创建一个Tensor:

```python
>>import torch
>>tensor1=torch.zeros(30,20)
>>tensor1.type()
'torch.FloatTensor'

>>tensor2=torch.Tensor([1,2])
>>tensor2.type()

'torch.FlatTensor'
```

可以看到，默认创建的tensor都是FloatTensor类型。而在Pytorch中，一共有10种类型的tensor:

```python
torch.FloatTensor(32bit floating point)
torch.DoubleTensor(64bit floating point)
torch.HalfTensor(16bit floating piont1)
torch.BFloat16Tensor(16bit floating piont2)
torch.ByteTensor(8bit integer(unsigned)
torch.CharTensor(8bit integer(signed))
torch.ShortTensor(16bit integer(signed))
torch.IntTensor(32bit integer(signed))
torch.LongTensor(64bit integer(signed))
torch.BoolTensor(Boolean)

默认Tensor是32bit floating point，这就是32位浮点型精度的tensor。
```

AMP(自动混合精度）的关键词有两个：自动，混合精度。

自动：Tensor的dtype类型会自动变化，框架按需自动调整tensor的dtype,当然有些地方还需手动干预。

混合精度：采用不止一种精度的Tensor，torch.FloatTensor和torch.HalfTensor

pytorch1.6的新包：torch.cuda.amp，是ＮVIDIA开发人员贡献到pytorch里的。只有支持tensor core的CUDA硬件才能享受到AMP带来的优势。Tensor core是一种矩阵乘累加的计算单元，每个tensor core时针执行64个浮点混合精度操作（FP16矩阵相乘和FP32累加）。

**二、为什么要使用AMP?**

前面已介绍，AMP其实就是Float32与Float16的混合，那为什么不单独使用Float32或Float16，而是两种类型混合呢？原因是：在某些情况下Float32有优势，而在另外一些情况下Float16有优势。这里先介绍下FP16：

　　优势有三个：

　１．减少显存占用；

　２．加快训练和推断的计算，能带来多一倍速的体验；

　３．张量核心的普及（NVIDIA　Tensor Core）,低精度计算是未来深度学习的一个重要趋势。

　　但凡事都有两面性，FP16也带来了些问题：１．溢出错误；２．舍入误差；

１．溢出错误：由于FP16的动态范围比FP32位的狭窄很多，因此，在计算过程中很容易出现上溢出和下溢出，溢出之后就会出现"NaN"的问题。在深度学习中，由于激活函数的梯度往往要比权重梯度小，更易出现下溢出的情况

![img](https://img2020.cnblogs.com/blog/1781642/202101/1781642-20210122195633442-1664737627.png)

2.舍入误差

　舍入误差指的是当梯度过小时，小于当前区间内的最小间隔时，该次梯度更新可能会失败：

![img](https://img2020.cnblogs.com/blog/1781642/202101/1781642-20210122195858319-733457410.png)

　为了消除torch.HalfTensor也就是FP16的问题，需要使用以下两种方法：

１）混合精度训练

　在内存中用FP16做储存和乘法从而加速计算，而用FP32做累加避免舍入误差。混合精度训练的策略有效地缓解了舍入误差的问题。

　什么时候用torch.FloatTensor,什么时候用torch.HalfTensor呢？这是由pytorch框架决定的，在pytorch1.6的AMP上下文中，以下操作中Tensor会被自动转化为半精度浮点型torch.HalfTensor：

```python
__matmul__
addbmm
addmm
addmv
addr
baddbmm
bmm
chain_matmul
conv1d
conv2d
conv3d
conv_transpose1d
conv_transpose2d
conv_transpose3d
linear
matmul
mm
mv
prelu
```

2)损失放大（Loss scaling)

即使了混合精度训练，还是存在无法收敛的情况，原因是激活梯度的值太小，造成了溢出。可以通过使用torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度的underflow（只在BP时传递梯度信息使用，真正更新权重时还是要把放大的梯度再unscale回去）；

反向传播前，将损失变化手动增大2^k倍，因此反向传播时得到的中间变量（激活函数梯度）则不会溢出；

反向传播后，将权重梯度缩小2^k倍，恢复正常值。

**三．如何使用AMP?**

　目前有两种版本：pytorch1.5之前使用的NVIDIA的三方包apex.amp和pytorch1.6自带的torch.cuda.amp

１．pytorch1.5之前的版本（包括１．５）

 使用方法如下：

```python
from apex import amp
model,optimizer = amp.initial(model,optimizer,opt_level="O1")   #注意是O,不是０
with amp.scale_loss(loss,optimizer) as scaled_loss:
    scaled_loss.backward()取代loss.backward()
```

其中,opt_level配置如下：

　O0:纯FP32训练，可作为accuracy的baseline；

　O1:混合精度训练（推荐使用），根据黑白名单自动决定使用FP16(GEMM,卷积）还是FP32（softmax)进行计算。

　O2:几乎FP16，混合精度训练，不存在黑白名单　，除了bacthnorm，几乎都是用FP16计算；

　O3:纯FP16训练，很不稳定，但是可以作为speed的baseline；

动态损失放大（dynamic loss scaling)部分，为了充分利用FP16的范围，缓解舍入误差，尽量使用最高的放大倍数2^24,如果产生上溢出，则跳出参数更新，缩小放大倍数使其不溢出。在一定步数后再尝试使用大的scale来充分利用FP16的范围。

分布式训练：

```python
import argparse
import apex import amp
import apex.parallel import convert_syncbn_model
import apex.parallel import DistributedDataParallel as DDP

定义超参数：
def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local_rank',type=int, default=0)  #local_rank指定了输出设备，默认为GPU可用列表中的第一个GPU，必须加上。
    ...
    args = parser.parser.parse_args()
    return args

主函数写：
def main():
    args = parse()
    torch.cuda.set_device(args.local_rank)  #必须写在下一句的前面
   torch.distributed.init_process_group(
       'nccl',
       init_method='env://')

导入数据接口，需要用DistributedSampler
    dataset = ...
    num_workers = 4 if cuda else 0
    train_sampler=torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batchsize, shuflle=False, num_worker=num_workers,pin_memory=cuda, drop_last=True, sampler=train_sampler)

定义模型：
net = XXXNet(using_amp=True)
net.train()
net= convert_syncbn_model(net)
device=torch.device('cuda:{}'.format(args.local_rank))
net=net.to(device)

定义优化器，损失函数，定义优化器一定要把模型搬运到GPU之上
apt = Adam([{'params':params_low_lr,'lr':4e-5},
    {'params':params_high_lr,'lr':1e-4}],weight_decay=settings.WEIGHT_DECAY)
crit = nn.BCELoss().to(device)

多GPU设置import torch.nn.parallel.DistributedDataParallel as DDP
net,opt = amp.initialize(net,opt,opt_level='o1')
net=DDP(net,delay_allreduce=True)loss使用方法：opt.zero_grad()with amp.scale_loss(loss, opt) as scaled_loss:    scaled_loss.backward()opt.step()加入主入口：if __name__ == '__main__':    main()无论是apex支持的DDP还是pytorch自身支持的DDP,都需使用torch.distributed.launch来使用，方法如下：CUDA_VISIBLE_DIVECES=1,2,4 python -m torch.distributed.launch --nproc_per_node=3 train.py1,2,4是GPU编号，nproc_per_node是指定用了哪些GPU,记得开头说的local_rank，是因为torch.distributed.launch会调用这个local_ran
```

 分布式训练时保存模型注意点：

　如果直接在代码中写torch.save来保存模型，则每个进程都会保存一次相同的模型，会存在写文件写到一半，会被个进程写覆盖的情况。如何避免呢?

  可以用local_rank == 0来仅仅在第一个GPU上执行进程来保存模型文件。

```
虽然是多个进程，但每个进程上模型的参数值都是一样的，而默认代号为０的进程是主进程
if arg.local_rank == 0:
    torch.save(xxx)
```

２．pytorch1.6及以上版本

　　有两个接口：autocast和Gradscaler

 1) autocast

  导入pytorch中模块torch.cuda.amp的类autocast

```python
from torch.cuda.amp import autocast as autocast

model=Net().cuda()
optimizer=optim.SGD(model.parameters(),...)

for input,target in data:
  optimizer.zero_grad()

  with autocast():
    output=model(input)
    loss = loss_fn(output,target)

  loss.backward()
  optimizer.step()
```

　　可以使用autocast的context managers语义（如上），也可以使用decorators语义。当进入autocast上下文后，在这之后的cuda ops会把tensor的数据类型转换为半精度浮点型，从而在不损失训练精度的情况下加快运算。而不需要手动调用.half(),框架会自动完成转换。

　　不过，autocast上下文只能包含网络的前向过程(包括loss的计算），不能包含反向传播，因为BP的op会使用和前向op相同的类型。

　　当然，有时在autocast中的代码会报错：

```python
Traceback (most recent call last):
......
 File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
  result = self.forward(*input, ** kwargs)
......
RuntimeError: expected scalar type float but found c10::Half
```

 对于RuntimeError:expected scaler type float but found c10:Half,应该是个bug,可在tensor上手动调用.float()来让type匹配。

２）GradScaler

　　使用前，需要在训练最开始前实例化一个GradScaler对象，例程如下：

```python
from torch.cuda.amp import autocast as autocast

model=Net().cuda()
optimizer=optim.SGD(model.parameters(),...)

scaler = GradScaler() #训练前实例化一个GradScaler对象

for epoch in epochs:
  for input,target in data:
    optimizer.zero_grad()

    with autocast():　＃前后开启autocast
      output=model(input)
      loss = loss_fn(output,targt)

    scaler.scale(loss).backward()  #为了梯度放大
    #scaler.step()　首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
    scaler.update()  #准备着，看是否要增大scaler
```

　scaler的大小在每次迭代中动态估计，为了尽可能减少梯度underflow，scaler应该更大；但太大，半精度浮点型又容易overflow（变成inf或NaN).所以，动态估计原理就是在不出现if或NaN梯度的情况下，尽可能的增大scaler值。在每次scaler.step(optimizer)中，都会检查是否有inf或NaN的梯度出现：

　　１．如果出现inf或NaN,scaler.step(optimizer)会忽略此次权重更新(optimizer.step()），并将scaler的大小缩小（乘上backoff_factor)；

　　２．如果没有出现inf或NaN,那么权重正常更新，并且当连续多次(growth_interval指定)没有出现inf或NaN，则scaler.update()会将scaler的大小增加(乘上growth_factor)。

对于分布式训练，由于autocast是thread local的，要注意以下情形：

１）torch.nn.DataParallel：

以下代码分布式是不生效的

```python
model = MyModel()
dp_model = nn.DataParallel(model)

with autocast():
    output=dp_model(input)
loss=loss_fn(output)
```

需使用autocast装饰model的forward函数

```python
MyModel(nn.Module):
    @autocast()
    def forward(self, input):
        ...
        
#alternatively
MyModel(nn.Module):
    def forward(self, input):
        with autocast():
            ...


model = MyModel()
dp_model=nn.DataParallel(model)

with autocast():
    output=dp_model(input)
    loss = loss_fn(output)
```

2）torch.nn.DistributedDataParallel:

  同样，对于多GPU,也需要autocast装饰model的forward方法，保证autocast在进程内部生效。

四. 注意事例：

　　在使用AMP时，由于报错信息并不明显，给调试带来了一定的难度。但只要注意以下一些点，相信会少走很多弯路。

１．判断GPU是否支持FP16，支持Tensor core的GPU（2080Ti,Titan,Tesla等），不支持的(Pascal系列）不建议；

1080Ti与2080Ti对比

```python
gtx 1080ti:
半精度浮点数：0.17TFLOPS
单精度浮点数：11.34TFLOPS
双精度浮点数：0.33TFLOPS
rtx 2080ti:
半精度浮点数：20.14TFLOPS
单精度浮点数：10.07TFLOPS
双精度浮点数：0.31TFLOPS
```

半精度浮点数即FP16，单精度浮点数即FP32，双精度浮点数即FP64。
在不使用apex的pytorch训练过程中，一般默认均为单精度浮点数，从上面的数据可以看到1080ti和2080ti的单精度浮点数运算能力差不多，因此不使用apex时用1080ti和2080ti训练模型时间上差别很小。

使用apex时用1个2080ti训练时一个epoch是2h31min，两者时间几乎一样，但是却少用了一张2080ti。这是因为在pytorch训练中使用apex时，此时大多数运算均为半精度浮点数运算，而2080ti的半精度浮点数运算能力是其单精度浮点数运算能力的两倍

２．常数范围：为了保证计算不溢出，首先保证人工设定的常数不溢出。如epsilon,INF等；

３．Dimension最好是8的倍数:维度是８的倍数，性能最好；

４．涉及sum的操作要小心，容易溢出，softmax操作，建议用官方API，并定义成layer写在模型初始化里；

５．模型书写要规范：自定义的Layer写在模型初始化函数里，graph计算写在forward里；

６．一些不常用的函数，使用前要注册：amp.register_float_function(torch,'sogmoid')

７．某些函数不支持FP16加速，建议不要用；

８．需要操作梯度的模块必须在optimizer的step里，不然AMP不能判断grad是否为NaN