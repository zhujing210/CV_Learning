## 【深度学习】翻译：60分钟入门PyTorch（一）——Tensors

## 前言

原文翻译自：Deep Learning with PyTorch: A 60 Minute Blitz

翻译：林不清（https://www.zhihu.com/people/lu-guo-92-42-88）

## 目录

60分钟入门PyTorch（一）——Tensors

60分钟入门PyTorch（二）——Autograd自动求导

60分钟入门Pytorch（三）——神经网络

60分钟入门PyTorch（四）——训练一个分类器

## Tensors

Tensors张量是一种特殊的数据结构，它和数组还有矩阵十分相似。在Pytorch中，我们使用tensors来给模型的输入输出以及参数进行编码。Tensors除了张量可以在gpu或其他专用硬件上运行来加速计算之外，其他用法类似于Numpy中的ndarrays。如果你熟悉ndarrays，您就会熟悉tensor的API。如果没有，请按照这个教程，快速了解一遍API。

```python
%matplotlib inline
import torch
import numpy as np
```

### 初始化Tensor

创建Tensor有多种方法，如：

#### 直接从数据创建

可以直接利用数据创建tensor,数据类型会被自动推断出

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

#### 从Numpy创建

Tensor 可以直接从numpy的array创建（反之亦然-参见`bridge-to-np-label`）

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

#### 从其他tensor创建

新的tensor保留了参数tensor的一些属性（形状，数据类型），除非显式覆盖

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

```

Ones Tensor: 
 tensor([[1, 1],
        [1, 1]]) 

Random Tensor: 
 tensor([[0.6075, 0.4581],
        [0.5631, 0.1357]])

#### 从常数或者随机数创建

`shape`是关于tensor维度的一个元组，在下面的函数中，它决定了输出tensor的维数。

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

```

Random Tensor: 
 tensor([[0.7488, 0.0891, 0.8417],
        [0.0783, 0.5984, 0.5709]]) 

Ones Tensor: 
 tensor([[1., 1., 1.],
        [1., 1., 1.]]) 

Zeros Tensor: 
 tensor([[0., 0., 0.],
        [0., 0., 0.]])

### Tensor的属性

Tensor的属性包括形状，数据类型以及存储的设备

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on : {tensor.device}")

```

Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu

### Tensor的操作

Tensor有超过100个操作，包括 transposing, indexing, slicing, mathematical operations, linear algebra, random sampling,更多详细的介绍请点击[这里](https://pytorch.org/docs/stable/torch.html)

它们都可以在GPU上运行（速度通常比CPU快），如果你使用的是Colab，通过编辑>笔记本设置来分配一个GPU。

```python
# we move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
```

尝试列表中的一些操作。如果你熟悉NumPy API，你会发现tensor的API很容易使用。

### **标准的numpy类索引和切片:**

```python
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)
```

```text
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

### **合并tensors**

可以使用`torch.cat`来沿着特定维数连接一系列张量。 `**[torch.stack](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.stack.html)**`另一个加入op的张量与`torch.cat`有细微的不同

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```text
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

### **tensors乘法**

```python
# This compute the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
```

```text
tensor.mul(tensor) 
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

tensor * tensor 
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

它计算两个tensor之间的**矩阵乘法**	

```python
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
```

```text
tensor.matmul(tensor.T) 
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]]) 

tensor @ tensor.T 
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
```

### **原地操作**

带有后缀`_`的操作表示的是原地操作，例如： `x.copy_(y)`, `x.t_()`将改变 `x`.

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

**注意**
原地操作虽然会节省许多空间，但是由于会立刻清除历史记录所以在计算导数时可能会有问题，因此不建议使用

### **Tensor转换为Numpy 数组**

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

```text
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

tensor的变化反映在NumPy数组中。

```python
t.add_(1)
print(f"t: {t})
print(f"n: {n}")
```

```text
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### **Numpy数组转换为Tensor**

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

NumPy数组的变化反映在tensor中

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```text
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

# 60分钟入门PyTorch（二）——Autograd自动求导

## **Autograd：自动求导**

torch.autograd是pytorch自动求导的工具，也是所有神经网络的核心。我们首先先简单了解一下这个包如何训练神经网络。

**背景介绍**

神经网络(NNs)是作用在输入数据上的一系列嵌套函数的集合，这些函数由权重和误差来定义，被存储在PyTorch中的tensors中。 神经网络训练的两个步骤： 前向传播：在前向传播中，神经网络通过将接收到的数据与每一层对应的权重和误差进行运算来对正确的输出做出最好的预测。 反向传播：在反向传播中，神经网络调整其参数使得其与输出误差成比例。反向传播基于梯度下降策略，是链式求导法则的一个应用，以目标的负梯度方向对参数进行调整。 更加详细的介绍可以参照下述地址： **[3Blue1Brown](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DtIeHLnjs5U8)**

**Pytorch应用**

来看一个简单的示例，我们从torchvision加载一个预先训练好的resnet18模型，接着创建一个随机数据tensor来表示一有3个通道、高度和宽度为64的图像，其对应的标签初始化为一些随机值。

```python
%matplotlib inline

import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
```

接下来，我们将输入数据向输出方向传播到模型的每一层中来预测输出，这就是前向传播。

```python
prediction = model(data)  #  前向传播
```

我们利用模型的预测输出和对应的权重来计算误差，然后反向传播误差。完成计算后，您可以调用.backward()并自动计算所有梯度。此张量的梯度将累积到.grad属性中。

```python
loss = (prediction - labels).sum()
loss.backward()  #  反向传播
```

接着，我们加载一个优化器，在本例中，SGD的学习率为0.01，momentum 为0.9。我们在优化器中注册模型的所有参数。

```python
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

最后，我们调用`.step()`来**执行**梯度下降，优化器通过存储在`.grad`中的梯度来调整每个参数。

```python
optim.step()  #  梯度下降
```

现在，你已经具备了训练神经网络所需所有条件。下面几节详细介绍了Autograd包的工作原理——可以跳过它们。

**Autograd中的求导**

先来看一下`autograd`是如何收集梯度的。我们创建两个张量a和b并设置requires_grad = True以跟踪它的计算。

```python
import torch
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
```

接着在`a`和`b`的基础上创建张量`Q`
$$
Q = 3a^3 - b^2
$$

```python
Q = 3*a**3 - b**2
```

假设`a`和`b`是一个神经网络的权重，`Q`是它的误差，在神经网络训练中，我们需要w.r.t参数的误差梯度，即


![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D%5Cfrac%7B%5Cpartial+Q%7D%7B%5Cpartial+a%7D+%3D+9a%5E2%5Cend%7Balign%7D%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D%5Cfrac%7B%5Cpartial+Q%7D%7B%5Cpartial+b%7D+%3D+-2b%5Cend%7Balign%7D%5C%5C)

当我们调用`Q`的`.backward()`时，autograd计算这些梯度并把它们存储在张量的 `.grad`属性中。我们需要在`Q.backward()`中显式传递`gradient`，`gradient`是一个与`Q`相同形状的张量，它表示Q w.r.t本身的梯度，即

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D%5Cfrac%7BdQ%7D%7BdQ%7D+%3D+1%5Cend%7Balign%7D%5C%5C)

同样，我们也可以将`Q`聚合为一个标量并隐式向后调用，如`Q.sum().backward()`。

```python
external_grad = torch.tensor([1.,1.])
Q.backward(gradient=external_grad)
```

现在梯度都被存放在`a.grad`和`b.grad`中

```python
# 检查一下存储的梯度是否正确
print(9*a**2 == a.grad)
print(-2*b == b.grad)
```

```text
tensor([True, True])
tensor([True, True])
```

**可选阅读----用autograd进行向量计算**

在数学上，如果你有一个向量值函数 ⃗ = ( ⃗ ) ，则 ⃗ 相对于 ⃗ 的梯度是雅可比矩阵： ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7DJ+%3D+%5Cleft%28%5Cbegin%7Barray%7D%7Bcc%7D+%5Cfrac%7B%5Cpartial+%5Cbf%7By%7D%7D%7B%5Cpartial+x_%7B1%7D%7D+%26+...+%26+%5Cfrac%7B%5Cpartial+%5Cbf%7By%7D%7D%7B%5Cpartial+x_%7Bn%7D%7D+%5Cend%7Barray%7D%5Cright%29+%3D+%5Cleft%28%5Cbegin%7Barray%7D%7Bccc%7D+%5Cfrac%7B%5Cpartial+y_%7B1%7D%7D%7B%5Cpartial+x_%7B1%7D%7D+%26+%5Ccdots+%26+%5Cfrac%7B%5Cpartial+y_%7B1%7D%7D%7B%5Cpartial+x_%7Bn%7D%7D%5C+%5Cvdots+%26+%5Cddots+%26+%5Cvdots%5C+%5Cfrac%7B%5Cpartial+y_%7Bm%7D%7D%7B%5Cpartial+x_%7B1%7D%7D+%26+%5Ccdots+%26+%5Cfrac%7B%5Cpartial+y_%7Bm%7D%7D%7B%5Cpartial+x_%7Bn%7D%7D+%5Cend%7Barray%7D%5Cright%29%5Cend%7Balign%7D%5C%5C)

一般来说，torch.autograd是一个计算雅可比向量积的引擎。 也就是说，给定任何向量 =( 1 2... ) ，计算乘积 ⋅ 。如果 恰好是标量函数的梯度 = ( ⃗ )，即![[公式]](https://www.zhihu.com/equation?tex=v%3D%7B%7B%28%5Cfrac%7B%5Cpartial+l%7D%7B%5Cpartial+%7B%7By%7D_%7B1%7D%7D%7D%5Ccdots+%5Cfrac%7B%5Cpartial+l%7D%7B%5Cpartial+%7B%7By%7D_%7Bm%7D%7D%7D%29%7D%5E%7BT%7D%7D) 然后根据链式法则，雅可比向量乘积将是 相对于 ⃗ 的梯度

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7DJ%5E%7BT%7D%5Ccdot+%5Cvec%7Bv%7D%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bccc%7D+%5Cfrac%7B%5Cpartial+y_%7B1%7D%7D%7B%5Cpartial+x_%7B1%7D%7D+%26+%5Ccdots+%26+%5Cfrac%7B%5Cpartial+y_%7Bm%7D%7D%7B%5Cpartial+x_%7B1%7D%7D%5C+%5Cvdots+%26+%5Cddots+%26+%5Cvdots%5C+%5Cfrac%7B%5Cpartial+y_%7B1%7D%7D%7B%5Cpartial+x_%7Bn%7D%7D+%26+%5Ccdots+%26+%5Cfrac%7B%5Cpartial+y_%7Bm%7D%7D%7B%5Cpartial+x_%7Bn%7D%7D+%5Cend%7Barray%7D%5Cright%29%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D+%5Cfrac%7B%5Cpartial+l%7D%7B%5Cpartial+y_%7B1%7D%7D%5C+%5Cvdots%5C+%5Cfrac%7B%5Cpartial+l%7D%7B%5Cpartial+y_%7Bm%7D%7D+%5Cend%7Barray%7D%5Cright%29%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D+%5Cfrac%7B%5Cpartial+l%7D%7B%5Cpartial+x_%7B1%7D%7D%5C+%5Cvdots%5C+%5Cfrac%7B%5Cpartial+l%7D%7B%5Cpartial+x_%7Bn%7D%7D+%5Cend%7Barray%7D%5Cright%29%5Cend%7Balign%7D%5C%5C)

雅可比向量积的这种特性使得将外部梯度馈送到具有非标量输出的模型中非常方便。`external_grad` 代表![[公式]](https://www.zhihu.com/equation?tex=%5Cvec%7Bv%7D).

**图计算**

从概念上讲，autograd在由**[函数](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/autograd.html%23torch.autograd.Function)**对象组成的有向无环图(DAG)中保存数据(tensor)和所有执行的操作(以及产生的新tensor)的记录，在这个DAG中，叶节点是输入数据，根节点是输出数据，通过从根节点到叶节点跟踪这个图，您可以使用链式法则自动计算梯度。

在前向传播中，autograd同时完成两件事情：

- 运行所请求的操作来计算结果tensor
- 保持DAG中操作的梯度

在反向传播中，当在DAG根节点上调用`.backward()`时，反向传播启动，`autograd`接下来完成：

- 计算每一个`.grad_fn`的梯度
- 将它们累加到各自张量的.grad属性中
- 利用链式法则，一直传播到叶节点

下面是DAG的可视化表示的示例。图中，箭头表示前向传播的方向，节点表示向前传递中每个操作的向后函数。蓝色标记的叶节点代表叶张量 `a`和`b`

![img](https://pic2.zhimg.com/80/v2-c7362e660c9a9dc8233e9b72a69291d5_720w.jpg)

**注意**
DAG在PyTorch中是动态的, 值得注意的是图是重新开始创建的; 在调用每一个.backward()后，autograd开始填充一个新图，这就是能够在模型中使用控制流语句的原因。你可以根据需求在每次迭代时更改形状、大小和操作。

`torch.autograd`追踪所有`requires_grad`为`True`的张量的相关操作。对于不需要梯度的张量，将此属性设置为False将其从梯度计算DAG中排除。 操作的输出张量将需要梯度，即使只有一个输入张量`requires_grad=True`。

```python3
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")
```



```text
Does `a` require gradients? : False
Does `b` require gradients?: True
```

在神经网络中，不计算梯度的参数通常称为冻结参数。如果您事先知道您不需要这些参数的梯度，那么“冻结”部分模型是很有用的(这通过减少autograd计算带来一些性能好处)。 另外一个常见的用法是微调一个**[预训练好的网络](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)**，在微调的过程中，我们冻结大部分模型——通常，只修改分类器来对新的<标签>做出预测,让我们通过一个小示例来演示这一点。与前面一样，我们加载一个预先训练好的resnet18模型，并冻结所有参数。

```python3
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# 冻结网络中所有的参数
for param in model.parameters():
    param.requires_grad = False
```

假设我们想在一个有10个标签的新数据集上微调模型。在resnet中，分类器是最后一个线性层模型`model.fc`。我们可以简单地用一个新的线性层(默认未冻结)代替它作为我们的分类器。

```python3
model.fc = nn.Linear(512, 10)
```

现在除了`model.fc`的参数外，模型的其他参数均被冻结，参与计算的参数是`model.fc`的权值和偏置。

```python3
# 只优化分类器
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
```

注意，尽管我们注册了优化器中所有参数，但唯一参与梯度计算(并因此在梯度下降中更新)的参数是分类器的权值和偏差。 **[torch.no_grad()](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.no_grad.html)**中也具有相同的功能。

**拓展阅读**

- **[就地修改操作以及多线程Autograd](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/notes/autograd.html)**
- **[反向模式autodiff的示例](https://link.zhihu.com/?target=https%3A//colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC)**

# 60分钟入门Pytorch（三）——神经网络

## **神经网络**

可以使用`torch.nn`包来构建神经网络. 你已知道`autograd`包,`nn`包依赖`autograd`包来定义模型并求导.一个`nn.Module`包含各个层和一个`forward(input)`方法,该方法返回`output`.

例如,我们来看一下下面这个分类数字图像的网络.

![img](https://pic3.zhimg.com/80/v2-72793d15afceaba29717b2aaed7cec42_720w.jpg)

他是一个简单的前馈神经网络,它接受一个输入,然后一层接着一层的输入,直到最后得到结果。

神经网络的典型训练过程如下:

- 定义神经网络模型,它有一些可学习的参数(或者权重);
- 在数据集上迭代;
- 通过神经网络处理输入;
- 计算损失(输出结果和正确值的差距大小)
- 将梯度反向传播会网络的参数;
- 更新网络的参数,主要使用如下简单的更新原则:`weight = weight - learning_rate * gradient`

## **定义网络**

我们先定义一个网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(NN.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channnels, 3×3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 120)  #  6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
         return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
net = Net()
print(net)
```

```text
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

你只需定义`forward`函数,`backward`函数(计算梯度)在使用`autograd`时自动为你创建.你可以在`forward`函数中使用`Tensor`的任何操作。

`net.parameters()`返回模型需要学习的参数

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

```text
10
torch.Size([6, 1, 3, 3])
```

构造一个随机的32×32的输入，注意:这个网络(LeNet)期望的输入大小是32×32.如果使用MNIST数据集来训练这个网络,请把图片大小重新调整到32×32.

```python3
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

```text
tensor([[-0.0765,  0.0522,  0.0820,  0.0109,  0.0004,  0.0184,  0.1024,  0.0509,
          0.0917, -0.0164]], grad_fn=<AddmmBackward>)
```

将所有参数的梯度缓存清零,然后进行随机梯度的的反向传播.

```text
net.zero_grad()
out.backward(torch.randn(1, 10))
```

**注意**
torch.nn只支持小批量输入,整个torch.nn包都只支持小批量样本,而不支持单个样本 例如,nn.Conv2d将接受一个4维的张量,每一维分别是![[公式]](https://www.zhihu.com/equation?tex=nSamples%5Ctimes+nChannels%5Ctimes+Height%5Ctimes+Width)(样本数*通道数*高*宽). 如果你有单个样本,只需使用`input.unsqueeze(0)`来添加其它的维数. 在继续之前,我们回顾一下到目前为止见过的所有类.

**回顾**

- `torch.Tensor`-支持自动编程操作（如`backward()`）的多维数组。 同时保持梯度的张量。
- `nn.Module`-神经网络模块.封装参数,移动到GPU上运行,导出,加载等
- `nn.Parameter`-一种张量,当把它赋值给一个`Module`时,被自动的注册为参数.
- `autograd.Function`-实现一个自动求导操作的前向和反向定义, 每个张量操作都会创建至少一个`Function`节点，该节点连接到创建张量并对其历史进行编码的函数。

**现在,我们包含了如下内容:**

- 定义一个神经网络
- 处理输入和调用`backward`

**剩下的内容:**

- 计算损失值
- 更新神经网络的权值

**损失函数**
一个损失函数接受一对(output, target)作为输入(output为网络的输出,target为实际值),计算一个值来估计网络的输出和目标值相差多少。
在nn包中有几种不同的**[损失函数](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/nn.html%23loss-functions%3E)**.一个简单的损失函数是:`nn.MSELoss`,它计算输入和目标之间的均方误差。
例如:

```python3
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```



```text
tensor(1.5801, grad_fn=<MseLossBackward>)
```

现在,你反向跟踪`loss`,使用它的`.grad_fn`属性,你会看到向下面这样的一个计算图: input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view -> linear -> relu -> linear -> relu -> linear -> MSELoss -> loss

所以, 当你调用`loss.backward()`,整个图被区分为损失以及图中所有具有`requires_grad = True`的张量，并且其`.grad` 张量的梯度累积。

为了说明,我们反向跟踪几步:

```python3
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
<MseLossBackward object at 0x0000023193A40E08>
<AddmmBackward object at 0x0000023193A40E48>
<AccumulateGrad object at 0x0000023193A40E08>
```

### **反向传播**

为了反向传播误差,我们所需做的是调用`loss.backward()`.你需要清除已存在的梯度,否则梯度将被累加到已存在的梯度。

现在,我们将调用`loss.backward()`,并查看conv1层的偏置项在反向传播前后的梯度。

```python3
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

```text
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([ 0.0013,  0.0068,  0.0096,  0.0039, -0.0105, -0.0016])
```

现在，我们知道了该如何使用损失函数

### **稍后阅读:**

神经网络包包含了各种用来构成深度神经网络构建块的模块和损失函数,一份完整的文档查看**[这里](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/nn)**

### **唯一剩下的内容:**

- 更新网络的权重

### **更新权重**

实践中最简单的更新规则是随机梯度下降(SGD)．

weight=weight−learning_rate∗gradient

我们可以使用简单的Python代码实现这个规则。

```python3
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

然而,当你使用神经网络是,你想要使用各种不同的更新规则,比如`SGD`,`Nesterov-SGD`,`Adam`, `RMSPROP`等.为了能做到这一点,我们构建了一个包`torch.optim`实现了所有的这些规则.使用他们非常简单：

```text
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```

**注意**

观察如何使用`optimizer.zero_grad()`手动将梯度缓冲区设置为零。 这是因为梯度是反向传播部分中的说明那样是累积的。

# 60分钟入门PyTorch（四）——训练一个分类器

## **训练一个分类器**

你已经学会如何去定义一个神经网络,计算损失值和更新网络的权重。

你现在可能在思考：数据哪里来呢？

## **关于数据**

通常，当你处理图像，文本，音频和视频数据时，你可以使用标准的Python包来加载数据到一个numpy数组中.然后把这个数组转换成`torch.*Tensor`。

- 对于图像,有诸如Pillow,OpenCV包等非常实用
- 对于音频,有诸如scipy和librosa包
- 对于文本,可以用原始Python和Cython来加载,或者使用NLTK和SpaCy 。对于视觉,我们创建了一个`torchvision`包,包含常见数据集的数据加载,比如Imagenet,CIFAR10,MNIST等,和图像转换器,也就是`torchvision.datasets`和`torch.utils.data.DataLoader`。

这提供了巨大的便利,也避免了代码的重复。

在这个教程中,我们使用CIFAR10数据集,它有如下10个类别:’airplane’,’automobile’,’bird’,’cat’,’deer’,’dog’,’frog’,’horse’,’ship’,’truck’。这个数据集中的图像大小为3*32*32,即,3通道,32*32像素。

![img](https://pic3.zhimg.com/80/v2-f2e63531b75c6601431a8e91fb69a6f6_720w.jpg)

## **训练一个图像分类器**

我们将按照下列顺序进行:

- 使用`torchvision`加载和归一化CIFAR10训练集和测试集.
- 定义一个卷积神经网络
- 定义损失函数
- 在训练集上训练网络
- 在测试集上测试网络

### **1. 加载和归一化CIFAR10**

使用`torchvision`加载CIFAR10是非常容易的。

```text
import torch
import torchvision
import torchvision.transforms as transforms
```

torchvision的输出是[0,1]的PILImage图像,我们把它转换为归一化范围为[-1, 1]的张量。

**注意**
如果在Windows上运行时出现BrokenPipeError，尝试将torch.utils.data.DataLoader()的num_worker设置为0。

```text
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#这个过程有点慢，会下载大约340mb图片数据。
```

我们展示一些有趣的训练图像。

```text
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

### **2. 定义一个卷积神经网络**

从之前的神经网络一节复制神经网络代码,并修改为接受3通道图像取代之前的接受单通道图像。

```text
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

### **3. 定义损失函数和优化器**

我们使用交叉熵作为损失函数,使用带动量的随机梯度下降。

```text
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### **4. 训练网络**

这是开始有趣的时刻，我们只需在**数据迭代器上**循环,把数据输入给网络,并优化。

```text
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

保存一下我们的训练模型

```text
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

点击**[这里](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/notes/serialization.html)**查看关于保存模型的详细介绍

### **5. 在测试集上测试网络**

我们在整个训练集上训练了两次网络,但是我们还需要检查网络是否从数据集中学习到东西。

我们通过预测神经网络输出的类别标签并根据实际情况进行检测，如果预测正确,我们把该样本添加到正确预测列表。

第一步，显示测试集中的图片一遍熟悉图片内容。

```text
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

接下来，让我们重新加载我们保存的模型(注意:保存和重新加载模型在这里不是必要的，我们只是为了说明如何这样做)：

```text
net = Net()
net.load_state_dict(torch.load(PATH))
```

现在我们来看看神经网络认为以上图片是什么?

```text
outputs = net(images)
```

输出是10个标签的概率。一个类别的概率越大,神经网络越认为他是这个类别。所以让我们得到最高概率的标签。

```text
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
```

这结果看起来非常的好。

接下来让我们看看网络在整个测试集上的结果如何。

```text
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

结果看起来好于偶然，偶然的正确率为10%,似乎网络学习到了一些东西。

那在什么类上预测较好，什么类预测结果不好呢？

```text
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

接下来干什么?

我们如何在GPU上运行神经网络呢?

### **在GPU上训练**

你是如何把一个Tensor转换GPU上,你就如何把一个神经网络移动到GPU上训练。这个操作会递归遍历有所模块,并将其参数和缓冲区转换为CUDA张量。

```  text
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
#假设我们有一台CUDA的机器，这个操作将显示CUDA设备。
print(device)
```

接下来假设我们有一台CUDA的机器，然后这些方法将递归遍历所有模块并将其参数和缓冲区转换为**CUDA张量**：

```text
net.to(device)
```

请记住，你也必须在每一步中把你的输入和目标值转换到GPU上:

```text
inputs, labels = inputs.to(device), labels.to(device)
```

为什么我们没注意到GPU的速度提升很多?那是因为网络非常的小。

### **实践:**

尝试增加你的网络的宽度(第一个`nn.Conv2d`的第2个参数, 第二个`nn.Conv2d`的第一个参数,他们需要是相同的数字),看看你得到了什么样的加速。

### **实现的目标:**

- 深入了解了PyTorch的张量库和神经网络
- 训练了一个小网络来分类图片

### **在多GPU上训练**

如果你希望使用所有GPU来更大的加快速度,请查看选读:**[数据并行](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)**

### **下来做什么?**

- 训练神经网络玩电子游戏
- 在ImageNet上训练最好的ResNet
- 使用对抗生成网络来训练一个人脸生成器
- 使用LSTM网络训练一个字符级的语言模型
- 更多示例
- 更多教程
- 在论坛上讨论PyTorch
- 在Slack上与其他用户聊天