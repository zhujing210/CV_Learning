

# [基本使用](https://zhuanlan.zhihu.com/p/56922793)

argsparse是python的命令行解析的标准模块，内置于python，不需要安装。这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行。

港真的，今天是我第一次学习argsparse。因为用不到，自然也就没有学习的动力。但是现在电脑有点卡，每次打开pycharm太卡了，逼得我不得不开始使用命令行来测试代码。

## **传入一个参数**

我们先在桌面新建“arg学习”的文件夹，在该文件夹中新建demo.py文件，来看一个最简单的argsparse库的使用的例子。

```python
import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('integers', type=str, help='传入的数字')

args = parser.parse_args()

#获得传入的参数
print(args)
```

在这个代码中，我们在命令行传入一个数字。使用方法是打开命令行，先将工作目录cd到`arg学习`

```text
cd desktop/arg学习
```

然后再命令行中输入`python demo.py -h`或者`python demo.py --help`, 这里我输入的是

```text
python demo.py -h
```

在命令行中看到demo.py的运行结果如下

```text
usage: demo.py [-h] integers

命令行中传入数字

positional arguments:
  integers    传入的数字

optional arguments:
  -h, --help  show this help message and exit
```

现在我们在命令行中给demo.py 传入一个参数5，

```text
python demo.py 5
```

运行，得到的运行结果是

```text
Namespace(integers='5')
```

- description - 在参数帮助文档之前显示的文本（默认值：无）
- add_help - 为解析器添加一个 -h/–help 选项（默认值： True）

## **操作args字典**

其实得到的这个结果`Namespace(integers='5')`是一种类似于python字典的数据类型。

我们可以使用 `arg.参数名`来提取这个参数

```text
import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('integers', type=str, help='传入的数字')

args = parser.parse_args()

#获得integers参数
print(args.integers)
```

在命令行中运行 `python demo.py 5` , 运行结果为

```text
5
```

## **传入多个参数**

现在在命令行中给demo.py 传入多个参数，例如传入1，2，3，4四个数字

```text
python demo.py 1 2 3 4
```

运行报错

```text
usage: demo.py [-h] integers 
demo.py: error: unrecognized arguments: 2 3 4
```

不能识别2 3 4，看源代码我们知道integers这个参数是位置参数，说明第一个数`1`是能识别。这里我们需要重新更改demo.py代码

```text
import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
parser.add_argument('integers', type=str, nargs='+',help='传入的数字')
args = parser.parse_args()

print(args.integers)
```

**nargs是用来说明传入的参数个数，'+' 表示传入至少一个参数**。这时候再重新在命令行中运行`python demo.py 1 2 3 4`得到

```text
['1', '2', '3', '4']
```

## **改变数据类型**

add_argument中有type参数可以设置传入参数的数据类型。我们看到代码中有**type**这个关键词，该关键词可以传入list, str, tuple, set, dict等。例如我们把上面的type=str，改成type=int,这时候我们就可以进行四则运算。

```text
import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
parser.add_argument('integers', type=int, nargs='+',help='传入的数字')
args = parser.parse_args()

#对传入的数据进行加总
print(sum(args.integers)
```

在命令行中输入 `python demo.py 1 2 3 4`, 运行结果为

```text
10
```

## **位置参数**

在命令行中传入参数时候，传入的参数的先后顺序不同，运行结果往往会不同，这是因为采用了位置参数,例如

```text
import argparse

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('param1', type=str,help='姓')
parser.add_argument('param2', type=str,help='名')
args = parser.parse_args()

#打印姓名
print(args.param1+args.param2)
```

在命令行中分别输入`python demo.py 张 三`和`python demo.py 三 张`，得到的 运行结果分别为

```text
张三
```

和

```text
三张
```

如果我们将代码`parser.add_argument('param1', type=str,help='姓')`和

`parser.add_argument('param2', type=str,help='名')`互换位置，即第4行和第五行代码，再重新运行

`python demo.py 张 三` 和 `python demo.py 三 张`，得到的 运行结果分别为

```text
三张
```

和

```text
张三
```

## **可选参数**

为了在命令行中避免上述位置参数的bug（容易忘了顺序），可以使用可选参数，这个有点像关键词传参，但是需要在关键词前面加`--`，例如

```text
import argparse

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--family', type=str,help='姓')
parser.add_argument('--name', type=str,help='名')
args = parser.parse_args()

#打印姓名
print(args.family+args.name)
```

在命令行中输入

```text
python demo.py --family=张 --name=三
```

运行结果

```text
张三
```

可选参数虽然写法比较繁琐，但是增加了命令行中的可读性，不容易因为参数传入顺序导致数据错乱。

## **默认值**

add_argument中有一个default参数。有的时候需要对某个参数设置默认值，即如果命令行中没有传入该参数的值，程序使用默认值。如果命令行传入该参数，则程序使用传入的值。具体请看下面的例子

```text
import argparse

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--family', type=str, default='张',help='姓')
parser.add_argument('--name', type=str, default='三', help='名')
args = parser.parse_args()

#打印姓名
print(args.family+args.name)
```

在命令行中分别输入 `python demo.py` 、 `python demo.py --family=李`

运行结果分别为

```text
张三
```

和

```text
李三
```

## **必需参数**

add_argument有一个required参数可以设置该参数是否必需。

```python
import argparse

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--family', type=str, help='姓')
parser.add_argument('--name', type=str, required=True, default='', help='名')
args = parser.parse_args()

#打印姓名
print(args.family+args.name)
```

在命令行中输入 `python demo.py --family=张`，运行结果

```text
usage: demo.py [-h] [--family FAMILY] --name NAME
demo.py: error: the following arguments are required: --name
```

因为可选参数`name`的required=True，所以必须要传入。如果我们将其更改为False，程序运行结果

```text
张
```



# [argparse官方文档](https://docs.python.org/dev/howto/argparse.html#id1)

- -c：可选参数的简写 