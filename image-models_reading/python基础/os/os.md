# [import os总结](https://blog.csdn.net/Li_haiyu/article/details/80448028)

最近在看死磕yolo开源项目，之前没有做过相关的，所以，每一个句代码都要死磕，碰到import os 所以记录一下假装自己学过... .. .

其实我主要还是在造轮子咯

主要是参考下面这个博客：https://www.cnblogs.com/wuxie1989/p/5623435.html

还有官方文档：http://docs.python.org/library/os.path.html

首先我的实验目录在这里

![img](https://img-blog.csdn.net/20180525103850466?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xpX2hhaXl1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# 一、os.path.abspath(path)

返回path的绝对路径

```python
>>> os.path.abspath("train.py")



'E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py'
```

# 二、os.path.split(path)

将path分割成目录和文件名并以元组方式返回

```python
>>> os.path.split("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



('E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master', 'train.py')



>>> os.path.split("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master")



('E:\\my_project\\darknet\\darknet-master\\keras-yolo3', 'keras-yolo3-master')



>>> os.path.split("E:\\my_project\\darknet\\darknet-master\\keras-yolo3")



('E:\\my_project\\darknet\\darknet-master', 'keras-yolo3')



>>> os.path.split("E:\\my_project\\darknet\\darknet-master")



('E:\\my_project\\darknet', 'darknet-master')



>>> os.path.split("E:\\my_project\\darknet")



('E:\\my_project', 'darknet')



>>> os.path.split("E:\\my_project")



('E:\\', 'my_project')



>>> os.path.split("E:")



('E:', '')
```

# 三、os.path.dirname(path)

返回path的目录，其实就是返回os.path.split(path)元组的第一个元素

```python
>>> os.path.dirname("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



'E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master'



>>> os.path.dirname("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master")



'E:\\my_project\\darknet\\darknet-master\\keras-yolo3'



>>> os.path.dirname("E:\\my_project\\darknet\\darknet-master\\keras-yolo3")



'E:\\my_project\\darknet\\darknet-master'



>>> os.path.dirname("E:\\my_project\\darknet\\darknet-master")



'E:\\my_project\\darknet'



>>> os.path.dirname("E:\\my_project\\darknet")



'E:\\my_project'



>>> os.path.dirname("E:\\my_project")



'E:\\'
```

# 四、os.path.basename(path)

返回path的文件名，其实就是返回os.path.split(path)元组的第二个元素

```python
>>> os.path.basename("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



'train.py'



>>> os.path.basename("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master")



'keras-yolo3-master'



>>> os.path.basename("E:\\my_project\\darknet\\darknet-master\\keras-yolo3")



'keras-yolo3'



>>> os.path.basename("E:\\my_project\\darknet\\darknet-master")



'darknet-master'



>>> os.path.basename("E:\\my_project\\darknet")



'darknet'



>>> os.path.basename("E:\\my_project")



'my_project'
```

# 五、osa.path.commonprefix(list)

list里面每一个元素都是一个路径，然后这个函数返回路径中的公共路径

```python
>>> list = ["E:\\my_project","E:\\my_project\\darknet","E:\\my_project\\darknet\\darknet-master"]



>>> os.path.commonprefix(list)



'E:\\my_project'
```

# 六、os.path.exists(path)

如果path是一个存在的路径，返回True，否则（otherwise） 返回 False

```python
>>> os.path.exists("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



True



>>> os.path.exists("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master")



True



>>> os.path.exists("E:\\my_project\\darknet\\darknet-master\\keras-yolo3")



True



>>> os.path.exists("E:\\my_project\\darknet\\darknet-master")



True



>>> os.path.exists("E:\\my_project\\darknet")



True



>>> os.path.exists("E:\\my_project")



True



>>> os.path.exists("train.py")



True
```

下面并不存在这个文件路径，输出为False

```python
>>> os.path.exists("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.txt")



False



>>> os.path.exists("train.txt")



False
```

# 应用六：判断路径是否存在，不存在则创建：

```html
log_dir = "logs/"



if not os.path.exists(log_dir):



    os.makedirs(log_dir)
```

# 七、os.path.isabs(path)

如果路径path是绝对路径返回True，否则（otherwise）返回False

```python
>>> os.path.isabs("E:\\my_project\\darknet\\darknet-master")



True



>>> os.path.isabs("E:\\my_project\\darknet\\darknet-master\\keras-yolo3")



True



>>> os.path.isabs("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



True
```

下面不是绝对路径，输出为False

```python
>>> os.path.isabs("train.py")



False
```

# 八、os.path.isfile(path)

如果path是一个存在的文件，返回True，否者（otherwise）返回False

```python
>>> os.path.isfile("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



True



>>> os.path.isfile("train.py")



True
```

下面是一个不存在的文件

```python
>>> os.path.isfile("train.txt")



False
```

看下面没有文件的，输出全是False

```python
>>> os.path.isfile("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master")



False



>>> os.path.isfile("E:\\my_project\\darknet\\darknet-master\\keras-yolo3")



False



>>> os.path.isfile("E:\\my_project\\darknet\\darknet-master")



False



>>> os.path.isfile("E:\\my_project\\darknet")



False



>>> os.path.isfile("E:\\my_project")



False
```

# 九、os.path.isdir(path)

如果path里面存在目录，返回True，否则返回False

```python
>>> os.path.isdir("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



False
>>> os.path.isdir("E:\\my_project\\darknet")



True



>>> os.path.isdir("E:\\my_project\\darknet\\darknet-master")



True



>>> os.path.isdir("E:\\my_project\\darknet\\darknet-master\\keras-yolo3")



True



>>> os.path.isdir("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master")



True
```

# 十、os.path.join(path[,path2[,...])

组合多个路径并返回

```python
>>> os.path.join("E:\\my_project","darknet\\darknet-master","darknet-master\\keras-yolo3","keras-yolo3-master\\train.py")



'E:\\my_project\\darknet\\darknet-master\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py'
```

利用os.getcwd（）获取当前路径并组合返回

```python
>>> os.getcwd()



'E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master'



>>> os.path.join(os.getcwd(),'train.py')



'E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py'
```

# 十一、os.path.splitdrive(path)

返回（drivename，fpath）的元组，也就是将驱动磁盘和文件路径split一下返回元组

```python
>>> os.path.splitdrive("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



('E:', '\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py')
```

下面利用安装在D盘的anacoda路径进行测试一下

```python
>>> os.path.splitdrive("D:\\software1\\anacoda3")



('D:', '\\software1\\anacoda3')
```

 

# 十二、os.splitext（path）

 

分离扩展名然后按照元组返回

```python
>>> os.path.splitext("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



('E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train', '.py')
```

# 十三、os.path.getsize(path)

返回path文件的字节大小，可以传入绝对路径和相对路径

```python
>>> os.getcwd()



'E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master'



>>> os.path.getsize("train.py")



6478



>>> os.path.getsize("E:\\my_project\\darknet\\darknet-master\\keras-yolo3\\keras-yolo3-master\\train.py")



6478
```

![img](https://img-blog.csdn.net/20180525132052224?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xpX2hhaXl1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 十四、os.walk(path) 

os.walk返回的是一个生长器，需要使用next（）获取，最后next（）获取返回一个元组，第一个元素一般是当前相对路径，第二个元素一般是文件夹的名字，第三个元素一般是文件的名字，在一些文件夹和文件共存的文件夹内操作比较方便。和os.listdir相比就是后者将文件夹内所有子文件夹和文件以列表返回，看例子

![img](https://img-blog.csdn.net/20180726102547511?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xpX2hhaXl1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 samples文件夹和文件如下

```python
li_haiyu@come-on:/media/li_haiyu/E/Mask_RCNN-master/samples$ python



Python 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 18:10:19) 



[GCC 7.2.0] on linux



Type "help", "copyright", "credits" or "license" for more information.



>>> import os



>>> os.getcwd()



'/media/li_haiyu/E/Mask_RCNN-master/samples'



>>> next(os.walk("./"))



('./', ['shapes', 'balloon', '.ipynb_checkpoints', 'coco', 'nucleus'], ['2.cpp', 'test', 'demo.ipynb', '1.py'])



>>> next(os.walk("./"))[2]



['2.cpp', 'test', 'demo.ipynb', '1.py']



>>> os.listdir("./")



['shapes', '2.cpp', 'test', 'balloon', '.ipynb_checkpoints', 'demo.ipynb', 'coco', '1.py', 'nucleus']



>>> 
```

# ***\*提醒：\****

我觉的应该注意一下的就是os.path.exists(path)和os.path.isfile(path),前者是判断路径是否存在，后者是判断该文件是否存在。

The end.

更多请参考：https://www.cnblogs.com/wuxie1989/p/5623435.html & http://docs.python.org/library/os.path.html

彩蛋：[import Image 总结](https://www.cnblogs.com/kongzhagen/p/6295925.html)

2019.4.26 add.

> 判断文件夹心是否存在，存在递归删除之

```python
import shutil



import os



path = os.path.join(os.getcwd(),'test')



if os.path.isdir(path):



    shutil.rmtree(path)



    print("test dir is removed!!!")
```

 

 

 