

[suppress](https://blog.csdn.net/zhou_438/article/details/109293498)

这不，官方的文档实现了一个方法[suppress](https://docs.python.org/3/library/contextlib.html#contextlib.closing)，用于处理异常

```python
from contextlib import suppress
 
with suppress(FileNotFoundError):
    os.remove('somefile.tmp')
 
with suppress(FileNotFoundError):
    os.remove('someotherfile.tmp')
```

接着文档指出上面的代码**等同于**：

```python
try:
    os.remove('somefile.tmp')
except FileNotFoundError:
    pass
 
try:
    os.remove('someotherfile.tmp')
except FileNotFoundError:
    pass
```

我查了suppress的源码，实现其实很简单：

```python
class suppress(ContextDecorator):
    def __init__(self, *exceptions):
        self._exceptions = exceptions
    def __enter__(self):
        pass
    def __exit__(self, exctype, excinst, exctb):
        return exctype is not None and issubclass(exctype, self._exceptions)
```

就是利用上下文管理器在实现的，不过我不满意的是没有使用装饰器，每次要使用suppress必须要加with 

于是实现了一版带带装饰器的：

```python
import os
from contextlib import ContextDecorator
 
class suppress(ContextDecorator):
    def __init__(self, *exceptions):
        self._exceptions = exceptions
    def __enter__(self):
        print(1)
    def __exit__(self, exctype, excinst, exctb):
        print(3)
        return exctype is not None and issubclass(exctype, self._exceptions)
    
@suppress(FileNotFoundError)
def test_exception():
    print(2)
    os.remove('data.py')
test_exception()
```

运行结果:

```python
1
2
3
```

没有弹出异常，并且打印出了数字，说明是效果是成功的 

如果能直接将源码的这部分替换，那么下次使用直接在函数上面加个装饰器不就可以处理函数的异常了