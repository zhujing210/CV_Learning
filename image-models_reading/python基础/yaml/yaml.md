# [python 第三方模块 yaml - 处理 YAML （专门用来写配置文件的语言）](https://blog.csdn.net/fenglepeng/article/details/112331659)

## 一、yaml文件介绍

yaml是一个专门用来写配置文件的语言。

###### 1. yaml文件规则

- 区分大小写；
- 使用缩进表示层级关系；
- 使用空格键缩进，而非Tab键缩进
- 缩进的空格数目不固定，只需要相同层级的元素左侧对齐；
- 文件中的字符串不需要使用引号标注，但若字符串包含有特殊字符则需用引号标注；
- 注释标识为#

###### 2. yaml文件数据结构

- 对象：键值对的集合（简称 "映射或字典"）
   键值对用冒号 “:” 结构表示，冒号与值之间需用空格分隔
- 数组：一组按序排列的值（简称 "序列或列表"）
   数组前加有 “-” 符号，符号与值之间需用空格分隔
- 纯量(scalars)：单个的、不可再分的值（如：字符串、bool值、整数、浮点数、时间、日期、null等）
   None值可用null可 ~ 表示

## 二、python中读取yaml配置文件

###### 1. 前提条件

python中读取yaml文件前需要安装pyyaml和导入yaml模块：

- 使用yaml需要安装的模块为pyyaml（pip3 install pyyaml）;
- 导入的模块为yaml（import yaml）

###### 2. 读取yaml文件数据

python通过open方式读取文件数据，再通过load函数将数据转化为列表或字典；



```python
import yaml
import os

def get_yaml_data(yaml_file):
    # 打开yaml文件
    print("***获取yaml文件数据***")
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    print(file_data)
    print("类型：", type(file_data))

    # 将字符串转化为字典或列表
    print("***转化yaml数据为字典或列表***")
    data = yaml.load(file_data)
    print(data)
    print("类型：", type(data))
    return data
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "config.yaml")
get_yaml_data(yaml_path)

"""
***获取yaml文件数据***
# yaml键值对：即python中字典
usr: my
psw: 123455
类型：<class 'str'>
***转化yaml数据为字典或列表***
{'usr': 'my', 'psw': 123455}
类型：<class 'dict'>
"""
```

###### 3. yaml文件数据为键值对

（1）yaml文件中内容为键值对：



```bash
# yaml键值对：即python中字典
usr: my
psw: 123455
s: " abc\n"
```

python解析yaml文件后获取的数据：



```bash
{'usr': 'my', 'psw': 123455, 's': ' abc\n'}
```

（2）yaml文件中内容为“键值对'嵌套"键值对"



```bash
# yaml键值对嵌套：即python中字典嵌套字典
usr1:
  name: a
  psw: 123
usr2:
  name: b
  psw: 456
```

python解析yaml文件后获取的数据：



```bash
{'usr1': {'name': 'a', 'psw': 123}, 'usr2': {'name': 'b', 'psw': 456}}
```

（3）yaml文件中“键值对”中嵌套“数组”



```bash
# yaml键值对中嵌套数组
usr3:
  - a
  - b
  - c
usr4:
  - b
```

python解析yaml文件后获取的数据：



```bash
{'usr3': ['a', 'b', 'c'], 'usr4': ['b']}
```

###### 4. yaml文件数据为数组

（1）yaml文件中内容为数组



```bash
# yaml数组
- a
- b
- 5
```

python解析yaml文件后获取的数据：



```json
['a', 'b', 5]
```

（2）yaml文件“数组”中嵌套“键值对”



```cpp
# yaml"数组"中嵌套"键值对"
- usr1: aaa
- psw1: 111
  usr2: bbb
  psw2: 222
```

python解析yaml文件后获取的数据：



```bash
[{'usr1': 'aaa'}, {'psw1': 111, 'usr2': 'bbb', 'psw2': 222}]
```

###### 5. yaml文件中基本数据类型：



```dart
# 纯量
s_val: name              # 字符串：{'s_val': 'name'}
spec_s_val: "name\n"    # 特殊字符串：{'spec_s_val': 'name\n'
num_val: 31.14          # 数字：{'num_val': 31.14}
bol_val: true           # 布尔值：{'bol_val': True}
nul_val: null           # null值：{'nul_val': None}
nul_val1: ~             # null值：{'nul_val1': None}
time_val: 2018-03-01t11:33:22.55-06:00     # 时间值：{'time_val': datetime.datetime(2018, 3, 1, 17, 33, 22, 550000)}
date_val: 2019-01-10    # 日期值：{'date_val': datetime.date(2019, 1, 10)}
```

###### 6. yaml文件中引用

yaml文件中内容



```bash
animal3: &animal3 fish
test: *animal3
```

python读取的数据



```bash
{'animal3': 'fish', 'test': 'fish'}
```

## 三、python中读取多个yaml文档

###### 1. 多个文档在一个yaml文件，使用 --- 分隔方式来分段

如：yaml文件中数据



```bash
# 分段yaml文件中多个文档
---
animal1: dog
age: 2
---
animal2: cat
age: 3
```

###### 2. python脚本读取一个yaml文件中多个文档方法

python获取yaml数据时需使用load_all函数来解析全部的文档，再从中读取对象中的数据



```python
# yaml文件中含有多个文档时，分别获取文档中数据
def get_yaml_load_all(yaml_file):
    # 打开yaml文件
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    all_data = yaml.load_all(file_data)
    for data in all_data:
        print(data)
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "config.yaml")
get_yaml_load_all(yaml_path)
"""结果
{'animal1': 'dog', 'age': 2}
{'animal2': 'cat', 'age': 3}
"""
```

## 四、python对象生成yaml文档

###### 1. 直接导入yaml（即import yaml）生成的yaml文档

通过yaml.dump()方法不会将列表或字典数据进行转化yaml标准模式，只会将数据生成到yaml文档中



```python
# 将python对象生成yaml文档
import yaml
def generate_yaml_doc(yaml_file):
    py_object = {'school': 'zhang',
                 'students': ['a', 'b']}
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(py_object, file)
    file.close()
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "generate.yaml")
generate_yaml_doc(yaml_path)
"""结果
school: zhang
students: [a, b]
"""
```

###### 2. 使用ruamel模块中的yaml方法生成标准的yaml文档

（1）使用ruamel模块中yaml前提条件

- 使用yaml需要安装的模块：ruamel.yaml（pip3 install ruamel.yaml）;
- 导入的模块：from ruamel import yaml

（2）ruamel模块生成yaml文档



```python
def generate_yaml_doc_ruamel(yaml_file):
    from ruamel import yaml
    py_object = {'school': 'zhang',
                 'students': ['a', 'b']}
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(py_object, file, Dumper=yaml.RoundTripDumper)
    file.close()
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "generate.yaml")
generate_yaml_doc_ruamel(yaml_path)
"""结果
school: zhang
students:
- a
- b
"""
```

（3）ruamel模块读取yaml文档



```python
# 通过from ruamel import yaml读取yaml文件
def get_yaml_data_ruamel(yaml_file):
    from ruamel import yaml
    file = open(yaml_file, 'r', encoding='utf-8')
    data = yaml.load(file.read(), Loader=yaml.Loader)
    file.close()
    print(data)
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "dict_config.yaml")
get_yaml_data_ruamel(yaml_path)
```



作者：rr1990
链接：https://www.jianshu.com/p/eaa1bf01b3a6
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。