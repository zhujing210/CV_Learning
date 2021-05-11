## [Python中OrderedDict的使用](https://www.cnblogs.com/notzy/p/9312049.html)



很多人认为python中的字典是无序的，因为它是按照hash来存储的，但是python中有个模块collections(英文，收集、集合)，里面自带了一个子类OrderedDict，实现了对字典对象中元素的排序。请看下面的实例：

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```python
 1 import collections
 2 print "Regular dictionary"
 3 d={}
 4 d['a']='A'
 5 d['b']='B'
 6 d['c']='C'
 7 for k,v in d.items():
 8     print k,v 
 9 print "\nOrder dictionary"
10 d1 = collections.OrderedDict()
11 d1['a'] = 'A'
12 d1['b'] = 'B'
13 d1['c'] = 'C'
14 d1['1'] = '1'
15 d1['2'] = '2'
16 for k,v in d1.items():
17     print k,v
18 
19 输出：
20 Regular dictionary
21 a A
22 c C
23 b B
24 
25 Order dictionary
26 a A
27 b B
28 c C
29 1 1
30 2 2
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

可以看到，同样是保存了ABC等几个元素，但是使用OrderedDict会根据放入元素的先后顺序进行排序。所以输出的值是排好序的。

OrderedDict对象的字典对象，如果其顺序不同那么Python也会把他们当做是两个不同的对象，请看事例：

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```python
 1 print 'Regular dictionary:'
 2 d2={}
 3 d2['a']='A'
 4 d2['b']='B'
 5 d2['c']='C'
 6 
 7 d3={}
 8 d3['c']='C'
 9 d3['a']='A'
10 d3['b']='B' 
11 
12 print 'OrderedDict:'
13 d4=collections.OrderedDict()
14 d4['a']='A'
15 d4['b']='B'
16 d4['c']='C'
17  
18 d5=collections.OrderedDict()
19 d5['c']='C'
20 d5['a']='A'
21 d5['b']='B'
22 
23 print  d1==d2
24 
25 输出：
26 Regular dictionary:
27 True
28  
29 OrderedDict:
30 False
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

再看几个例子：

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```python
 1 dd = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}
 2 #按key排序
 3 kd = collections.OrderedDict(sorted(dd.items(), key=lambda t: t[0]))
 4 print kd
 5 #按照value排序
 6 vd = collections.OrderedDict(sorted(dd.items(),key=lambda t:t[1]))
 7 print vd
 8 
 9 #输出
10 OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
11 OrderedDict([('pear', 1), ('orange', 2), ('banana', 3), ('apple', 4)])
```