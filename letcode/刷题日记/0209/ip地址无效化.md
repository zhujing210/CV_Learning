**题目：**

给你一个有效的 [IPv4](https://baike.baidu.com/item/IPv4) 地址 `address`，返回这个 IP 地址的无效化版本。

所谓无效化 IP 地址，其实就是用 `"[.]"` 代替了每个 `"."`。



**自己解：**

```python
class Solution:
    def defangIPaddr(self, address: str) -> str:
        add_list = list(address)
        for i in range(len(add_list)):
            if add_list[i] == '.':
                add_list[i] = '[.]'
        return "".join(add_list)
```

执行用时：44 ms

内存消耗：14.8 MB

**更优解：**

```python
class Solution:
    def defangIPaddr(self, address: str) -> str:
        return address.replace(".", "[.]")
```

- str.replace('a', 'b')