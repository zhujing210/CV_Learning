**题目：**

给定一个 N 叉树，返回其节点值的 后序遍历 。

N 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 null 分隔（请参见示例）。

 

进阶：

递归法很简单，你可以使用迭代法完成此题吗?

示例 1：

 ![img](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
输入：root = [1,null,3,2,4,null,5,6]
输出：[5,6,3,2,4,1]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

```
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：[2,6,14,11,7,3,12,8,4,13,9,10,5,1]
```

链接：https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。





**自己解：**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        def dfs(node):
            if not node:return
            if node.children:
                for children in node.children:
                    dfs(children)
            self.out.append(node.val)
        self.out = []
        dfs(root)
        return self.out
```

执行用时：76 ms, 在所有 Python3 提交中击败了8.08%的用户

内存消耗：16.5 MB, 在所有 Python3 提交中击败了78.82%的用户



迭代法没有写出来



**递归法**

```python
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if not root: return []
        res = []
        for child in root.children:
            res.extend(postorder(child))
        res.append(root.val)
        return res
```

**迭代法：**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if not root : return []
        stack , out = [root], []
        while stack:
            node = stack.pop()
            out.append(node.val)
            stack.extend(node.children)
        return out[::-1]
```

