#### [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)



# 自己

尝试用哈希表的方式实现min()函数，不过哈希表还是无法找到有定义的正确最小索引

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        #self.hashtable = [float('inf')]*10000


    def push(self, x:  int) -> None:
        self.stack.append(x)
        #self.hashtable[x] = x
  

    def pop(self) -> None:
        self.stack.pop()
        #self.hashtable[self.stack[-1]] = float('inf')


    def top(self) -> int:
        return self.stack[-1]


    def min(self) -> int:
        return min(self.stack)
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```



# 辅助栈

#### 解题思路：

普通栈的 `push()` 和 `pop()` 函数的复杂度为 O(1)；而获取栈最小值 `min()` 函数需要遍历整个栈，复杂度为 O(N) 。

- **本题难点：**将 `min()` 函数复杂度降为 O(1)，可通过建立辅助栈实现；
  - **数据栈 A：** 栈 A 用于存储所有元素，保证入栈 `push()` 函数、出栈 `pop()` 函数、获取栈顶 `top()` 函数的正常逻辑。
  - **辅助栈 B：**栈 B 中存储栈 A 中所有 **非严格降序** 的元素，则栈 A 中的最小元素始终对应栈 B 的栈顶元素，即 min() 函数只需返回栈 B 的栈顶元素即可。

- 因此，只需设法维护好 栈 B 的元素，使其保持非严格降序，即可实现 `min()`函数的 O(1) 复杂度。

<img src="https://pic.leetcode-cn.com/f31f4b7f5e91d46ea610b6685c593e12bf798a9b8336b0560b6b520956dd5272-Picture1.png" alt="Picture1.png" style="zoom:50%;" />

##### 函数设计：

- **`push(x)` 函数：** 重点为保持栈 B 的元素是 **非严格降序** 的。

  1.将 x 压入栈 A（即 `A.add(x)` ）；

  2.若 ① 栈 B 为空**或** ②  x**小于等于** 栈 B 的栈顶元素，则将 *x* 压入栈 B （即 `B.add(x)`  )。

- **`pop()` 函数：** 重点为保持栈 A, B 的 **元素一致性** 。

  1.执行栈 A 出栈（即 `A.pop()`），将出栈元素记为 y；

  2.若 y 等于栈 B 的栈顶元素，则执行栈 `B` 出栈（即 `B.pop()` ）。

- **`top()`函数：** 直接返回栈 A 的栈顶元素即可，即返回 `A.peek()` 。
- **`min()`** **函数：**直接返回栈 B 的栈顶元素即可，即返回 `B.peek()` 。

##### 复杂度分析：

- **时间复杂度 O(1) ：** `push()`，`pop()`， `top()`，`min()`四个函数的时间复杂度均为常数级别。
- **空间复杂度 O(N)：**当共有 N 个待入栈元素时，辅助栈 B 最差情况下存储 N 个元素，使用 O(N) 额外空间。

#### 代码：

python:

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = []


    def push(self, x:  int) -> None:
        self.stack.append(x)
        if (not self.min_stack) or x <= self.min_stack[-1]:
            self.min_stack.append(x)
  

    def pop(self) -> None:
        y = self.stack[-1]
        self.stack.pop()
        if y==self.min_stack[-1]:
            self.min_stack.pop()


    def top(self) -> int:
        return self.stack[-1]


    def min(self) -> int:
        return self.min_stack[-1]
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```



C++：

```C++
class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {
        ;
    }
    
    void push(int x) {
        s.push(x);
        if (f.empty() || x<=f.top()){
            f.push(x);
        } 
    }
    
    void pop() {
        int y = s.top();
        s.pop();
        if (y==f.top()){
            f.pop();
        }
    }
    
    int top() {
        return s.top();
    }
    
    int min() {
        return f.top();
    }
private:
    stack<int> s, f;
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```





- 辅助栈 可用于记录 有序序列