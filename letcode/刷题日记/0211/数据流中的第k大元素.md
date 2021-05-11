设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，不是第 k 个不同的元素。

请实现 KthLargest 类：

KthLargest(int k, int[] nums) 使用整数 k 和整数流 nums 初始化对象。
int add(int val) 将 val 插入数据流 nums 后，返回当前数据流中第 k 大的元素。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/kth-largest-element-in-a-stream
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。



**自己解：**

```python
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = nums


    def add(self, val: int) -> int:
        self.nums.append(val)
        self.nums.sort(reverse=True)
        return self.nums[self.k-1]



# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```

执行用时：1056 ms

内存消耗：18.8 MB



**更优解：**

思路：

- python的`list.sort()`函数内部实现机制为：Timesort

  最坏时间复杂度为：O（n log n）

  空间复杂度为：O（n）

  使用**数组**的核心问题是：数组自身不带排序功能，只能用 `sort()` 函数，导致时间复杂度过高。

- 考虑使用自带排序功能的数据结构——**堆**。

本题的操作步骤如下：

1. 使用大小为 K 的**小根堆**，在初始化的时候，保证堆中的元素个数不超过 K 。
2. 在每次 `add()` 的时候，将新元素 `push()` 到堆中，如果此时堆中的元素超过了 K，那么需要把堆中的最小元素（堆顶）`pop()` 出来。
3. 此时堆中的最小元素（堆顶）就是整个数据流中的第 K 大元素。

问答：

1.为什么使用小根堆？
因为我们需要在堆中保留数据流中的前 K 大元素，使用小根堆能保证每次调用堆的 `pop()` 函数时，从堆中删除的是堆中的最小的元素（堆顶）。
2.为什么能保证堆顶元素是第 K 大元素？
因为小根堆中保留的一直是堆中的前 K 大的元素，堆的大小是 K，所以堆顶元素是第 K 大元素。
3.每次 add() 的时间复杂度是多少？
每次 add() 时，调用了堆的 push() 和 pop() 方法，两个操作的时间复杂度都是 log(K).

```python
class KthLargest(object):

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.k = k
        self.que = nums
        heapq.heapify(self.que)

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        heapq.heappush(self.que, val)
        while len(self.que) > self.k:
            heapq.heappop(self.que)
        return self.que[0]

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```

执行用时：88 ms

内存消耗：18.5 MB

- 因为我没使用过heapq, 所以看到相关内容时有些懵. 赶紧到Python doc阅读了一下, 传送门在这里https://docs.python.org/3/library/heapq.html. 大家加油~ 新年快乐

