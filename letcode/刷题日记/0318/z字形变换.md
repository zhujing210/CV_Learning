**题目：**

将一个给定字符串 `s` 根据给定的行数 `numRows` ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 `"PAYPALISHIRING"` 行数为 `3` 时，排列如下：

```
P   A   H   N
A P L S I I G
Y   I   R
```

之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如：`"PAHNAPLSIIGYIR"`。

请你实现这个将字符串进行指定行数变换的函数：

```
string convert(string s, int numRows);
```

**示例 1：**

```
输入：s = "PAYPALISHIRING", numRows = 3
输出："PAHNAPLSIIGYIR"
```

**示例 2：**

```
输入：s = "PAYPALISHIRING", numRows = 4
输出："PINALSIGYAHRPI"
解释：
P     I    N
A   L S  I G
Y A   H R
P     I
```

**示例 3：**

```
输入：s = "A", numRows = 1
输出："A"
```

**提示：**

- `1 <= s.length <= 1000`
- `s` 由英文字母（小写和大写）、`','` 和 `'.'` 组成
- `1 <= numRows <= 1000`



**自己解：**

。。。



**其他解：**

- **题目理解：**
  - 字符串 `s` 是以 Z字形为顺序存储的字符串，目标是按行打印。
  - 设 `numRows` 行字符串分别为 $s_1 , s_2，s_2 ,..., s_n$，则容易发现：按顺序遍历字符串 `s` 时，每个字符 `c` 在 Z 字形中对应的 **行索引** 先从 $ s_1$增大至 $s_n$，，再从 $s_n$ 减小至 $s_1$…… 如此反复。
  - 因此，解决方案为：模拟这个行索引的变化，在遍历 `s` 中把每个字符填到正确的行 `res[i]` 。

- 算法流程：按顺序遍历字符串 `s`；

1. `res[i]+=c`：把每个字符`c`填入对应行$s_i$
2. `i += flag`：更新当前字符`c`对应的行索引
3. flag =- flag：在到达Z字形转折时，执行反向

- 复杂度分析：
  - 时间复杂度分析$O(N)$:遍历一遍字符串`s`;
  - 空间复杂度$O(N)$：各行字符串共占用$O(N)$额外空间

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows < 2: return s
        res = ["" for _ in range(numRows)]
        i, flag = 0, -1
        for c in s:
            res[i] += c
            if i == 0 or i == numRows - 1: flag = -flag
            i += flag
        return "".join(res)
```

