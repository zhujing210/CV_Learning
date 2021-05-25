#### [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)



# 自己解

左右指针做法

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        l, r = 0, len(nums)-1
        while l<r:
            if nums[l]+nums[r]==target: return [nums[l], nums[r]]
            elif nums[l] + nums[r] < target:
                l+=1
            else: r-=1
        return []
```

