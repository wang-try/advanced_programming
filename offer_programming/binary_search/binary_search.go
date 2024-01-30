package binary_search

import (
	"math/rand"
	"time"
)

//查找插入位置
/*
输入一个排序的整数数组nums和一个目标值t，如果数组nums中包含t，则返回t在数组中的下标；
如果数组nums中不包含t，则返回将t按顺序插入数组nums中的下标。假设数组中没有相同的数字。例如，输入数组nums为[1，3，6，8]，如果目标值t为3，则输出1；如果t为5，则返回2
*/
func SearchInsert(nums []int, target int) int {
	lhs := 0
	rhs := len(nums) - 1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] >= target {
			if mid == 0 || nums[mid-1] < target {
				return mid
			}
			rhs = mid - 1
		} else {
			lhs = mid + 1
		}
	}
	return len(nums)

}

//山峰数组的顶部
/*
在一个长度大于或等于3的数组中，任意相邻的两个数字都不相等。该数组的前若干数字是递增的，之后的数字是递减的，因此它的值看起来像一座山峰。
请找出山峰顶部，即数组中最大值的位置。例如，在数组[1，3，5，4，2]中，最大值是5，输出它在数组中的下标2
*/
func PeakIndexInMountainArray(arr []int) int {
	lhs := 1
	rhs := len(arr) - 2
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if arr[mid] > arr[mid-1] && arr[mid] > arr[mid+1] {
			return mid
		} else if arr[mid] < arr[mid+1] {
			lhs = mid + 1
		} else {
			rhs = mid - 1
		}
	}
	return -1
}

//排序数组中只出现一次的数字
/*
在一个排序的数组中，除一个数字只出现一次之外，其他数字都出现了两次，请找出这个唯一只出现一次的数字。例如，在数组[1，1，2，2，3，4，4，5，5]中，数字3只出现了一次。
*/

func SingleNonDuplicate(nums []int) int {
	ret := nums[0]
	for i := 1; i < len(nums); i++ {
		ret ^= nums[i]
	}
	return ret
}

func SingleNonDuplicateV2(nums []int) int {
	lhs := 0
	rhs := len(nums) / 2
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		i := mid * 2
		if i < len(nums)-1 && nums[i] != nums[i+1] {
			if mid == 0 || nums[i-2] == nums[i-1] {
				return nums[i]
			}
			rhs = mid - 1
		} else {
			lhs = mid + 1
		}
	}
	return nums[len(nums)-1]
}

//按权重生成随机数
/*
输入一个正整数数组w，数组中的每个数字w[i]表示下标i的权重，请实现一个函数pickIndex根据权重比例随机选择一个下标。
例如，如果权重数组w为[1，2，3，4]，那么函数pickIndex将有10%的概率选择0、20%的概率选择1、30%的概率选择2、40%的概率选择3。
*/

// 超时
type SolutionV2 struct {
	Index2rate map[int]Interval
	Sum        int
}

type Interval struct {
	Start int
	End   int
}

func ConstructorV2(w []int) SolutionV2 {
	s := SolutionV2{
		Index2rate: make(map[int]Interval),
		Sum:        0,
	}
	for _, weight := range w {
		s.Sum += weight
	}

	start := 0
	for i := 0; i < len(w); i++ {
		end := w[i] + start
		s.Index2rate[i] = Interval{
			Start: start,
			End:   end,
		}
		start = end
	}
	return s
}

// 1
func (this *SolutionV2) PickIndex() int {
	rand.Seed(time.Now().UnixNano())
	r := rand.Intn(this.Sum)
	for k, v := range this.Index2rate {
		if r >= v.Start && r < v.End {
			return k
		}
	}
	return -1
}

type Solution struct {
	Sum    int
	Weight []int
}

func Constructor(w []int) Solution {
	s := Solution{
		Sum:    0,
		Weight: make([]int, 0, len(w)),
	}
	for _, weight := range w {
		s.Sum += weight
		s.Weight = append(s.Weight, s.Sum)
	}
	return s
}

func (this *Solution) PickIndex() int {
	rand.Seed(time.Now().UnixNano())
	r := rand.Intn(this.Sum) + 1
	lhs := 0
	rhs := len(this.Weight) - 1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if this.Weight[mid] >= r {
			if mid == 0 || this.Weight[mid-1] < r {
				return mid
			}
			rhs = mid - 1
		} else {
			lhs = mid + 1
		}
	}
	return -1
}

//在数值范围内二分查找
/*
如果一开始不知道问题的解是什么，但是知道解的范围是多少，则可以尝试在这个范围内应用二分查找。假设解的范围的最小值是min，最大值是max，先尝试范围内的中间值mid。如果mid正好是问题的解，那么固然好。
当mid不是问题的解时，如果能够判断接下来应该在从min到mid-1或从mid+1到max的范围内查找，那么就可以继续重复二分查找的过程，直到找到解为止。应用这种思路的关键在于两点：一是确定解的范围，即解的可能的最小值和最大值。
二是在发现中间值不是解之后如何判断接下来应该在解的范围的前半部分还是后半部分查找。只有每次将查找范围减少一半时才能应用二分查找算法
*/

//求平方根

/*
输入一个非负整数，请计算它的平方根。正数的平方根有两个，只输出其中的正数平方根。如果平方根不是整数，那么只需要输出它的整数部分。例如，如果输入4则输出2；如果输入18则输出4。
*/

func MySqrt(x int) int {
	lhs := 0
	rhs := x
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if mid*mid <= x {
			if (mid+1)*(mid+1) > x {
				return mid
			}
			lhs = mid + 1
		} else {
			rhs = mid - 1
		}
	}
	return -1
}

// mid*mid 可能溢出
func MySqrtV2(x int) int {
	lhs := 0
	rhs := x
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if mid <= x/mid {
			if (mid + 1) > x/(mid+1) {
				return mid
			}
			lhs = mid + 1
		} else {
			rhs = mid - 1
		}
	}
	return -1
}

//狒狒吃香蕉
/*
狒狒很喜欢吃香蕉。一天它发现了n堆香蕉，第i堆有piles[i]根香蕉。门卫刚好走开，H小时后才会回来。狒狒吃香蕉喜欢细嚼慢咽，但又想在门卫回来之前吃完所有的香蕉。
请问狒狒每小时至少吃多少根香蕉？如果狒狒决定每小时吃k根香蕉，而它在吃的某一堆剩余的香蕉的数目少于k，那么它只会将这一堆的香蕉吃完，下一个小时才会开始吃另一堆的香蕉。
*/

func MinEatingSpeed(piles []int, h int) int {
	max := 0
	for _, pile := range piles {
		if pile > max {
			max = pile
		}
	}
	lhs := 1
	rhs := max
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		uh := getHours(piles, mid)
		if uh <= h {
			if mid == 1 || getHours(piles, mid-1) > h {
				return mid
			}
			rhs = mid - 1
		} else {
			lhs = mid + 1
		}
	}
	return -1
}

func getHours(piles []int, speed int) int {
	hours := 0
	for _, pile := range piles {
		hours += (pile + speed - 1) / speed
	}
	return hours
}
