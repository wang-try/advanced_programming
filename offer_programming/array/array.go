package array

import (
	"math"
	"sort"
)

// 排序数组中的两个数字之和
func TwoSum(numbers []int, target int) []int {
	lhs := 0
	rhs := len(numbers) - 1

	for lhs < rhs {
		sum := numbers[lhs] + numbers[rhs]
		if target == sum {
			return []int{lhs, rhs}
		} else if target > sum {
			rhs--
		} else {
			lhs++
		}
	}
	return nil
}

// 数组中和为0的3个数字
func ThreeSum(nums []int) [][]int {
	sort.Slice(nums, func(i, j int) bool { return nums[i] < nums[j] })
	var ret [][]int
	for i, num := range nums {
		if i == 0 || num > nums[i-1] {
			target := -num
			lhs := i + 1
			rhs := len(nums) - 1
			for lhs < rhs {
				sum := nums[lhs] + nums[rhs]
				if sum == target {
					ret = append(ret, []int{num, nums[lhs], nums[rhs]})
					lhs++
					for lhs < rhs && nums[lhs] == nums[lhs-1] {
						lhs++
					}
					rhs--
					for lhs < rhs && nums[rhs] == nums[rhs+1] {
						rhs--
					}
				} else if sum > target {
					rhs--
				} else {
					lhs++
				}
			}
		}
	}
	return ret
}

func ThreeSumV2(nums []int) [][]int {
	sort.Slice(nums, func(i, j int) bool { return nums[i] < nums[j] })
	lth := len(nums)
	var res [][]int
	for i := 0; i <= lth-3 && nums[i] <= 0; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		lhs := i + 1
		rhs := lth - 1
		target := 0 - nums[i]
		for lhs < rhs {
			if nums[rhs]+nums[lhs] == target {
				res = append(res, []int{nums[i], nums[lhs], nums[rhs]})
				lhs++
				rhs--
				for lhs < rhs && nums[lhs] == nums[lhs-1] {
					lhs++
				}
				for lhs < rhs && nums[rhs] == nums[rhs+1] {
					rhs--
				}
			} else if nums[rhs]+nums[lhs] > target {
				rhs--
			} else {
				lhs++
			}
		}
	}
	return res
}

// 和大于或等于k的最短子数组
func MinSubArrayLen(k int, nums []int) int {
	left := 0
	sum := 0
	minLth := math.MaxInt32
	for right := 0; right < len(nums); right++ {
		sum += nums[right]
		for left <= right && sum >= k {
			if (right - left + 1) < minLth {
				minLth = right - left + 1
			}
			sum -= nums[left]
			left++
		}
	}
	if minLth == math.MaxInt32 {
		minLth = 0
	}
	return minLth

}

// 乘积小于k的子数组
func NumSubarrayProductLessThanK(nums []int, k int) int {
	left := 0
	product := 1
	cnt := 0
	for right := 0; right < len(nums); right++ {
		product *= nums[right]
		for left <= right && product >= k {
			product /= nums[left]
			left++
		}
		if right >= left {
			cnt += right - left + 1
		}
	}
	return cnt
}

// 和为k的子数组
/*输入一个正整数组成的数组和一个正整数k，请问数组中和大于或等于k的连续子数组的最短长度是多少？
如果不存在所有数字之和大于或等于k的子数组，则返回0。例如，输入数组[5，1，4，3]，k的值为7，和大于或等于7的最短连续子数组是[4，3]，因此输出它的长度2。*/
func SubArraySum(nums []int, k int) int {
	sum2cnt := make(map[int]int)
	sum2cnt[0] = 1
	sum := 0
	cnt := 0
	for _, num := range nums {
		sum += num
		cnt += sum2cnt[sum-k]
		sum2cnt[sum]++
	}
	return cnt
}

//0和1个数相同的子数组
/*
输入一个只包含0和1的数组，请问如何求0和1的个数相同的最长连续子数组的长度？例如，在数组[0，1，0]中有两个子数组包含相同个数的0和1，分别是[0，1]和[1，0]，它们的长度都是2，因此输出2
*/
func FindMaxLength01Same(nums []int) int {
	sum := 0
	maxLth := 0
	sum2index := make(map[int]int)
	//虚拟位置，数组-1位置的值为0，index为-1，这里为了兼容边界问题
	sum2index[0] = -1
	for i, num := range nums {
		tmp := num
		if num == 0 {
			tmp = -1
		}
		sum += tmp
		if index, ok := sum2index[sum]; ok {
			if (i - index) > maxLth {
				maxLth = i - index
			}
		} else {
			sum2index[sum] = i
		}
	}
	return maxLth
}

//左右两边子数组的和相等
/*
输入一个整数数组，如果一个数字左边的子数组的数字之和等于右边的子数组的数字之和，那么返回该数字的下标。如果存在多个这样的数字，则返回最左边一个数字的下标。如果不存在这样的数字，则返回-1。
例如，在数组[1，7，3，6，2，9]中，下标为3的数字（值为6）的左边3个数字1、7、3的和与右边两个数字2和9的和相等，都是11，因此正确的输出值是3
*/

func PivotIndex(nums []int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	subSum := 0
	for i, num := range nums {
		subSum += num
		if subSum-num == sum-subSum {
			return i
		}
	}
	return -1
}

//二维子矩阵的数字之和
/*
输入一个二维矩阵，如何计算给定左上角坐标和右下角坐标的子矩阵的数字之和？对于同一个二维矩阵，计算子矩阵的数字之和的函数可能由于输入不同的坐标而被反复调用多次。
例如，输入图2.1中的二维矩阵，以及左上角坐标为（2，1）和右下角坐标为（4，3）的子矩阵，该函数输出8。
*/
func SumRegion(row1, col1, row2, col2 int, matrix [][]int) int {
	sum := 0
	for i := row1; i <= row2; i++ {
		for j := col1; j <= col2; j++ {
			sum += matrix[i][j]
		}
	}
	return sum
}

func SumRegionV2(row1, col1, row2, col2 int, matrix [][]int) int {
	sum := make([][]int, len(matrix)+1)
	for i := 0; i < len(matrix); i++ {
		sum[i] = make([]int, len(matrix[0])+1)
	}
	for i := 0; i < len(matrix); i++ {
		rowSum := 0
		for j := 0; j < len(matrix[0]); j++ {
			rowSum += matrix[i][j]
			sum[i+1][j+1] = sum[i][j+1] + rowSum
		}
	}
	return sum[row2+1][col2+1] - sum[row1][col2+1] - sum[row2+1][col1] + sum[row1][col1]
}
