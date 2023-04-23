package array

import "sort"

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
