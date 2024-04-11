package high_frequency_summary

import (
	"math"
	"sort"
	"strconv"
	"strings"
)

//数 & 数组 & 矩形 & 指针

func largestRectangleAreaV2(heights []int) int {
	var stack []int
	lr := 0
	for i, height := range heights {
		for len(stack) > 0 && height < heights[stack[len(stack)-1]] {
			h := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			width := 0
			if len(stack) > 0 {
				width = i - stack[len(stack)-1] - 1
			} else {
				width = i - (0 - 1) - 1
			}
			lr = max(lr, width*h)
		}
		stack = append(stack, i)
	}

	rhs := len(heights)
	for i := len(stack) - 1; i >= 0; i-- {
		h := heights[stack[i]]
		lhs := 0 - 1
		if i > 0 {
			lhs = stack[i-1]
		}
		width := rhs - lhs - 1
		lr = max(lr, width*h)
	}
	return lr
}

// leetcode85 最大矩形
func maximalRectangle(matrix [][]byte) int {
	height := make([]int, len(matrix[0]))
	maxArea := 0
	for _, str := range matrix {
		for i := 0; i < len(str); i++ {
			if str[i] == '0' {
				height[i] = 0
			} else {
				height[i]++
			}
		}
		area := largestRectangleAreaV2(height)
		if maxArea < area {
			maxArea = area
		}
	}
	return maxArea
}

/*
"1","0","1","0","0"
"1","0","1","1","1"
"1","1","1","1","1"
"1","0","0","1","0"
*/
// leetcod221 最大正方形
func maximalSquare(matrix [][]byte) int {
	dp := make([][]int, len(matrix))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(matrix[0]))
	}

	maxSide := 0
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if matrix[i][j] == '1' {
				if i == 0 || j == 0 {
					dp[i][j] = 1
				} else {
					dp[i][j] = min(min(dp[i][j-1], dp[i-1][j-1]), dp[i-1][j]) + 1
				}
			}
			maxSide = max(maxSide, dp[i][j])
		}
	}
	return maxSide * maxSide
}

// leetcode33 搜索旋转排序数组
func search(nums []int, target int) int {
	lhs := 0
	rhs := len(nums) - 1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < nums[rhs] {
			if target > nums[mid] && target <= nums[rhs] {
				lhs = mid + 1
			} else {
				rhs = mid - 1
			}

		} else {
			if target >= nums[lhs] && target < nums[mid] {
				rhs = mid - 1
			} else {
				lhs = mid + 1
			}
		}
	}
	return -1
}

// leetcode 轮转数组
func rotate(nums []int, k int) {
	k %= len(nums)
	reverse(nums, 0, len(nums)-1)
	reverse(nums, 0, k-1)
	reverse(nums, k, len(nums)-1)
}

func reverse(nums []int, start, end int) {
	for start <= end {
		nums[start], nums[end] = nums[end], nums[start]
		start++
		end--
	}
}

// leetcode54 螺旋矩阵
func spiralOrder(matrix [][]int) []int {
	var ans []int
	times := 0
	i, j := 0, 0
	cnt := 0
	isDone := len(matrix) * len(matrix[0])
	for {
		if cnt == isDone {
			return ans
		}
		//右
		for ; cnt < isDone && j < len(matrix[0])-times; j++ {
			ans = append(ans, matrix[i][j])
			cnt++
		}
		//下
		j--
		for i = i + 1; cnt < isDone && i < len(matrix)-times; i++ {
			ans = append(ans, matrix[i][j])
			cnt++
		}
		//左
		i--
		for j = j - 1; cnt < isDone && j >= times; j-- {
			ans = append(ans, matrix[i][j])
			cnt++
		}
		times++
		//上
		j++
		for i = i - 1; cnt < isDone && i >= times; i-- {
			ans = append(ans, matrix[i][j])
			cnt++
		}
		i++
		j++

	}
}

// leetcode4 寻找两个正序数组的中位数
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	tot := len(nums1) + len(nums2)
	if tot&1 == 0 {
		left := find(nums1, 0, nums2, 0, tot/2)
		right := find(nums1, 0, nums2, 0, tot/2+1)
		return float64(left+right) / 2.0
	} else {
		return float64(find(nums1, 0, nums2, 0, tot/2+1))
	}

}

func find(nums1 []int, i int, nums2 []int, j int, k int) int {
	if len(nums1)-i > len(nums2)-j {
		return find(nums2, j, nums1, i, k)
	}
	if k == 1 {
		if i == len(nums1) {
			return nums2[j]
		} else {
			return min(nums1[i], nums2[j])
		}
	}
	if i == len(nums1) {
		return nums2[j+k-1]
	}
	si := min(len(nums1), i+k/2)
	sj := j + k/2
	if nums1[si-1] < nums2[sj-1] {
		return find(nums1, si, nums2, j, k-(si-i))
	} else {
		return find(nums1, i, nums2, sj, k-(sj-j))
	}
}

func firstMissingPositive(nums []int) int {
	lth := len(nums)
	index := 0
	for index < len(nums) {
		num := nums[index]
		if num == index+1 {
			index++
		} else if num > 0 && index+1 != num {
			if num < lth && nums[num-1] != num {
				nums[index], nums[num-1] = nums[num-1], nums[index]
			} else {
				index++
			}
		} else {
			index++
		}
	}

	for i := 0; i < len(nums); i++ {
		if (i + 1) != nums[i] {
			return i + 1
		}
	}
	return len(nums) + 1
}

// leetcode268 丢失的数字
func missingNumber(nums []int) int {
	lth := len(nums)
	allSum := lth * (lth + 1) / 2
	for _, num := range nums {
		allSum -= num
	}
	return allSum
}

// leetcode448 找到所有数组中消失的数字
func findDisappearedNumbers(nums []int) []int {
	for i := 0; i < len(nums); i++ {
		num := nums[i]
		for num != i+1 && nums[num-1] != num {
			nums[i], nums[num-1] = nums[num-1], nums[i]
			num = nums[i]
		}
	}
	var ans []int
	for i := 0; i < len(nums); i++ {
		if i+1 != nums[i] {
			ans = append(ans, i+1)
		}
	}
	return ans
}

func findDisappearedNumbersV2(nums []int) (ans []int) {
	n := len(nums)
	for _, v := range nums {
		v = (v - 1) % n
		nums[v] += n
	}
	for i, v := range nums {
		if v <= n {
			ans = append(ans, i+1)
		}
	}
	return
}

// leetcode15 三数之和
func threeSum(nums []int) [][]int {
	sort.Slice(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	var ans [][]int
	for i := 0; i < len(nums)-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		target := -nums[i]
		lhs := i + 1
		rhs := len(nums) - 1
		for lhs < rhs {
			sum := nums[lhs] + nums[rhs]
			if sum == target {
				ans = append(ans, []int{nums[i], nums[lhs], nums[rhs]})
				lhs++
				rhs--
				for ; lhs < rhs && nums[lhs] == nums[lhs-1]; lhs++ {
				}
				for ; lhs < rhs && nums[rhs] == nums[rhs+1]; rhs-- {
				}
			} else if sum < target {
				lhs++
			} else {
				rhs--
			}

		}
	}
	return ans
}

// leetcode16 最接近的三数之和
func threeSumClosest(nums []int, target int) int {
	sort.Slice(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})
	ans := nums[0] + nums[1] + nums[2]
	for i := 0; i < len(nums)-2; i++ {
		t := target - nums[i]
		lhs := i + 1
		rhs := len(nums) - 1
		for lhs < rhs {
			sum := nums[lhs] + nums[rhs]
			if t == sum {
				return target
			} else if sum < t {
				diff := t - sum
				ma := max(ans, target)
				mi := min(ans, target)
				if ma-mi > diff {
					ans = sum + nums[i]
				}
				lhs++
			} else {
				diff := sum - t
				ma := max(ans, target)
				mi := min(ans, target)
				if ma-mi > diff {
					ans = sum + nums[i]
				}
				rhs--
			}
		}
	}
	return ans
}

// leetcode628. 三个数的最大乘积
func maximumProduct(nums []int) int {
	min1, min2 := math.MaxInt32, math.MaxInt32
	max1, max2, max3 := math.MinInt32, math.MinInt32, math.MinInt32

	for _, num := range nums {
		if num < min1 {
			min2 = min1
			min1 = num
		} else if num < min2 {
			min2 = num
		}
		if num > max1 {
			max3 = max2
			max2 = max1
			max1 = num

		} else if num > max2 {
			max3 = max2
			max2 = num
		} else if num > max3 {
			max3 = num
		}
	}
	return max(max1*max2*max3, max1*min1*min2)
}

// leetcode 658. 找到 K 个最接近的元素
func findClosestElements(arr []int, k int, x int) []int {
	startIndex := findCloestIndex(arr, x)
	lhs := startIndex - 1
	rhs := startIndex + 1
	k--
	for k > 0 {
		diff1 := math.MaxInt32
		diff2 := math.MaxInt32
		if lhs >= 0 {
			ma := max(arr[lhs], x)
			mi := min(arr[lhs], x)
			diff1 = ma - mi
		}
		if rhs < len(arr) {
			ma := max(arr[rhs], x)
			mi := min(arr[rhs], x)
			diff2 = ma - mi
		}
		if diff1 <= diff2 {
			lhs--
		} else {
			rhs++

		}
		k--
	}
	return arr[lhs+1 : rhs]
}

// [1,2,3,4,5] 4, 3
func findCloestIndex(arr []int, x int) int {
	lhs := 0
	rhs := len(arr) - 1
	mC := 0
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if arr[mid] == x {
			return mid
		} else if arr[mid] > x {
			diff := arr[mid] - x
			ma := max(arr[mC], x)
			mi := min(arr[mC], x)
			if ma-mi > diff || (ma-mi == diff && mid < mC) {
				mC = mid
			}
			rhs = mid - 1
		} else {
			diff := x - arr[mid]
			ma := max(arr[mC], x)
			mi := min(arr[mC], x)
			if ma-mi > diff || (ma-mi == diff && mid < mC) {
				mC = mid
			}
			lhs = mid + 1
		}
	}
	return mC
}

// leetcode88 合并两个有序数组
func merge(nums1 []int, m int, nums2 []int, n int) {
	index1, index2 := m-1, n-1
	iterIndex := m + n - 1
	for index1 >= 0 || index2 >= 0 {
		if index1 < 0 || index2 >= 0 && nums2[index2] > nums1[index1] {
			nums1[iterIndex] = nums2[index2]
			index2--
			iterIndex--
		} else {
			nums1[iterIndex] = nums1[index1]
			index1--
			iterIndex--
		}
	}
}

// leetcode7 整数反转
func reverseInt(x int) int {
	ans := 0
	sign := 1
	if x < 0 {
		sign = -1
		x = -x
	}

	for x > 0 {
		num := x % 10
		ans = ans*10 + num
		if ans*sign < math.MinInt32 || ans*sign > math.MaxInt32 {
			return 0
		}
		x /= 10
	}
	return ans * sign
}

// leetcode 11盛最多水的容器
func maxArea(height []int) int {
	ma := 0
	lhs := 0
	rhs := len(height)
	for lhs < rhs {
		h := min(height[lhs], height[rhs])
		width := rhs - lhs
		ma = max(ma, h*width)
		if height[lhs] < height[rhs] {
			lhs++
		} else {
			rhs--
		}
	}
	return ma
}

// leetcode26删除排序数组中的重复项目
func removeDuplicates(nums []int) int {
	iterIndex := 1
	for i := 1; i < len(nums); i++ {
		if nums[i] != nums[iterIndex-1] {
			nums[iterIndex] = nums[i]
			iterIndex++
		}
	}
	return iterIndex + 1
}

func permuteV2(nums []int) [][]int {
	var ret [][]int
	var dfs func(index int, nums []int)
	dfs = func(index int, nums []int) {
		if index == len(nums) {
			ret = append(ret, append([]int{}, nums...))
			return
		}
		for i := index; i < len(nums); i++ {
			nums[i], nums[index] = nums[index], nums[i]
			dfs(index+1, nums)
			nums[i], nums[index] = nums[index], nums[i]
		}
	}
	dfs(0, nums)
	return ret
}

// leetcode287 寻找重复数
func findDuplicate(nums []int) int {
	index := 0
	for index < len(nums) {
		if index+1 == nums[index] {
			index++
		} else {
			num := nums[index]
			if nums[num-1] != num {
				nums[num-1], nums[index] = nums[index], nums[num-1]
			} else {
				return num
			}
		}
	}
	return -1
}

// leetcode166 分数到小数
func fractionToDecimal(numerator int, denominator int) string {
	if numerator == 0 {
		return "0"
	}
	sign := 1
	ns := 1
	if numerator < 0 {
		numerator = -numerator
		ns = 0
	}
	ds := 1
	if denominator < 0 {
		denominator = -denominator
		ds = 0
	}
	if ns^ds == 1 {
		sign = -1
	}

	remain := numerator % denominator
	k := numerator / denominator
	var ans string
	if sign == -1 {
		ans += "-"
	}
	ans += strconv.Itoa(k)
	if remain > 0 {
		ans += "."
	}
	r2exist := make(map[int]int)
	pos := 0
	var decimals []int
	isCyclePos := -1
	for remain != 0 {
		remain *= 10
		if p, ok := r2exist[remain]; ok {
			isCyclePos = p
			break
		}
		r2exist[remain] = pos
		n := remain / denominator
		remain = remain % denominator
		decimals = append(decimals, n)
		pos++
	}
	for i := 0; i < len(decimals); i++ {
		if i == isCyclePos {
			ans += "(" + strconv.Itoa(decimals[i])
		} else {
			ans += strconv.Itoa(decimals[i])
		}
	}
	if isCyclePos != -1 {
		ans += ")"
	}
	return ans
}

// leetcode842 将数组拆分成斐波那契数列
func splitIntoFibonacci(num string) []int {

	var ret []int
	var dfs func(index int, ans []int)
	dfs = func(index int, ans []int) {
		if len(ret) > 0 {
			return
		}
		if index == len(num) && len(ans) >= 3 {
			ret = append([]int{}, ans...)
			return
		}
		for i := index; i < len(num); i++ {
			str := num[index : i+1]
			nt, _ := strconv.Atoi(str)
			if nt > math.MaxInt {
				break
			}
			if len(ans) < 2 {
				ans = append(ans, nt)
				dfs(i+1, ans)
				ans = ans[:len(ans)-1]
			} else {
				lth := len(ans) - 1
				if nt == ans[lth]+ans[lth-1] {
					ans = append(ans, nt)
					dfs(i+1, ans)
					ans = ans[:len(ans)-1]
				}
			}
			if str == "0" {
				break
			}

		}
	}
	dfs(0, []int{})
	return ret

}

// leetcode209. 长度最小的子数组
func minSubArrayLen(target int, nums []int) int {
	subM := 0
	lhs, rhs := 0, 0
	sum := 0
	for rhs < len(nums) {
		sum += nums[rhs]
		if sum >= target {
			for lhs < rhs && sum-nums[lhs] >= target {
				sum -= nums[lhs]
				lhs++
			}
			if subM == 0 || rhs-lhs-1+1 < subM {
				subM = rhs - lhs - 1 + 1
			}
		}
		rhs++
	}
	return subM
}

// leetcode56 合并区间
func mergeInter(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	start := intervals[0][0]
	end := intervals[0][1]
	var ans [][]int
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] <= end {
			if intervals[i][1] > end {
				end = intervals[i][1]
			}
		} else {
			ans = append(ans, []int{start, end})
			start = intervals[i][0]
			end = intervals[i][1]
		}
	}
	ans = append(ans, []int{start, end})
	return ans
}

// leetcode283 移动零值
func moveZeroes(nums []int) {
	iterIndex := 0
	for _, num := range nums {
		if num != 0 {
			nums[iterIndex] = num
			iterIndex++
		}
	}
	for ; iterIndex < len(nums); iterIndex++ {
		nums[iterIndex] = 0
	}
}

func moveZeroesV2(nums []int) {
	swapIndex := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[i], nums[swapIndex] = nums[swapIndex], nums[i]
			swapIndex++
		}
	}
}

// leetcode179 最大数
func largestNumber(nums []int) string {
	strList := make([]string, len(nums))
	for i, num := range nums {
		strList[i] = strconv.Itoa(num)
	}
	sort.Slice(strList, func(i, j int) bool {
		return strList[i]+strList[j] > strList[j]+strList[i]
	})
	return strings.Join(strList, "")
}

// leetcode169 多数元素
func majorityElement(nums []int) int {
	candidate := nums[0]
	cnt := 0
	for _, num := range nums {
		if cnt == 0 {
			candidate = num
			cnt++
		} else {
			if candidate == num {
				cnt++
			} else {
				cnt--
			}
		}
	}
	return candidate
}

// leetcode53 最大子数组和
func maxSubArrayV2(nums []int) int {
	dp := make([]int, len(nums))
	maxSub := nums[0]
	dp[0] = nums[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(nums[i], dp[i-1]+nums[i])
		maxSub = max(maxSub, dp[i])
	}
	return maxSub
}

// leetcode162 寻找峰值
func findPeakElement(nums []int) int {
	lhs, rhs := 0, len(nums)-1
	for lhs < rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] > nums[mid+1] {
			rhs = mid
		} else {
			lhs = mid + 1
		}
	}
	return rhs
}

func findPeakElementV2(nums []int) int {
	if len(nums) == 1 {
		return 0
	}
	lhs, rhs := 0, len(nums)-1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if (mid == 0 && nums[mid] > nums[mid+1]) || (mid == len(nums)-1 && nums[mid] > nums[mid-1]) || (mid > 0 && mid < len(nums) && nums[mid] > nums[mid-1] && nums[mid] > nums[mid+1]) {
			return mid
		} else if nums[mid] < nums[mid+1] {
			lhs = mid + 1
		} else {
			rhs = mid - 1
		}
	}
	return -1
}

// leetcode78 子集
func subsets(nums []int) [][]int {
	var ans [][]int
	var dfs func(index int, subs []int)
	dfs = func(index int, subs []int) {
		ans = append(ans, append([]int{}, subs...))
		if index == len(nums) {
			return
		}
		for i := index; i < len(nums); i++ {
			subs = append(subs, nums[i])
			dfs(i+1, subs)
			subs = subs[:len(subs)-1]
		}
	}
	dfs(0, []int{})
	return ans
}

// leetcode42 接雨水
func trap(height []int) int {
	contain := 0
	var stack []int
	for i, h := range height {
		for len(stack) > 0 && h < height[stack[len(stack)-1]] {

		}
		stack = append(stack, i)
	}
}
