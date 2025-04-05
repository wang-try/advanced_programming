package high_frequency_summary

import (
	"container/heap"
	"math"
	"sort"
	"strconv"
	"strings"
)

//数 & 数组 & 矩形 & 指针

// leetcode 84
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

// leetcode 189 轮转数组
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

// i,j分别表示计算到nums1和nums2中的哪个元素了，k表示第几小的元素
func find(nums1 []int, i int, nums2 []int, j int, k int) int {
	if len(nums1)-i > len(nums2)-j {
		return find(nums2, j, nums1, i, k)
	}

	//num1的长度小于nums2
	if k == 1 {
		//nums1 中已经没有可以算的元素了
		if i == len(nums1) {
			//返回当前nums2中最小的元素
			return nums2[j]
		} else {
			//选取nums1和nums2中最小的
			return min(nums1[i], nums2[j])
		}
	}

	//nums1 中已经没有可以算的元素了
	if i == len(nums1) {
		//返回当前nums2中最小的元素
		return nums2[j+k-1]
	}

	//折半计算nums1和nums2中的元素
	si := min(len(nums1), i+k/2)
	sj := j + k/2
	//哪个小选哪个
	if nums1[si-1] < nums2[sj-1] {
		return find(nums1, si, nums2, j, k-(si-i))
	} else {
		return find(nums1, i, nums2, sj, k-(sj-j))
	}
}

func findMedianSortedArraysV2(nums1 []int, nums2 []int) float64 {
	total := len(nums1) + len(nums2)
	mid1Pos, mid2Pos := total/2, total/2+1
	i, j := 0, 0
	lth1, lth2 := len(nums1), len(nums2)
	iter := 0
	mid1, mid2 := 0, 0
	for (i < lth1 || j < lth2) && iter < mid2Pos {
		if i == lth1 || j < lth2 && nums1[i] > nums2[j] {
			iter++
			if iter == mid1Pos {
				mid1 = nums2[j]
			}
			if iter == mid2Pos {
				mid2 = nums2[j]
			}
			j++
		} else {

			iter++
			if iter == mid1Pos {
				mid1 = nums1[i]
			}
			if iter == mid2Pos {
				mid2 = nums1[i]
			}
			i++
		}
	}
	if total&1 == 0 {
		return float64(mid1+mid2) / 2.0
	} else {
		return float64(mid2)
	}
}

// leetcode 41
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
	//原地hash
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

func threeSumClosestV2(nums []int, target int) int {
	sort.Ints(nums)
	closest := math.MaxInt
	ans := 0
	for i := 0; i < len(nums)-2; i++ {
		t := target - nums[i]
		lhs := i + 1
		rhs := len(nums) - 1
		for lhs < rhs {
			sum := nums[lhs] + nums[rhs]
			if sum == t {
				return target
			} else if sum > t {
				rhs--
				if closest > sum-t {
					closest = sum - t
					ans = sum + nums[i]
				}
			} else {
				lhs++
				if closest > t-sum {
					closest = t - sum
					ans = sum + nums[i]
				}
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

func findClosestElementsV2(arr []int, k int, x int) []int {
	pivot := findX(arr, x)
	cnt := 1
	lhs := pivot - 1
	rhs := pivot + 1
	for (lhs >= 0 || rhs < len(arr)) && cnt < k {
		diff1, diff2 := math.MaxInt32, math.MaxInt32
		if lhs >= 0 {
			diff1 = arr[lhs] - x
			if diff1 < 0 {
				diff1 *= -1
			}
		}
		if rhs < len(arr) {
			diff2 = arr[rhs] - x
			if diff2 < 0 {
				diff2 *= -1
			}
		}
		if diff1 <= diff2 {
			lhs--
		} else {
			rhs++
		}
		cnt++
	}

	return arr[lhs+1 : rhs]

}

func findX(arr []int, x int) int {
	lhs := 0
	rhs := len(arr) - 1
	closest := math.MaxInt32
	ans := 0
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if arr[mid] == x {
			return mid
		} else if arr[mid] > x {
			rhs = mid - 1
			if closest > arr[mid]-x {
				closest = arr[mid] - x
				ans = mid
			}
		} else {
			lhs = mid + 1
			if closest >= x-arr[mid] {
				closest = x - arr[mid]
				ans = mid
			}
		}
	}
	return ans

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

func splitIntoFibonacciV2(num string) []int {
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
		for i := index; i < len(num) && len(ret) == 0; i++ {
			s := num[index : i+1]
			n, _ := strconv.Atoi(s)
			if n > math.MaxInt32 {
				break
			}
			if len(ans) < 2 || (len(ans) >= 2 && n == ans[len(ans)-1]+ans[len(ans)-2]) {
				ans = append(ans, n)
				dfs(i+1, ans)
				ans = ans[:len(ans)-1]
			}
			if s == "0" {
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
			if subM == 0 || rhs-lhs+1 < subM {
				subM = rhs - lhs + 1
			}
		}
		rhs++
	}
	return subM
}

// leetcode560 和为k的子数组
func subarraySum(nums []int, k int) int {
	sum2cnt := make(map[int]int)
	sum2cnt[0] = 1
	sum := 0
	cnt := 0
	for _, num := range nums {
		sum += num
		//先求个数再将sum2cnt累加
		cnt += sum2cnt[sum-k]
		sum2cnt[sum]++
	}
	return cnt
}

// leetcode 862. 和至少为 K 的最短子数组
func shortestSubarray(nums []int, k int) int {
	//前缀和
	lth := len(nums)
	preSum := make([]int, lth+1)
	for i := 0; i < len(nums); i++ {
		preSum[i+1] = preSum[i] + nums[i]
	}
	ans := lth + 1
	//单调双端队列
	var queue []int
	for i, sum := range preSum {
		for len(queue) > 0 && sum-preSum[queue[0]] >= k {
			ans = min(ans, i-queue[0])
			queue = queue[1:]
		}

		for len(queue) > 0 && sum < preSum[queue[len(queue)-1]] {
			queue = queue[:len(queue)-1]
		}
		queue = append(queue, i)
	}
	if ans < lth+1 {
		return ans
	}
	return -1
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

	if strList[0] == "0" {
		return "0"
	}
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
		if (mid == 0 && nums[mid] > nums[mid+1]) ||
			(mid == len(nums)-1 && nums[mid] > nums[mid-1]) ||
			(mid > 0 && mid < len(nums) && nums[mid] > nums[mid-1] && nums[mid] > nums[mid+1]) {
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
		for len(stack) > 0 && h > height[stack[len(stack)-1]] {
			if len(stack) >= 2 {
				top := stack[len(stack)-1]
				left := stack[len(stack)-2]
				hei := min(h, height[left]) - height[top]
				width := i - left - 1
				contain += hei * width
			}
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return contain
}

// leetcode153  旋转数组中的最小数字
func findMin(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		pivot := low + (high-low)/2
		if nums[pivot] < nums[high] {
			high = pivot
		} else {
			low = pivot + 1
		}
	}
	return nums[low]
}

// leetcode154 旋转数组中的最小数字II
func findMinII(nums []int) int {
	lhs := 0
	rhs := len(nums) - 1
	for lhs < rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] < nums[rhs] {
			rhs = mid
		} else if nums[mid] > nums[rhs] {
			lhs = mid + 1
		} else {
			rhs--
		}
	}
	return nums[lhs]
}

// leetcode 240 搜索二维矩阵
func searchMatrix(matrix [][]int, target int) bool {
	i := len(matrix) - 1
	j := 0
	for i >= 0 && i < len(matrix) && j >= 0 && j < len(matrix[0]) {
		if target == matrix[i][j] {
			return true
		} else if target > matrix[i][j] {
			j++
		} else {
			i--
		}
	}
	return false
}

// leetcode 9. 回文数
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}

	str := strconv.Itoa(x)
	lhs := 0
	rhs := len(str) - 1
	for lhs <= rhs {
		if str[lhs] != str[rhs] {
			return false
		}
		lhs++
		rhs--
	}
	return true
}

func isPalindromeV2(x int) bool {
	// 特殊情况：
	// 如上所述，当 x < 0 时，x 不是回文数。
	// 同样地，如果数字的最后一位是 0，为了使该数字为回文，
	// 则其第一位数字也应该是 0
	// 只有 0 满足这一属性
	if x < 0 || (x%10 == 0 && x != 0) {
		return false
	}

	revertedNumber := 0
	for x > revertedNumber {
		revertedNumber = revertedNumber*10 + x%10
		x /= 10
	}

	// 当数字长度为奇数时，我们可以通过 revertedNumber/10 去除处于中位的数字。
	// 例如，当输入为 12321 时，在 while 循环的末尾我们可以得到 x = 12，revertedNumber = 123，
	// 由于处于中位的数字不影响回文（它总是与自己相等），所以我们可以简单地将其去除。
	return x == revertedNumber || x == revertedNumber/10
}

// leetcode 349. 两个数组的交集
func intersection(nums1 []int, nums2 []int) []int {
	var ans []int
	num12exist := make(map[int]bool)
	for _, num := range nums1 {
		num12exist[num] = true
	}

	a2exist := make(map[int]bool)
	for _, num := range nums2 {
		if _, ok := num12exist[num]; ok && !a2exist[num] {
			ans = append(ans, num)
			a2exist[num] = true
		}
	}
	return ans
}

// leetcode 350. 两个数组的交集 II
func intersect(nums1 []int, nums2 []int) []int {
	var ans []int
	num2cnt := make(map[int]int)
	for _, num := range nums1 {
		num2cnt[num]++
	}

	for _, num := range nums2 {
		if cnt, ok := num2cnt[num]; ok && cnt > 0 {
			ans = append(ans, num)
			num2cnt[num]--
		}
	}
	return ans
}

// leetcode31 下一个排列
func nextPermutation(nums []int) {
	splitIndex := len(nums) - 2
	for splitIndex >= 0 && nums[splitIndex] >= nums[splitIndex+1] {
		splitIndex--
	}

	if splitIndex >= 0 {
		for index := len(nums) - 1; index > splitIndex; index-- {
			if nums[index] > nums[splitIndex] {
				nums[index], nums[splitIndex] = nums[splitIndex], nums[index]
				break
			}
		}
	}

	lhs := splitIndex + 1
	rhs := len(nums) - 1
	for lhs < rhs {
		nums[lhs], nums[rhs] = nums[rhs], nums[lhs]
		lhs++
		rhs--
	}
}

// leetcode 503. 下一个更大元素 II
func nextGreaterElements(nums []int) []int {
	n := len(nums)
	ans := make([]int, n)
	for i := range ans {
		ans[i] = -1
	}
	var stack []int
	for i := 0; i < n*2-1; i++ {
		for len(stack) > 0 && nums[stack[len(stack)-1]] < nums[i%n] {
			ans[stack[len(stack)-1]] = nums[i%n]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i%n)
	}
	return ans
}

// leetcode239. 滑动窗口最大值
func maxSlidingWindow(nums []int, k int) []int {
	var stack []int
	for i := 0; i < k; i++ {
		for len(stack) > 0 && nums[i] > nums[stack[len(stack)-1]] {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	var ans []int
	ans = append(ans, nums[stack[0]])
	for i := k; i < len(nums); i++ {
		for len(stack) > 0 && nums[i] > nums[stack[len(stack)-1]] {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
		if stack[0] == i-k {
			stack = stack[1:]
		}
		ans = append(ans, nums[stack[0]])
	}
	return ans
}

// ？？？放弃 TODO
// leetcode 480. 滑动窗口中位数
type hp struct {
	sort.IntSlice
	size int
}

func (h *hp) Push(v interface{}) {
	h.IntSlice = append(h.IntSlice, v.(int))
}
func (h *hp) Pop() interface{} {
	a := h.IntSlice
	v := a[len(a)-1]
	h.IntSlice = a[:len(a)-1]
	return v
}
func (h *hp) push(v int) {
	h.size++
	heap.Push(h, v)
}
func (h *hp) pop() int {
	h.size--
	return heap.Pop(h).(int)
}
func (h *hp) prune() {
	for h.Len() > 0 {
		num := h.IntSlice[0]
		if h == small {
			num = -num
		}
		if d, has := delayed[num]; has {
			if d > 1 {
				delayed[num]--
			} else {
				delete(delayed, num)
			}
			heap.Pop(h)
		} else {
			break
		}
	}
}

var delayed map[int]int
var small, large *hp

func medianSlidingWindow(nums []int, k int) []float64 {
	delayed = map[int]int{} // 哈希表，记录「延迟删除」的元素，key 为元素，value 为需要删除的次数
	small = &hp{}           // 大根堆，维护较小的一半元素
	large = &hp{}           // 小根堆，维护较大的一半元素
	makeBalance := func() {
		// 调整 small 和 large 中的元素个数，使得二者的元素个数满足要求
		if small.size > large.size+1 { // small 比 large 元素多 2 个
			large.push(-small.pop())
			small.prune() // small 堆顶元素被移除，需要进行 prune
		} else if small.size < large.size { // large 比 small 元素多 1 个
			small.push(-large.pop())
			large.prune() // large 堆顶元素被移除，需要进行 prune
		}
	}
	insert := func(num int) {
		if small.Len() == 0 || num <= -small.IntSlice[0] {
			small.push(-num)
		} else {
			large.push(num)
		}
		makeBalance()
	}
	erase := func(num int) {
		delayed[num]++
		if num <= -small.IntSlice[0] {
			small.size--
			if num == -small.IntSlice[0] {
				small.prune()
			}
		} else {
			large.size--
			if num == large.IntSlice[0] {
				large.prune()
			}
		}
		makeBalance()
	}
	getMedian := func() float64 {
		if k&1 > 0 {
			return float64(-small.IntSlice[0])
		}
		return float64(-small.IntSlice[0]+large.IntSlice[0]) / 2
	}

	for _, num := range nums[:k] {
		insert(num)
	}
	n := len(nums)
	ans := make([]float64, 1, n-k+1)
	ans[0] = getMedian()
	for i := k; i < n; i++ {
		insert(nums[i])
		erase(nums[i-k])
		ans = append(ans, getMedian())
	}
	return ans
}

// TODO leetcode LCR 160. 数据流中的中位数
type MedianFinder struct {
}

/** initialize your data structure here. */
func Constructor() MedianFinder {
	return MedianFinder{}
}

func (this *MedianFinder) AddNum(num int) {

}

func (this *MedianFinder) FindMedian() float64 {
	return 0
}

// leetcode977. 有序数组的平方
func sortedSquares(nums []int) []int {
	lhs := 0
	rhs := len(nums) - 1
	ans := make([]int, len(nums))
	iterIndex := len(nums) - 1
	for lhs <= rhs {
		num := 0
		if nums[lhs] < 0 && -nums[lhs] > nums[rhs] {
			num = nums[lhs] * nums[lhs]
			lhs++
		} else {
			num = nums[rhs] * nums[rhs]
			rhs--
		}
		ans[iterIndex] = num
		iterIndex--
	}
	return ans
}

// leetcode69. x 的平方根
func mySqrt(x int) int {
	lhs := 0
	rhs := x
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if mid*mid == x {
			return mid
		} else if mid*mid < x {
			lhs = mid + 1
		} else if mid*mid > x {
			if (mid-1)*(mid-1) < x {
				return mid - 1
			} else {
				rhs = mid - 1
			}
		}
	}
	return -1
}

// leetcode LCR 164. 破解闯关密码
func crackPassword(password []int) string {
	var list []string
	for _, num := range password {
		list = append(list, strconv.Itoa(num))
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i]+list[j] < list[j]+list[i]
	})
	return strings.Join(list, "")
}

// leetcode562. 矩阵中最长的连续1线段
func longestLine(mat [][]int) int {
	maxL := 0
	dp := make([][][]int, 4)
	for i := 0; i < 4; i++ {
		dp[i] = make([][]int, len(mat))
		for j := 0; j < len(mat); j++ {
			dp[i][j] = make([]int, len(mat[0]))
		}
	}

	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[0]); j++ {
			if mat[i][j] == 1 {
				//行
				dp[0][i][j] = 1
				//竖
				dp[1][i][j] = 1
				//斜
				dp[2][i][j] = 1
				//反斜
				dp[3][i][j] = 1
				if j > 0 {
					dp[0][i][j] = dp[0][i][j-1] + 1
				}

				if i > 0 {
					dp[1][i][j] = dp[1][i-1][j] + 1
				}

				if i > 0 && j > 0 {
					dp[2][i][j] = dp[2][i-1][j-1] + 1
				}

				if i > 0 && j < len(mat[0])-1 {
					dp[3][i][j] = dp[3][i-1][j+1] + 1
				}
				a := max(dp[0][i][j], dp[1][i][j])
				b := max(dp[2][i][j], dp[3][i][j])
				m := max(a, b)
				maxL = max(maxL, m)
			}
		}

	}

	return maxL
}

// leetcode40. 组合总和 II
func combinationSum2(candidates []int, target int) [][]int {
	var ans [][]int
	var dfs func(index, sum int, base []int)

	dfs = func(index, sum int, base []int) {
		if sum == target {
			ans = append(ans, append([]int{}, base...))
			return
		}
		for i := index; i < len(candidates); i++ {
			if sum > target {
				break
			}
			if i > index && candidates[i] == candidates[i-1] {
				continue
			}
			sum += candidates[i]
			base = append(base, candidates[i])
			dfs(i+1, sum, base)
			sum -= candidates[i]
			base = base[:len(base)-1]
		}

	}
	sort.Ints(candidates)
	dfs(0, 0, []int{})

	return ans
}

// leetcode560. 和为 K 的子数组
func subarraySumII(nums []int, k int) int {
	sum2cnt := make(map[int]int)
	sum2cnt[0] = 1
	sum := 0
	cnt := 0
	for _, num := range nums {
		sum += num
		//cnt语句要在sum2cnt前面
		cnt += sum2cnt[sum-k]
		sum2cnt[sum]++
	}
	return cnt
}

func findDuplicates(nums []int) []int {
	var ans []int
	index := 0
	for index < len(nums) {
		if nums[index] > 0 {
			if nums[index] != index+1 {
				num := nums[index]
				pos := num - 1
				if nums[pos] == num {
					ans = append(ans, num)
					nums[index] = 0
				} else {
					nums[index], nums[pos] = nums[pos], nums[index]
				}
			} else {
				index++
			}
		} else {
			index++
		}
	}
	return ans
}

func findDuplicatesV2(nums []int) (ans []int) {
	for i := range nums {
		for nums[i] != nums[nums[i]-1] {
			nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
		}
	}
	for i, num := range nums {
		if num-1 != i {
			ans = append(ans, num)
		}
	}
	return
}

// leetcode LCR 139. 训练计划 I
func trainingPlan(actions []int) []int {
	lhs := 0
	rhs := len(actions) - 1
	index := 0
	for lhs <= rhs {
		if actions[index]&1 == 0 {
			actions[index], actions[rhs] = actions[rhs], actions[index]
			rhs--
		} else {
			index++
			lhs++
		}
	}
	return actions
}
