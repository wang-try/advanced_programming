package high_frequency_summary

import "math"

// 动态规划 & 子序 & 子串

// leetcode 509. 斐波那契数
func fib(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 1
	}
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// leetcode 70 爬楼梯
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func biSearchIdx(incNums []int, target, incLth int) int {
	start := 0
	for start <= incLth {
		mid := (incLth + start) / 2
		if incNums[mid] > target {
			incLth = mid - 1
		} else if incNums[mid] < target {
			start = mid + 1
		} else {
			return mid
		}
	}
	return start
}

// leetcode 5 最长回文子串
func longestPalindrome(s string) string {
	maxPal := ""
	for i := 0; i < len(s); i++ {
		lhs, rhs := i, i
		for lhs > 0 && s[lhs] == s[lhs-1] {
			lhs--
		}
		for rhs < len(s)-1 && s[rhs] == s[rhs+1] {
			rhs++
		}
		for lhs >= 0 && rhs < len(s) {
			if s[lhs] == s[rhs] {
				lhs--
				rhs++
			} else {
				break
			}
		}
		if len(s[lhs+1:rhs]) > len(maxPal) {
			maxPal = s[lhs+1 : rhs]
		}
	}
	return maxPal
}

func longestPalindromeV2(s string) string {
	maxPal := ""
	dp := make([][]bool, len(s))
	for i := 0; i < len(s); i++ {
		dp[i] = make([]bool, len(s))
	}

	for i := len(s) - 1; i >= 0; i-- {
		for j := len(s) - 1; j >= i; j-- {
			if s[i] == s[j] && (i == j || j == i+1) {
				dp[i][j] = true
			} else if s[i] == s[j] && dp[i+1][j-1] {
				dp[i][j] = true
			}
			if dp[i][j] {
				if len(s[i:j+1]) > len(maxPal) {
					maxPal = s[i : j+1]
				}
			}
		}
	}
	return maxPal

}

// leetcode 674 最长连续递增序列
func findLengthOfLCIS(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 1
	maxL := 1
	for i := 1; i < len(nums); i++ {
		dp[i] = 1
		if nums[i] > nums[i-1] {
			dp[i] = dp[i-1] + 1
		}
		if dp[i] > maxL {
			maxL = dp[i]
		}
	}
	return maxL
}

// leetcode 300 最长递增子序列
func lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	maxLth := 0
	for i := 0; i < len(nums); i++ {
		dp[i] = 1
		for j := i - 1; j >= 0; j-- {
			if nums[i] > nums[j] {
				if dp[i] < dp[j]+1 {
					dp[i] = dp[j] + 1
				}
			}
		}
		if maxLth < dp[i] {
			maxLth = dp[i]
		}
	}
	return maxLth
}

func lengthOfLISV2(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	incNums := make([]int, len(nums))
	incNums[0] = nums[0]
	incLth := 0
	for i := 1; i < len(nums); i++ {
		idx := biSearchIdx(incNums, nums[i], incLth)
		incNums[idx] = nums[i]
		if idx == incLth+1 {
			incLth++
		}
	}

	return incLth + 1
}

// leetcode 673 最长递增子序列的个数（可不连续）
func findNumberOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	maxLth := 0
	cnt := make([]int, len(nums))
	ret := 0
	for i := 0; i < len(nums); i++ {
		dp[i] = 1
		cnt[i] = 1
		for j := i - 1; j >= 0; j-- {
			if nums[i] > nums[j] {
				if dp[j]+1 > dp[i] {
					dp[i] = dp[j] + 1
					cnt[i] = cnt[j]
				} else if dp[j]+1 == dp[i] {
					cnt[i] += cnt[j]
				}
			}
		}
		if dp[i] > maxLth {
			maxLth = dp[i]
			ret = cnt[i]
		} else if dp[i] == maxLth {
			ret += cnt[i]
		}
	}
	return ret
}

// leetcode 128 最长连续序列
func longestConsecutive(nums []int) int {
	num2exist := make(map[int]bool)
	for _, num := range nums {
		num2exist[num] = true
	}
	maxL := 0
	for _, num := range nums {
		if !num2exist[num-1] {
			curL := 1
			bigNum := num + 1
			for num2exist[bigNum] {
				curL++
				bigNum++
			}
			if curL > maxL {
				maxL = curL
			}
		}
	}
	return maxL
}

func longestConsecutiveV2(nums []int) int {
	numSet := map[int]bool{}
	for _, num := range nums {
		numSet[num] = true
	}
	longestStreak := 0
	for num := range numSet {
		if !numSet[num-1] {
			currentNum := num
			currentStreak := 1
			for numSet[currentNum+1] {
				currentNum++
				currentStreak++
			}
			if longestStreak < currentStreak {
				longestStreak = currentStreak
			}
		}
	}
	return longestStreak
}

// leetcode 3 无重复字符的最长子串
func lengthOfLongestSubstring(s string) int {
	ch2index := make(map[uint8]int)
	lhs := 0
	rhs := 0
	maxL := 0
	for rhs < len(s) {
		if index, ok := ch2index[s[rhs]]; !ok || lhs > index {
			lth := rhs - lhs + 1
			if lth > maxL {
				maxL = lth
			}
		} else {
			lhs = index + 1
		}
		ch2index[s[rhs]] = rhs
		rhs++
	}
	return maxL
}

// leetcode 1143 最长公共子序列
func longestCommonSubsequence(text1 string, text2 string) int {
	dp := make([][]int, len(text1))
	for i := 0; i < len(text1); i++ {
		dp[i] = make([]int, len(text2))
	}
	for i := 0; i < len(text1); i++ {
		for j := 0; j < len(text2); j++ {
			if text1[i] == text2[j] {
				if i == 0 || j == 0 {
					dp[i][j] = 1
				} else {
					dp[i][j] = dp[i-1][j-1] + 1
				}
			} else {
				if i > 0 {
					dp[i][j] = max(dp[i][j], dp[i-1][j])
				}
				if j > 0 {
					dp[i][j] = max(dp[i][j], dp[i][j-1])
				}
			}
		}
	}
	return dp[len(text1)-1][len(text2)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/**0-1背包问题**/
// leetcode 416 分隔等和子集
func canPartition(nums []int) bool {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if sum&1 == 1 {
		return false
	}
	target := sum / 2
	dp := make([][]bool, len(nums)+1)
	for i := 0; i <= len(nums); i++ {
		dp[i] = make([]bool, target+1)
	}
	dp[0][0] = true
	for i := 1; i <= len(nums); i++ {
		for j := 0; j <= target; j++ {
			dp[i][j] = dp[i-1][j]
			if !dp[i][j] && j >= nums[i-1] {
				dp[i][j] = dp[i-1][j-nums[i-1]]
			}
		}
	}
	return dp[len(nums)][target]
}

func canPartitionV2(nums []int) bool {
	lth := len(nums)
	if lth < 2 {
		return false
	}
	sum := 0

	for _, num := range nums {
		sum += num
	}
	if sum&1 == 1 {
		return false
	}
	half := sum / 2
	dp := make([]bool, half+1)
	dp[0] = true
	for i := 1; i <= lth; i++ {
		for j := half; j > 0; j-- {
			if !dp[j] && j >= nums[i-1] {
				dp[j] = dp[j-nums[i-1]]
			}
		}
	}
	return dp[half]
}

// LCR102 or 494 目标和
// +和 A -和B A+B=sum A-B=target 2A=sum+target A=(sum+target)/2,从数组中取出数组和为A的方法有多少
func findTargetSumWays(nums []int, target int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if target > sum || (target+sum)&1 == 1 {
		return 0
	}
	t := (target + sum) / 2
	if t < 0 {
		return 0
	}
	dp := make([][]int, len(nums)+1)
	for i := 0; i <= len(nums); i++ {
		dp[i] = make([]int, t+1)
	}
	dp[0][0] = 1

	for i := 1; i <= len(nums); i++ {
		for j := 0; j <= t; j++ {
			dp[i][j] = dp[i-1][j]
			if j >= nums[i-1] {
				dp[i][j] += dp[i-1][j-nums[i-1]]
			}
		}
	}
	return dp[len(nums)][t]
}

// leetcode322 零钱兑换
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = math.MaxInt32
		for _, coin := range coins {
			if i >= coin {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

// leetcode518 零钱兑换II
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}

// leetcode 72 编辑距离
func minDistance(word1 string, word2 string) int {
	lth1 := len(word1)
	lth2 := len(word2)
	dp := make([][]int, lth1+1)
	for i := 0; i <= lth1; i++ {
		dp[i] = make([]int, lth2+1)
		dp[i][0] = i
	}
	for i := 0; i <= lth2; i++ {
		dp[0][i] = i
	}

	for i := 1; i <= lth1; i++ {
		for j := 1; j <= lth2; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				//替换 dp[i-1][j-1]+1
				//增加 dp[i][j-1]+1
				//删除 dp[i-1][j]+1
				dp[i][j] = min(min(dp[i-1][j]+1, dp[i][j-1]+1), dp[i-1][j-1]+1)
			}
		}
	}
	return dp[lth1][lth2]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// leetcode 213 打家劫舍
func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	for i := 1; i < len(nums)-1; i++ {
		if i == 1 {
			dp[i] = max(dp[i-1], nums[i])
		} else {
			dp[i] = max(dp[i-1], dp[i-2]+nums[i])
		}
	}
	noLast := dp[len(nums)-2]
	dp[1] = nums[1]
	for i := 2; i < len(nums); i++ {
		if i == 2 {
			dp[i] = max(dp[i-1], nums[i])
		} else {
			dp[i] = max(dp[i-1], dp[i-2]+nums[i])
		}
	}
	return max(noLast, dp[len(nums)-1])
}

// leetcode279完全平方数
func numSquares(n int) int {
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = math.MaxInt32
		for j := 1; j*j <= i; j++ {
			dp[i] = min(dp[i], dp[i-j*j]+1)
		}
	}
	return dp[n]
}

// leetcode62 不同路径
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 || j == 0 {
				dp[i][j] = 1
			} else {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}

// leetcode 63 不同路径II
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	row := len(obstacleGrid)
	column := len(obstacleGrid[0])
	dp := make([][]int, row)
	for i := 0; i < row; i++ {
		dp[i] = make([]int, column)
	}
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if obstacleGrid[i][j] == 1 {
				dp[i][j] = 0
			} else {
				if i == 0 && j == 0 {
					dp[i][j] = 1
				} else if i == 0 {
					dp[i][j] = dp[i][j-1]
				} else if j == 0 {
					dp[i][j] = dp[i-1][j]
				} else {
					dp[i][j] = dp[i-1][j] + dp[i][j-1]
				}
			}
		}
	}

	return dp[row-1][column-1]
}

// leetcode64 最小路径和
func minPathSum(grid [][]int) int {
	m := len(grid)
	n := len(grid[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 && j == 0 {
				dp[i][j] = grid[i][j]
			} else if i == 0 {
				dp[i][j] = dp[i][j-1] + grid[i][j]
			} else if j == 0 {
				dp[i][j] = dp[i-1][j] + grid[i][j]
			} else {
				dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
			}
		}
	}
	return dp[m-1][n-1]
}

// leetcode 121买卖股票的最佳时机
func maxProfit(prices []int) int {
	minPrice := prices[0]
	mp := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] < minPrice {
			minPrice = prices[i]
		} else {
			mp = max(mp, prices[i]-minPrice)
		}
	}
	return mp
}

// leetcode122 买卖股票的最佳时机II
func maxProfitII(prices []int) int {
	lth := len(prices)
	hold := -prices[0]
	noHold := 0
	for i := 1; i < lth; i++ {
		hold = max(hold, noHold-prices[i])
		noHold = max(noHold, hold+prices[i])
	}
	return noHold
}

func maxProfitIIV2(prices []int) int {
	mp := 0
	for i := 0; i < len(prices)-1; i++ {
		if prices[i+1] > prices[i] {
			mp += prices[i+1] - prices[i]
		}
	}
	return mp
}

// leetcode53 最大子数组和
func maxSubArray(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	ret := dp[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(nums[i], dp[i-1]+nums[i])
		ret = max(ret, dp[i])
	}
	return ret
}

// leetcode76 最小覆盖子串
func minWindow(s string, t string) string {
	ch2cnt := make(map[uint8]int)
	for i := 0; i < len(t); i++ {
		ch2cnt[t[i]]++
	}
	lhs := 0
	rhs := 0
	minStr := ""
	for rhs < len(s) {
		if _, ok := ch2cnt[s[rhs]]; ok {
			ch2cnt[s[rhs]]--
			if isAllContain(ch2cnt) {
				for {
					cnt, ok2 := ch2cnt[s[lhs]]
					if !ok2 {
						lhs++
					} else if ok2 && cnt < 0 {
						ch2cnt[s[lhs]]++
						lhs++
					} else {
						break
					}
				}
				if len(s[lhs:rhs+1]) < len(minStr) || minStr == "" {
					minStr = s[lhs : rhs+1]
				}
			}
		}

		rhs++
	}
	return minStr
}

func isAllContain(ch2cnt map[uint8]int) bool {
	for _, v := range ch2cnt {
		if v > 0 {
			return false
		}
	}
	return true
}

// leetcode316 去除重复字母 ???
func removeDuplicateLetters(s string) string {
	left := [26]int{}
	for _, ch := range s {
		left[ch-'a']++
	}
	stack := []byte{}
	inStack := [26]bool{}
	for i := range s {
		ch := s[i]
		if !inStack[ch-'a'] {
			//如果当前栈不为空，且当前的字母比栈顶的字母小，可以将栈顶的字母出栈
			for len(stack) > 0 && ch < stack[len(stack)-1] {
				last := stack[len(stack)-1] - 'a'
				//如果该字母已经全部遍历过，退出循环，后续无法再进入栈中，所以此字母无法出栈
				if left[last] == 0 {
					break
				}
				stack = stack[:len(stack)-1]
				inStack[last] = false
			}
			stack = append(stack, ch)
			inStack[ch-'a'] = true
		}
		left[ch-'a']--
	}
	return string(stack)
}

// LCR 039 柱状图中最大的矩形
func largestRectangleArea(heights []int) int {
	var stack []int
	maxArea := 0
	for i, height := range heights {
		j := len(stack) - 1
		for ; j >= 0 && height < heights[stack[j]]; j-- {
			h := heights[stack[j]]
			var width int
			if j == 0 {
				width = i - (0 - 1) - 1
			} else {
				width = i - stack[j-1] - 1
			}
			maxArea = max(maxArea, h*width)
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}

	rhs := len(heights)
	for i := len(stack) - 1; i >= 0; i-- {
		lhs := -1
		if i > 0 {
			lhs = stack[i-1]
		}
		maxArea = max(maxArea, heights[stack[i]]*(rhs-lhs-1))
	}
	return maxArea
}
