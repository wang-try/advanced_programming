package algorithm

import (
	"fmt"
	"math"
	"sort"
)

/*********动态规划************/

// 动态规划解目标和
// leetcode 494. Target Sum
func findTargetSumWays(nums []int, target int) int {
	cnt := 0
	dfsFindTargetSumWays(nums, target, 0, 0, &cnt)
	return cnt
}

func dfsFindTargetSumWays(nums []int, target, current, index int, cnt *int) {
	if index == len(nums) {
		if current == target {
			*cnt++
		}
	}
	dfsFindTargetSumWays(nums, target, current+nums[index], index+1, cnt)
	dfsFindTargetSumWays(nums, target, current-nums[index], index+1, cnt)
}

func findTargetSumWaysV2(nums []int, S int) int {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if (sum+S)%2 == 1 || S > sum {
		return 0
	}
	target := (sum + S) / 2
	dp := make([]int, sum+1)
	dp[0] = 1
	for _, num := range nums {
		for j := sum; j >= num; j-- {
			dp[j] += dp[j-num]
		}
	}
	return dp[target]
}

// 动态规划解分割子集
// leetcode 416. Partition Equal Subset Sum
// 二维dp
func canPartition(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%2 == 1 {
		return false
	}
	target := sum >> 1
	lth := len(nums)
	dp := make([][]bool, lth+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, target+1)
	}
	dp[0][0] = true
	for i := 1; i <= lth; i++ {
		for j := 1; j <= target; j++ {
			if nums[i-1] <= j {
				dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i-1]]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}
	return dp[lth][target]
}

// 一维dp,倒序处理还是不懂
func canPartitionV2(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum&1 == 1 {
		return false
	}
	target := sum >> 1
	dp := make([]bool, target+1)
	dp[0] = true
	for i := 1; i <= len(nums); i++ {
		for j := target; j >= 1; j-- {
			if j >= nums[i-1] {
				dp[j] = dp[j] || dp[j-nums[i-1]]
			}
		}
	}
	return dp[target]
}

func canPartitionV3(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum&1 == 1 {
		return false
	}
	target := sum >> 1
	return dfsCanPartitionV3(nums, target, 0)
}

func dfsCanPartitionV3(nums []int, target, index int) bool {
	lth := len(nums)

	if target == 0 {
		return true
	}
	if index == lth || target < 0 {
		return false
	}

	return dfsCanPartitionV3(nums, target-nums[index], index+1) || dfsCanPartitionV3(nums, target, index)
}

func canPartitionV4(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum&1 == 1 {
		return false
	}
	target := sum >> 1
	target2judge := make(map[int]bool)
	return dfsCanPartitionV4(nums, target, 0, target2judge)
}

func dfsCanPartitionV4(nums []int, target, index int, target2judge map[int]bool) bool {
	lth := len(nums)

	if target == 0 {
		return true
	}
	if index == lth || target < 0 {
		return false
	}
	contain := false
	notContain := false
	if _, ok := target2judge[target-nums[index]]; ok {
		contain = target2judge[target-nums[index]]
	} else {
		contain = dfsCanPartitionV4(nums, target-nums[index], index+1, target2judge)
		target2judge[target-nums[index]] = contain
	}

	if _, ok := target2judge[target]; ok {
		notContain = target2judge[target]
	} else {
		notContain = dfsCanPartitionV4(nums, target, index+1, target2judge)
		target2judge[target] = notContain
	}

	return contain || notContain
}

// 最大的以1为边界的正方形
// leetcode 1139. Largest 1-Bordered Square
func largest1BorderedSquare(grid [][]int) int {
	row := len(grid)
	column := len(grid[0])
	dpRow := make([][]int, row+1)
	for i := 0; i < row; i++ {
		dpRow[i] = make([]int, column+1)
	}
	dpColumn := make([][]int, row+1)
	for i := 0; i <= row; i++ {
		dpColumn[i] = make([]int, column+1)
	}

	for i := 1; i <= row; i++ {
		for j := 1; j <= column; j++ {
			if grid[i-1][j-1] == 0 {
				continue
			}
			dpRow[i][j] = dpRow[i][j-1] + 1
			dpColumn[i][j] = dpColumn[i-1][j] + 1
		}
	}
	maxSide := 0

	for i := 1; i <= row; i++ {
		for j := 1; j <= column; j++ {
			curSide := min(dpRow[i][j], dpColumn[i][j])
			for curSide > maxSide {
				pointX := i - curSide + 1
				pointY := j - curSide + 1
				if dpRow[pointX][j] >= curSide && dpColumn[i][pointY] >= curSide {
					maxSide = curSide
					break
				}
				curSide--
			}

		}
	}
	return maxSide * maxSide
}
func min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

// 动态规划解最长公共子串
// 牛客题霸127
// NC127 最长公共子串
// https://www.nowcoder.com/practice/f33f5adc55f444baa0e0ca87ad8a6aac?tpId=117&tqId=37799&rp=1&ru=/ta/job-code-high&qru=/ta/job-code-high&difficulty=&judgeStatus=&tags=/question-ranking
func LCS(str1 string, str2 string) string {
	// write code here
	lth1 := len(str1)
	lth2 := len(str2)
	dp := make([][]string, lth1+1)
	for i := 0; i <= lth1; i++ {
		dp[i] = make([]string, lth2+1)
	}
	max := ""
	for i := 1; i <= lth1; i++ {
		for j := 1; j <= lth2; j++ {
			if str1[i-1] == str2[j-1] {
				dp[i][j] = dp[i-1][j-1] + string(str1[i-1])
			}
			if len(dp[i][j]) > len(max) {
				max = dp[i][j]
			}
		}
	}
	return max
}

// 动态规划解单词拆分
// leetcode 139. Word Break
func wordBreak(s string, wordDict []string) bool {
	lth := len(s)
	dp := make([][]bool, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]bool, lth)
	}

	word2Exist := make(map[string]bool)
	for _, word := range wordDict {
		word2Exist[word] = true
	}

	for i := 0; i < lth; i++ {
		for j := i; j < lth; j++ {
			str := s[i : j+1]
			if _, ok := word2Exist[str]; ok {
				if i == 0 {
					dp[i][j] = true
					continue
				}
				for row := i - 1; row >= 0; row-- {
					if dp[row][i-1] == true {
						dp[i][j] = true
						break
					}
				}
			}
		}
	}
	for i := 0; i < len(dp); i++ {
		if dp[i][lth-1] == true {
			return true
		}
	}
	return false

}

func wordBreakV2(s string, wordDict []string) bool {
	dp := make([]bool, len(s))
	for i := 0; i < len(s); i++ {
		for j := i; j >= 0; j-- {
			str := s[j : i+1]
			if findStrIsExist(wordDict, str) {
				if j == 0 {
					dp[i] = true
					break
				}
				if dp[j-1] == true {
					dp[i] = true
					break
				}
			}
		}
	}
	return dp[len(s)-1]
}

func findStrIsExist(wordDict []string, target string) bool {
	for _, str := range wordDict {
		if str == target {
			return true
		}
	}
	return false
}

//动态规划解分割回文串 III
//leetcode 1278. Palindrome Partitioning III

func palindromePartition(s string, k int) int {
	lth := len(s)
	if k == lth {
		return 0
	}
	dp := make([][]int, lth+1)
	for i := 0; i <= lth; i++ {
		for j := 0; j <= k; j++ {
			dp[i] = append(dp[i], lth)
		}
	}
	palindromeCnt := make([][]int, lth)
	for i := 0; i < lth; i++ {
		palindromeCnt[i] = make([]int, lth)
	}
	for i := lth - 2; i >= 0; i-- {
		for j := i + 1; j < lth; j++ {
			cnt := 0
			if s[i] != s[j] {
				cnt = 1
			}
			palindromeCnt[i][j] = palindromeCnt[i+1][j-1] + cnt
		}
	}

	for i := 1; i <= lth; i++ {
		curK := min(i, k)
		for j := 1; j <= curK; j++ {
			if j == 1 {
				dp[i][j] = palindromeCnt[i-1][j-1]
			} else {
				for m := j - 1; m < i; m++ {
					dp[i][j] = min(dp[i][j], dp[m][j-1]+palindromeCnt[m+1-1][i-1])
				}
			}
		}
	}
	return dp[lth][k]

}

//func palindromePartitionV2(s string, k int) int {
//	lth := len(s)
//	if k == lth {
//		return 0
//	}
//	mem := make(map[string]int)
//	dp := make([][]int, k)
//	for i := 0; i < k; i++ {
//		dp[i] = make([]int, lth+1)
//	}
//	for idx := range s {
//		dp[0][idx] = computeMem(s[0:idx+1], mem)
//	}
//	for i := 1; i < k; i++ {
//		for j := i; j <= lth; j++ {
//			cur := math.MaxInt32
//			for p := j - 1; p > i; p-- {
//				cur = min(cur, dp[i-1][p-1]+computeMem(s[p-1:j], mem))
//			}
//		}
//	}
//	return dp[k-1][lth]
//
//}
//
//func computeMem(str string, mem map[string]int) int {
//	if len(str) == 0 {
//		return 0
//	}
//	if val, ok := mem[str]; ok {
//		return val
//	}
//	lhs := 0
//	rhs := len(str) - 1
//	cnt := 0
//	for lhs < rhs {
//		if str[lhs] != str[rhs] {
//			cnt++
//		}
//		lhs++
//		rhs--
//	}
//	mem[str] = cnt
//	return mem[str]
//}

// 376动态规划编辑距离
// 72. Edit Distance
func minDistance(word1 string, word2 string) int {
	lth1 := len(word1)
	lth2 := len(word2)
	if lth1 == 0 && lth2 != 0 {
		return lth2
	}
	if lth1 != 0 && lth2 == 0 {
		return lth1
	}
	dp := make([][]int, lth1+1)
	for i := 0; i <= lth1; i++ {
		dp[i] = make([]int, lth2+1)
	}
	for i := 0; i <= lth1; i++ {
		for j := 0; j <= lth2; j++ {
			if i == 0 {
				dp[i][j] = j
			} else if j == 0 {
				dp[i][j] = i
			} else {
				if word1[i-1] == word2[j-1] {
					dp[i][j] = dp[i-1][j-1]
				} else {
					dp[i][j] = min(min(dp[i-1][j-1]+1, dp[i][j-1]+1), dp[i-1][j]+1)

				}
			}
		}
	}
	return dp[lth1][lth2]
}

// 395动态归划解配符匹配问题
// 44. Wildcard Matching
func isMatch(s string, p string) bool {
	return false
}

// 407动态归划和滑动窗口解决最长重复子数组
// 718. Maximum Length of Repeated Subarray
func findLength(nums1 []int, nums2 []int) int {
	lth1 := len(nums1)
	lth2 := len(nums2)
	dp := make([][]int, lth1+1)
	for i := 0; i <= lth1; i++ {
		dp[i] = make([]int, lth2+1)
	}
	maxLength := 0
	for i := 1; i <= lth1; i++ {
		for j := 1; j <= lth2; j++ {
			if nums1[i-1] == nums2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			}
			maxLength = max(maxLength, dp[i][j])
		}
	}
	return maxLength
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

//409动态划求不同路径
//leetcode 62. Unique Paths

func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 {
				dp[i][j] = 1
			} else if j == 0 {
				dp[i][j] = 1
			} else {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}

		}
	}
	return dp[m-1][n-1]
}

func uniquePathsV2(m int, n int) int {
	dp := make([]int, m)
	for i := 0; i < m; i++ {
		dp[i] = 1
	}
	for j := 1; j < n; j++ {
		for i := 1; i < m; i++ {
			dp[i] += dp[i-1]
		}
	}
	return dp[m-1]
}

// 411 动态归划和递归求不同路径 II
// 63. Unique Paths II
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

// 413动态归划求最长上升子序列
// leetcode 300. Longest Increasing Subsequence
func lengthOfLIS(nums []int) int {
	lth := len(nums)
	dp := make([][]int, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]int, lth)
	}
	maxSub := 1
	dp[0][0] = 1
	for i := 1; i < lth; i++ {
		maxC := 1
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i][j] = dp[j][j] + 1
			} else if nums[i] == nums[j] {
				dp[i][j] = dp[j][j]
			} else {
				dp[i][j] = 1
			}

			maxC = max(maxC, dp[i][j])
		}
		dp[i][i] = maxC
		maxSub = max(maxSub, maxC)
	}
	return maxSub
}

func lengthOfLISV2(nums []int) int {
	lth := len(nums)
	dp := make([]int, lth)
	dp[0] = 1
	maxSub := 1
	for i := 1; i < lth; i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
		maxSub = max(maxSub, dp[i])
	}
	return maxSub

}

func lengthOfLISV3(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	dp := make([]int, len(nums))
	dp[0] = nums[0]

	hi := 0

	for i := 1; i < len(nums); i++ {
		idx := biSearchIdx(dp, nums[i], 0, hi)
		dp[idx] = nums[i]
		if idx == hi+1 {
			hi++
		}
	}

	return hi + 1
}

func biSearchIdx(dp []int, target, lo, hi int) int {
	for lo <= hi {
		mid := (hi + lo) / 2
		if dp[mid] > target {
			hi = mid - 1
		} else if dp[mid] < target {
			lo = mid + 1
		} else {
			return mid
		}
	}
	return lo
}

// 423动态规划划和递归解最小路径和
// 64. Minimum Path Sum
func minPathSum(grid [][]int) int {
	row := len(grid)
	column := len(grid[0])

	dp := make([][]int, row)
	for i := 0; i < row; i++ {
		dp[i] = make([]int, column)
	}
	dp[0][0] = grid[0][0]
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if j == 0 && i == 0 {
				continue
			}
			if i == 0 {
				dp[i][j] = dp[i][j-1] + grid[i][j]
			} else if j == 0 {
				dp[i][j] = dp[i-1][j] + grid[i][j]
			} else {
				dp[i][j] = min(dp[i][j-1]+grid[i][j], dp[i-1][j]+grid[i][j])
			}

		}
	}
	return dp[row-1][column-1]
}

func minPathSumV2(grid [][]int) int {
	row := len(grid)
	column := len(grid[0])
	mem := make([]int, row*column)
	return dfsMinPathSumV2(0, 0, row, column, grid, mem)

}

func dfsMinPathSumV2(i, j, row, column int, grid [][]int, mem []int) int {
	if i == row-1 && j == column-1 {
		return grid[row-1][column-1]
	}
	if i >= row || j >= column {
		return math.MaxInt32
	}
	if mem[i*column+j] != 0 {
		return mem[i*column+j]
	}
	mem[i*column+j] = grid[i][j] + min(dfsMinPathSumV2(i+1, j, row, column, grid, mem), dfsMinPathSumV2(i, j+1, row, column, grid, mem))
	return mem[i*column+j]

}

// 430剑指 Offer-动态归划求正则表式匹配
// JZ19 正则表达式匹配
func match(str string, pattern string) bool {
	lth1 := len(str)
	lth2 := len(pattern)
	dp := make([][]bool, lth1+1)
	for i := 0; i <= lth1; i++ {
		dp[i] = make([]bool, lth2+1)
	}

	dp[0][0] = true
	for i := 0; i < lth2; i++ {
		if pattern[i] == '*' && dp[i][i-1] {
			dp[0][i+1] = true
		}
	}
	return false
}

//465.递归和动态规划解三角形最小路径和
//leetcode 120. Triangle

func minimumTotal(triangle [][]int) int {
	row := len(triangle)
	for i := 1; i < row; i++ {
		rhs := len(triangle[i]) - 1
		for j := 0; j <= rhs; j++ {
			if j == 0 {
				triangle[i][j] += triangle[i-1][j]
			} else if j == rhs {
				triangle[i][j] += triangle[i-1][j-1]
			} else {
				triangle[i][j] = min(triangle[i-1][j-1], triangle[i-1][j]) + triangle[i][j]
			}

		}
	}
	res := triangle[row-1][0]
	for j := 1; j < len(triangle[row-1]); j++ {
		if triangle[row-1][j] < res {
			res = triangle[row-1][j]
		}
	}
	return res
}

func minimumTotalV2(triangle [][]int) int {
	row := len(triangle)
	mem := make([]int, row*row)
	return dfsMinimumTotalV2(0, 0, triangle, row, mem)

}

func dfsMinimumTotalV2(i, j int, triangle [][]int, size int, mem []int) int {
	if i >= size || j >= size {
		return 0
	}
	key := i*size + j
	if mem[key] != 0 {
		return mem[key]
	}
	left := dfsMinimumTotalV2(i+1, j, triangle, size, mem)
	right := dfsMinimumTotalV2(i+1, j+1, triangle, size, mem)
	val := triangle[i][j] + min(left, right)
	mem[key] = val
	return val

}

// 477动态规划解按摩师的最长预约时间
// leetcode 面试题 17.16. 按摩师
func massage(nums []int) int {
	lth := len(nums)
	if lth == 0 {
		return 0
	}

	dp := make([]int, lth+1)
	dp[0] = 0
	dp[1] = nums[0]
	for i := 1; i < lth; i++ {
		dp[i+1] = max(dp[i], dp[i-1]+nums[i])
	}
	return dp[lth]

}

//486动态归划求最大子序和
//leetcode 53. Maximum Subarray

func maxSubArray(nums []int) int {
	lth := len(nums)
	dp := make([]int, lth)
	dp[0] = nums[0]
	res := dp[0]
	for i := 1; i < lth; i++ {
		dp[i] = max(nums[i], dp[i-1]+nums[i])
		res = max(res, dp[i])
	}
	return res
}

// 490动态规划和双指针买卖股票的最佳时机
// leetcode 121. Best Time to Buy and Sell Stock
func maxProfit(prices []int) int {
	maxPro := 0
	minPri := prices[0]
	for i := 1; i < len(prices); i++ {
		if prices[i] < minPri {
			minPri = prices[i]
		} else {
			maxPro = max(maxPro, prices[i]-minPri)
		}
	}
	return maxPro
}

// 492动态规划和贪心算法解买卖股票的最佳时机 II
// leetcode 122. Best Time to Buy and Sell Stock II
func maxProfitII(prices []int) int {
	sum := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			sum += prices[i] - prices[i-1]
		}
	}
	return sum
}

func maxProfitIIV2(prices []int) int {
	lth := len(prices)
	dp := make([][]int, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i := 1; i < lth; i++ {
		dp[i][0] = max(dp[i-1][1]-prices[i], dp[i-1][0])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return max(dp[lth-1][0], dp[lth-1][1])
}

func maxProfitIIV3(prices []int) int {
	lth := len(prices)
	hold := -prices[0]
	noHold := 0
	for i := 1; i < lth; i++ {
		hold = max(hold, noHold-prices[i])
		noHold = max(noHold, hold+prices[i])
	}
	return noHold
}

//493动态归划解打家劫舍 III
//337. House Robber III

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func rob(root *TreeNode) int {
	res := robHelper(root)
	return max(res[0], res[1])
}

func robHelper(node *TreeNode) [2]int {
	if node == nil {
		return [2]int{}
	}
	left := robHelper(node.Left)
	right := robHelper(node.Right)
	return [2]int{max(left[0], left[1]) + max(right[0], right[1]), node.Val + left[0] + right[0]}
}

// 515动态规划解买卖股票的最佳时机含手续费
// 714. Best Time to Buy and Sell Stock with Transaction Fee
func maxProfitIV(prices []int, fee int) int {
	hold := -prices[0]
	noHold := 0
	for i := 1; i < len(prices); i++ {
		hold = max(noHold-prices[i], hold)
		noHold = max(hold+prices[i]-fee, noHold)
	}
	return noHold
}

//517最长回文子串的3种解决方式
//leetcode Longest Palindromic Substring

func longestPalindrome(s string) string {
	idx := 0
	lth := len(s) - 1
	maxSub := s[:1]
	for idx <= lth {
		lhs := idx
		rhs := lth
		for lhs < rhs && (rhs-lhs+1) > len(maxSub) {
			if isPalindrome(lhs, rhs, s) {
				if len(s[lhs:rhs+1]) > len(maxSub) {
					maxSub = s[lhs : rhs+1]

				}
				break
			} else {
				rhs--
			}
		}
		idx++

	}
	return maxSub

}

func isPalindrome(lhs, rhs int, s string) bool {
	for lhs < rhs {
		if s[lhs] != s[rhs] {
			return false
		}
		lhs++
		rhs--

	}
	return true
}

// 中心扩散法
func longestPalindromeV2(s string) string {
	maxSub := s[:1]
	lth := len(s)
	for i := 1; i < lth; i++ {
		left := i - 1
		right := i + 1
		for left >= 0 && s[left] == s[i] {
			left--
		}
		for right < lth && s[right] == s[i] {
			right++
		}
		for left >= 0 && right < lth {
			if s[left] != s[right] {
				break
			}
			left--
			right++
		}
		if len(s[left+1:right]) > len(maxSub) {
			maxSub = s[left+1 : right]
		}
	}
	return maxSub

}

// 522俄罗斯套娃信封问题
// leetcode 354. Russian Doll Envelopes
func maxEnvelopes(envelopes [][]int) int {
	sort.Slice(envelopes, func(i, j int) bool { return envelopes[i][0] < envelopes[j][0] })
	lth := len(envelopes)
	dp := make([]int, lth)
	dp[0] = 1
	maxEnve := dp[0]
	for i := 1; i < lth; i++ {
		dp[i] = 1
		iMax := dp[i]
		for j := i - 1; j >= 0; j-- {
			if iMax > j+1 {
				break
			}
			if envelopes[i][1] > envelopes[j][1] && envelopes[i][0] > envelopes[j][0] {
				iMax = max(iMax, dp[i]+dp[j])
			}
		}
		dp[i] = iMax
		maxEnve = max(maxEnve, dp[i])
	}
	return maxEnve
}

func main() {

	//[[5,4],[6,4],[6,7],[2,3]]
	fmt.Println(maxEnvelopesV3([][]int{{30, 50}, {12, 2}, {3, 4}, {12, 15}}))
}

func maxEnvelopesV2(envelopes [][]int) int {
	sort.Slice(envelopes, func(i, j int) bool {
		if envelopes[i][0] == envelopes[j][0] {
			return envelopes[i][1] > envelopes[j][1]
		} else {
			return envelopes[i][0] < envelopes[j][0]
		}
	})

	n := len(envelopes)
	if n <= 1 {
		return n
	}

	size := 0
	c := make([]int, 0)

	for i := 0; i < n; i++ {
		l, r := 0, size-1
		for l <= r {
			mid := (l + r) / 2
			if c[mid] >= envelopes[i][1] {
				r = mid - 1
			} else {
				l = mid + 1
			}
		}

		if l < size {
			c[l] = envelopes[i][1]
		} else {
			c = append(c, envelopes[i][1])
			size = max(size, l+1)
		}
	}

	return size
}

func maxEnvelopesV3(envelopes [][]int) int {
	sort.Slice(envelopes, func(i, j int) bool {
		if envelopes[i][0] == envelopes[j][0] {
			return envelopes[i][1] > envelopes[j][1]
		} else {
			return envelopes[i][0] < envelopes[j][0]
		}
	})

	lth := len(envelopes)
	dp := make([]int, lth)

	dp[0] = envelopes[0][1]
	hi := 0
	for i := 1; i < lth; i++ {
		idx := biSearchIdx(dp, envelopes[i][1], 0, hi)
		dp[idx] = envelopes[i][1]
		if idx == hi+1 {
			hi++
		}
	}
	return hi + 1
}

// 529，动态规划解最长回文子序列
// leetcode 516. Longest Palindromic Subsequence
func longestPalindromeSubseq(s string) int {
	lth := len(s)
	dp := make([][]int, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]int, lth)
	}

	for i := lth - 1; i >= 0; i-- {
		dp[i][i] = 1
		for j := i + 1; j < lth; j++ {
			if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[0][lth-1]
}

//530，动态规划解最大正方形
//leetcode 221. Maximal Square

func maximalSquare(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}
	m, n := len(matrix), len(matrix[0])
	dp := make([][]int, m+1)
	ans := 0
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if matrix[i-1][j-1] == '1' {
				dp[i][j] = 1 + min(min(dp[i-1][j-1], dp[i-1][j]), dp[i][j-1])
			}
			ans = max(ans, dp[i][j])
		}
	}
	return ans * ans
}

// 540，动态规划和中心扩散法解回文子串
// 647. Palindromic Substrings
func countSubstrings(s string) int {
	total := 0
	for mid := 0; mid < len(s); mid++ {
		left := mid
		right := mid + 1
		for left >= 0 && right < len(s) && s[left] == s[right] {
			left--
			right++
			total++
		}

		left = mid
		right = mid
		for left >= 0 && right < len(s) && s[left] == s[right] {
			left--
			right++
			total++
		}
	}
	return total
}
