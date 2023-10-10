package dp

import (
	"math"
	"strconv"
)

// 119. Pascal's Triangle II
func getRow(rowIndex int) []int {
	ret := make([]int, rowIndex+1)
	ret[0] = 1
	for i := 0; i <= rowIndex; i++ {
		tmp := make([]int, len(ret))
		copy(tmp, ret)
		lth := i + 1
		for j := 0; j < lth; j++ {
			if j == 0 || j == lth-1 {
				ret[j] = 1
			} else {
				ret[j] = tmp[j] + tmp[j-1]
			}
		}
	}
	return ret
}

// 从后往前加 efficient
func getRowV2(rowIndex int) []int {
	res := make([]int, rowIndex+1)
	res[0] = 1
	for i := 0; i <= rowIndex; i++ {
		res[i] = 1
		for j := i - 1; j > 0; j-- {
			res[j] = res[j] + res[j-1]
		}
	}

	return res[0 : rowIndex+1]
}

// 1137. N-th Tribonacci Number
func tribonacci(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 || n == 2 {
		return 1
	}
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	dp[2] = 1
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-3] + dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func tribonacciV2(n int) int {
	memo := [3]int{0, 1, 1}
	if n < 3 {
		return memo[n]
	}
	for i := 3; i <= n; i++ {
		memo[0], memo[1], memo[2] = memo[1], memo[2], memo[0]+memo[1]+memo[2]
	}
	return memo[2]
}

func getMaximumGenerated(n int) int {
	if n <= 1 {
		return n
	}
	nums := make([]int, n+1)
	nums[1] = 1
	maxNum := 1
	for i := 2; i <= n; i++ {
		k := i / 2
		if i&1 == 0 {
			nums[i] = nums[k]
		} else {
			nums[i] = nums[k] + nums[k+1]
		}
		if nums[i] > maxNum {
			maxNum = nums[i]
		}
	}
	return maxNum
}

// 45. Jump Game II
func jump(nums []int) int {
	//nums = [2,3,1,1,4]
	dp := make([]int, len(nums))
	for i := 1; i < len(nums); i++ {
		dp[i] = math.MaxInt
		for j := 0; j < i; j++ {
			if nums[j] >= i-j {
				step := dp[j] + 1
				if step < dp[i] {
					dp[i] = step
				}
			}
		}
	}
	return dp[len(nums)-1]
}

func jumpV2(nums []int) int {
	curJump, farthestJump, jumps := 0, 0, 0
	for i := 0; i < len(nums)-1; i++ {
		// push index of furthest jump during current iteration
		if i+nums[i] > farthestJump {
			farthestJump = i + nums[i]
		}

		// if current iteration is ended - setup the next one
		if i == curJump {
			jumps, curJump = jumps+1, farthestJump

			if curJump >= len(nums)-1 {
				return jumps
			}
		}
	}

	// it's guaranteed to never hit it
	return 0
}

// TODO 没看懂
func jumpV3(nums []int) int {
	if len(nums) < 2 {
		return 0
	}

	dp := make([]int, len(nums))

	for i := len(nums) - 1; i >= 0; i-- {
		if nums[i] == 0 {
			continue
		}

		if i+nums[i] >= len(nums)-1 {
			dp[i] = 1
			continue
		}

		for step := 1; step <= nums[i]; step++ {
			if dp[step+i] == 0 {
				continue
			}

			if dp[i] == 0 {
				dp[i] = dp[step+i] + 1
			} else {
				dp[i] = min(dp[i], dp[step+i]+1)
			}
		}
	}

	return dp[0]
}

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}

// 96. Unique Binary Search Trees
func numTrees(n int) int {
	f := make([]int, n+2)
	f[0] = 1
	for i := 1; i <= n; i++ {
		x := i - 1
		for j := 0; j <= x; j++ {
			f[i] += f[x-j] * f[j]
		}
	}
	return f[n]
}

// 97. Interleaving String
func isInterleave(s1 string, s2 string, s3 string) bool {
	lth1 := len(s1)
	lth2 := len(s2)
	lth3 := len(s3)
	if (lth1 + lth2) != lth3 {
		return false
	}
	dp := make([][]bool, lth1+1)
	for i := 0; i <= lth1; i++ {
		dp[i] = make([]bool, lth2+1)
	}
	dp[0][0] = true

	for i := 0; i < lth1; i++ {
		if s1[i] == s3[i] && dp[i][0] {
			dp[i+1][0] = true
		}
	}

	for i := 0; i < lth2; i++ {
		if s2[i] == s3[i] && dp[0][i] {
			dp[0][i+1] = true
		}
	}

	for i := 0; i < lth1; i++ {
		for j := 0; j < lth2; j++ {
			if s1[i] == s3[i+j+1] && dp[i][j+1] {
				dp[i+1][j+1] = true
			}
			if s2[j] == s3[i+j+1] && dp[i+1][j] {
				dp[i+1][j+1] = true
			}

		}
	}
	return dp[lth1][lth2]
}

// 213. House Robber II
// [1,2,3,1] 1+3=4
func rob(nums []int) int {
	lth := len(nums)
	if lth == 1 {
		return nums[0]
	}
	if lth == 2 {
		return max(nums[0], nums[1])
	}
	var dp [2][]int
	dp[0] = make([]int, lth)
	dp[1] = make([]int, lth)
	dp[1][0] = nums[0]
	dp[0][1] = nums[1]
	dp[1][1] = max(dp[1][0], nums[1])
	for i := 2; i < lth; i++ {
		if i == lth-1 {
			dp[0][i] = max(dp[0][i-1], dp[0][i-2]+nums[i])
			break
		}
		dp[1][i] = max(dp[1][i-1], dp[1][i-2]+nums[i])
		dp[0][i] = max(dp[0][i-1], dp[0][i-2]+nums[i])
	}

	return max(dp[0][lth-1], dp[1][lth-2])

}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 241. Different Ways to Add Parentheses
/*
Example 1:

Input: expression = "2-1-1"
Output: [0,2]
Explanation:
((2-1)-1) = 0
(2-(1-1)) = 2
Example 2:

Input: expression = "2*3-4*5"
Output: [-34,-14,-10,-10,10]
Explanation:
(2*(3-(4*5))) = -34
((2*3)-(4*5)) = -14
((2*(3-4))*5) = -10
(2*((3-4)*5)) = -10
(((2*3)-4)*5) = 10


Constraints:

1 <= expression.length <= 20
expression consists of digits and the operator '+', '-', and '*'.
All the integer values in the input expression are in the range [0, 99].
*/

// excellent
func diffWaysToCompute(expression string) []int {
	result := []int{}

	// note: base case when expression is a number only
	if v, err := strconv.Atoi(expression); err == nil {
		return []int{v}
	}

	for i := 0; i < len(expression); i++ {
		if expression[i] == '+' || expression[i] == '-' || expression[i] == '*' {

			leftResult := diffWaysToCompute(expression[:i])
			rightResult := diffWaysToCompute(expression[i+1:])

			for _, left := range leftResult {
				for _, right := range rightResult {
					value := 0

					if expression[i] == '+' {
						value = left + right
					}

					if expression[i] == '-' {
						value = left - right
					}

					if expression[i] == '*' {
						value = left * right
					}

					result = append(result, value)
				}
			}
		}
	}

	return result
}
