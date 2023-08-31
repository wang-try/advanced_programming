package dynamic_programming

/*
通常按如下4个步骤来设计一个动态规划算法
1.刻画一个最优解的结构特征
2.递归的定义最优解的值
3.计算最优解的值，通常采用自底向上的方法
4.利用计算出的信息构造一个最优解。
*/

//爬楼梯的最少成本
/*
一个数组cost的所有数字都是正数，它的第i个数字表示在一个楼梯的第i级台阶往上爬的成本，在支付了成本cost[i]之后可以从第i级台阶往上爬1级或2级。
假设台阶至少有2级，既可以从第0级台阶出发，也可以从第1级台阶出发，请计算爬上该楼梯的最少成本。
例如，输入数组[1，100，1，1，100，1]，则爬上该楼梯的最少成本是4，分别经过下标为0、2、3、5的这4级台阶
*/
func minCostClimbingStairs(cost []int) int {
	dp := make([]int, len(cost)+1)
	dp[0] = 0
	dp[1] = cost[0]
	for i := 2; i <= len(cost); i++ {
		dp[i] = min(dp[i-2]+cost[i-1], dp[i-1]+cost[i-1])
	}
	return min(dp[len(cost)], dp[len(cost)-1])
}

func minCostClimbingStairsV2(cost []int) int {
	dp := [2]int{cost[0], cost[1]}
	for i := 2; i < len(cost); i++ {
		dp[i%2] = min(dp[0], dp[1]) + cost[i]
	}
	return min(dp[0], dp[1])
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

//单序列问题
/*
单序列问题是与动态规划相关的问题中最有可能在算法面试中遇到的题型。这类题目都有适合运用动态规划的问题的特点，如解决问题需要若干步骤，
并且每个步骤都面临若干选择，需要计算解的数目或最优解。除此之外，这类题目的输入通常是一个序列，如一个一维数组或字符串。应用动态规划解决单序列问题的关键是每一步在序列中增加一个元素，
根据题目的特点找出该元素对应的最优解（或解的数目）和前面若干元素（通常是一个或两个）的最优解（或解的数目）的关系，
并以此找出相应的状态转移方程。一旦找出了状态转移方程，只要注意避免不必要的重复计算，问题就能迎刃而解
*/

//房屋偷盗
/*
输入一个数组表示某条街道上的一排房屋内财产的数量。如果这条街道上相邻的两幢房屋被盗就会自动触发报警系统。请计算小偷在这条街道上最多能偷取到多少财产。
例如，街道上5幢房屋内的财产用数组[2，3，4，5，3]表示，如果小偷到下标为0、2和4的房屋内盗窃，那么他能偷取到价值为9的财物，这是他在不触发报警系统的情况下能偷取到的最多的财物，
*/

func rob(nums []int) int {
	dp := make([]int, len(nums)+1)
	dp[0] = 0
	dp[1] = nums[0]
	for i := 2; i <= len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i-1])
	}
	return dp[len(nums)]
}

func robV2(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	var dp [2]int
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp[i%2] = max(dp[(i-1)%2], dp[(i-2)%2]+nums[i])
	}
	return dp[(len(nums)-1)%2]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

//环形房屋偷盗
/*
一条环形街道上有若干房屋。输入一个数组表示该条街道上的房屋内财产的数量。如果这条街道上相邻的两幢房屋被盗就会自动触发报警系统。请计算小偷在这条街道上最多能偷取的财产的数量。
例如，街道上5家的财产用数组[2，3，4，5，3]表示，如果小偷到下标为1和3的房屋内盗窃，那么他能偷取到价值为8的财物，这是他在不触发报警系统的情况下能偷取到的最多的财物，如图14.4所示。被盗的房屋上方用特殊符号标出
*/

func robCircle(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}

	dp := make([][]int, 2)
	for i := 0; i < 2; i++ {
		dp[i] = make([]int, len(nums)+1)
	}

	dp[0][0] = 0
	dp[1][0] = 0
	dp[0][1] = nums[0]
	dp[1][1] = 0
	for i := 2; i <= len(nums); i++ {
		if i == len(nums) {
			dp[1][i] = max(dp[1][i-1], dp[1][i-2]+nums[i-1])
		} else {
			dp[0][i] = max(dp[0][i-1], dp[0][i-2]+nums[i-1])
			dp[1][i] = max(dp[1][i-1], dp[1][i-2]+nums[i-1])
		}
	}
	return max(dp[0][len(nums)-1], dp[1][len(nums)])
}

func robCircleV2(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	var dp [2][2]int
	dp[0][0] = nums[0]
	dp[1][0] = 0
	dp[0][1] = max(nums[0], nums[1])
	dp[1][1] = nums[1]
	for i := 2; i < len(nums); i++ {
		if i == len(nums)-1 {
			dp[1][i%2] = max(dp[1][(i-1)%2], dp[1][(i-2)%2]+nums[i])
		} else {
			dp[0][i%2] = max(dp[0][(i-1)%2], dp[0][(i-2)%2]+nums[i])
			dp[1][i%2] = max(dp[1][(i-1)%2], dp[1][(i-2)%2]+nums[i])
		}
	}
	return max(dp[0][(len(nums)-2)%2], dp[1][(len(nums)-1)%2])
}

func robCircleV3(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	result1 := helper(nums, 0, len(nums)-2)
	result2 := helper(nums, 1, len(nums)-1)
	return max(result2, result1)
}

func helper(nums []int, start, end int) int {
	var dp [2]int
	dp[0] = nums[start]
	if start < end {
		dp[1] = max(nums[start], nums[start+1])
	}
	for i := start + 2; i <= end; i++ {
		j := i - start
		dp[j%2] = max(dp[(j-1)%2], dp[(j-2)%2]+nums[i])
	}
	return dp[(end-start)%2]
}

//粉刷房子
/*
一排n幢房子要粉刷成红色、绿色和蓝色，不同房子被粉刷成不同颜色的成本不同。用一个n×3的数组表示n幢房子分别用3种颜色粉刷的成本。要求任意相邻的两幢房子的颜色都不一样，请计算粉刷这n幢房子的最少成本。
例如，粉刷3幢房子的成本分别为[[17，2，16]，[15，14，5]，[13，3，1]]，如果分别将这3幢房子粉刷成绿色、蓝色和绿色，那么粉刷的成本是10，是最少的成本
*/

func minCost(costs [][]int) int {
	if len(costs) == 0 {
		return 0
	}
	r := make([]int, len(costs))
	g := make([]int, len(costs))
	b := make([]int, len(costs))
	r[0] = costs[0][0]
	g[0] = costs[0][1]
	b[0] = costs[0][2]
	for i := 1; i < len(costs); i++ {
		r[i] = min(g[i-1], b[i-1]) + costs[i][0]
		g[i] = min(r[i-1], b[i-1]) + costs[i][1]
		b[i] = min(g[i-1], r[i-1]) + costs[i][2]
	}
	return min(r[len(costs)-1], min(g[len(costs)-1], b[len(costs)-1]))
}

func minCostV2(costs [][]int) int {
	if len(costs) == 0 {
		return 0
	}
	var dp [3][2]int
	for j := 0; j < 3; j++ {
		dp[j][0] = costs[0][j]
	}
	for i := 1; i < len(costs); i++ {
		for j := 0; j < 3; j++ {
			prev1 := dp[(j+2)%3][(i-1)%2]
			prev2 := dp[(j+1)%3][(i-1)%2]
			dp[j][i%2] = min(prev1, prev2) + costs[i][j]
		}
	}
	last := (len(costs) - 1) % 2
	return min(dp[0][last], min(dp[1][last], dp[2][last]))
}

//翻转字符
/*
输入一个只包含'0'和'1'的字符串，其中，'0'可以翻转成'1'，'1'可以翻转成'0'。请问至少需要翻转几个字符，才可以使翻转之后的字符串中所有的'0'位于'1'的前面？翻转之后的字符串可能只包含字符'0'或'1'。
例如，输入字符串"00110"，至少需要翻转一个字符才能使所有的'0'位于'1'的前面。可以将最后一个字符'0'翻转成'1'，得到字符串"00111
*/

func minFlipsMonoIncr(s string) int {
	if len(s) == 0 {
		return 0
	}
	if len(s) == 1 {
		return 1
	}
	//0
	f := make([]int, len(s))
	//1
	g := make([]int, len(s))
	if s[0] == '0' {
		g[0] = 1
	} else {
		f[0] = 1
	}
	for i := 1; i < len(s); i++ {
		if s[i] == '0' {
			f[i] = f[i-1]
			g[i] = min(f[i-1], g[i-1]) + 1
		} else {
			f[i] = f[i-1] + 1
			g[i] = min(f[i-1], g[i-1])
		}
	}
	return min(g[len(s)-1], f[len(s)-1])

}

func minFlipsMonoIncrV2(s string) int {
	if len(s) == 0 {
		return 0
	}
	if len(s) == 1 {
		return 1
	}
	var dp [2][]int
	dp[0] = make([]int, len(s))
	dp[1] = make([]int, len(s))

	if s[0] == '0' {
		dp[1][0] = 1
	} else {
		dp[0][0] = 1
	}
	for i := 1; i < len(s); i++ {
		if s[i] == '0' {
			dp[0][i] = dp[0][i-1]
			dp[1][i] = min(dp[0][i-1], dp[1][i-1]) + 1
		} else {
			dp[0][i] = dp[0][i-1] + 1
			dp[1][i] = min(dp[0][i-1], dp[1][i-1])
		}
	}
	return min(dp[0][len(s)-1], dp[1][len(s)-1])

}

func minFlipsMonoIncrV3(s string) int {
	if len(s) == 0 {
		return 0
	}
	if len(s) == 1 {
		return 1
	}
	var dp [2][1]int
	if s[0] == '0' {
		dp[1][0] = 1
	} else {
		dp[0][0] = 1
	}
	for i := 1; i < len(s); i++ {
		if s[i] == '0' {
			dp[1][0] = min(dp[0][0], dp[1][0]) + 1
		} else {
			tmp := dp[0][0]
			dp[0][0] = dp[0][0] + 1
			dp[1][0] = min(tmp, dp[1][0])
		}
	}
	return min(dp[0][0], dp[1][0])

}

//最长斐波那契数列
/*
输入一个没有重复数字的单调递增的数组，数组中至少有3个数字，请问数组中最长的斐波那契数列的长度是多少？
例如，如果输入的数组是[1，2，3，4，5，6，7，8]，由于其中最长的斐波那契数列是1、2、3、5、8，因此输出是5。
*/
func lenLongestFibSubSeq(arr []int) int {
	if len(arr) <= 2 {
		return 0
	}
	flag := make(map[int]int)
	for i, num := range arr {
		flag[num] = i
	}
	dp := make([][]int, len(arr))
	for i := 0; i < len(arr); i++ {
		dp[i] = make([]int, len(arr))
	}
	maxLen := 2
	for i := 1; i < len(arr); i++ {
		for j := 0; j < i; j++ {
			if k, ok := flag[arr[i]-arr[j]]; ok && k < j {
				dp[i][j] = dp[j][k] + 1
			} else {
				dp[i][j] = 2
			}
			maxLen = max(maxLen, dp[i][j])
		}
	}
	if maxLen > 2 {
		return maxLen
	}
	return 0
}

//最少回文分割
/*
输入一个字符串，请问至少需要分割几次才可以使分割出的每个子字符串都是回文？例如，输入字符串"aaba"，至少需要分割1次，从两个相邻字符'a'中间切一刀将字符串分割成两个回文子字符串"a"和"aba"。
*/

func minCut(s string) int {
	isPal := make([][]bool, len(s))
	for i := 0; i < len(s); i++ {
		isPal[i] = make([]bool, len(s))

	}
	for i := len(s) - 1; i >= 0; i-- {
		for j := i; j < len(s); j++ {
			if j == i {
				isPal[i][j] = true
			} else if j == i+1 && s[i] == s[j] {
				isPal[i][j] = true
			} else {
				if isPal[i+1][j-1] && s[i] == s[j] {
					isPal[i][j] = true
				}
			}
		}
	}

	dp := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		if isPal[0][i] {
			dp[i] = 0
		} else {
			dp[i] = i
			for j := 1; j <= i; j++ {
				if isPal[j][i] {
					dp[i] = min(dp[i], dp[j-1]+1)
				}
			}
		}

	}
	return dp[len(s)-1]
}

//双序列问题
/*
和单序列问题不同，双序列问题的输入有两个或更多的序列，通常是两个字符串或数组。由于输入是两个序列，因此状态转移方程通常有两个参数，即f（i，j），
定义第1个序列中下标从0到i的子序列和第2个序列中下标从0到j的子序列的最优解（或解的个数）。一旦找到了f（i，j）与f（i-1，j-1）、f（i-1，j）和f（i，j-1）的关系，通常问题也就迎刃而解。
由于双序列的状态转移方程有两个参数，因此通常需要使用一个二维数组来保存状态转移方程的计算结果。但在大多数情况下，可以优化代码的空间效率，只需要保存二维数组中的一行就可以完成状态转移方程的计算，
因此可以只用一个一维数组就能实现二维数组的缓存功能
*/
//最长公共子序列
/*
输入两个字符串，请求出它们的最长公共子序列的长度。如果从字符串s1中删除若干字符之后能得到字符串s2，那么字符串s2就是字符串s1的一个子序列。
例如，从字符串"abcde"中删除两个字符之后能得到字符串"ace"，因此字符串"ace"是字符串"abcde"的一个子序列。
但字符串"aec"不是字符串"abcde"的子序列。如果输入字符串"abcde"和"badfe"，那么它们的最长公共子序列是"bde"，因此输出3
*/

func longestCommonSubsequence(text1 string, text2 string) int {
	if len(text1) == 0 || len(text2) == 0 {
		return 0
	}
	dp := make([][]int, len(text1)+1)
	for i := 0; i <= len(text1); i++ {
		dp[i] = make([]int, len(text2)+1)
	}
	for i := 0; i < len(text1); i++ {
		for j := 0; j < len(text2); j++ {
			if text1[i] == text2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
			}
		}
	}
	return dp[len(text1)][len(text2)]

}

func longestCommonSubsequenceV2(text1 string, text2 string) int {
	len1 := len(text1)
	len2 := len(text2)
	if len1 < len2 {
		return longestCommonSubsequenceV2(text2, text1)
	}
	var dp [2][]int
	for i := 0; i < 2; i++ {
		dp[i] = make([]int, len2+1)
	}
	for i := 0; i < len1; i++ {
		for j := 0; j < len2; j++ {
			if text1[i] == text2[j] {
				dp[(i+1)%2][j+1] = dp[i%2][j] + 1
			} else {
				dp[(i+1)%2][j+1] = max(dp[i%2][j+1], dp[(i+1)%2][j])
			}
		}
	}
	return dp[len1%2][len2]
}

func longestCommonSubsequenceV3(text1 string, text2 string) int {
	len1 := len(text1)
	len2 := len(text2)
	if len1 < len2 {
		return longestCommonSubsequenceV3(text2, text1)
	}
	dp := make([]int, len2+1)
	for i := 0; i < len1; i++ {
		prev := dp[0]
		for j := 0; j < len2; j++ {
			var cur int
			if text1[i] == text2[j] {
				cur = prev + 1
			} else {
				cur = max(dp[j], dp[j+1])
			}
			prev = dp[j+1]
			dp[j+1] = cur
		}
	}
	return dp[len2]
}

//字符串交织
/*
输入3个字符串s1、s2和s3，请判断字符串s3能不能由字符串s1和s2交织而成，即字符串s3的所有字符都是字符串s1或s2中的字符，字符串s1和s2中的字符都将出现在字符串s3中且相对位置不变。
例如，字符串"aadbbcbcac"可以由字符串"aabcc"和"dbbca"交织而成
*/

func isInterleave(s1 string, s2 string, s3 string) bool {
	lth1 := len(s1)
	lth2 := len(s2)
	lth3 := len(s3)
	if lth3 != (lth1 + lth2) {
		return false
	}
	dp := make([][]bool, lth1+1)
	for i := 0; i < lth1+1; i++ {
		dp[i] = make([]bool, lth2+1)
	}
	dp[0][0] = true
	for i := 0; i < lth1; i++ {
		if s1[i] == s3[i] && dp[i][0] {
			dp[i+1][0] = true
		}
	}
	for j := 0; j < lth2; j++ {
		if s2[j] == s3[j] && dp[0][j] {
			dp[0][j+1] = true
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

//子序列的数目
/*
输入字符串S和T，请计算字符串S中有多少个子序列等于字符串T。例如，在字符串"appplep"中，有3个子序列等于字符串"apple"
*/
func numDistinct(s string, t string) int {
	lth1 := len(s)
	lth2 := len(t)
	dp := make([][]int, lth1+1)
	for i := 0; i <= lth1; i++ {
		dp[i] = make([]int, lth2+1)
	}
	dp[0][0] = 1
	for i := 0; i < lth1; i++ {
		dp[i+1][0] = 1
		for j := 0; j <= i && j < lth2; j++ {
			if s[i] == t[j] {
				dp[i+1][j+1] = dp[i][j] + dp[i][j+1]
			} else {
				dp[i+1][j+1] = dp[i][j+1]
			}
		}
	}
	return dp[lth1][lth2]
}

//矩阵路径问题
/*
矩阵路径是一类常见的可以用动态规划来解决的问题。这类问题通常输入的是一个二维的格子，一个机器人按照一定的规则从格子的某个位置走到另一个位置，要求计算路径的条数或找出最优路径。
矩阵路径相关问题的状态方程通常有两个参数，即f（i，j）的两个参数i、j通常是机器人当前到达的坐标。需要根据路径的特点找出到达坐标（i，j）之前的位置，通常是坐标（i-1，j-1）、（i-1，j）、（i，j-1）中的一个或多个。
相应地，状态转移方程就是找出f（i，j）与f（i-1，j-1）、f（i-1，j）或f（i，j-1）的关系。可以根据状态转移方程写出递归代码，但值得注意的是一定要将f（i，j）的计算结果用一个二维数组缓存，以避免不必要的重复计算。
也可以将计算所有f（i，j）看成填充二维表格的过程，相应地，可以创建一个二维数组并逐一计算每个元素的值。通常，矩阵路径相关问题的代码都可以优化空间效率，用一个一维数组就能保存所有必需的数据。
*/

//路径的数目
/*
一个机器人从m×n的格子的左上角出发，它每步要么向下要么向右，直到抵达格子的右下角。请计算机器人从左上角到达右下角的路径的数目。例如，如果格子的大小是3×3，那么机器人从左上角到达右下角有6条符合条件的不同路径
*/
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

func uniquePathsV2(m, n int) int {
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[j] += dp[j-1]
		}
	}
	return dp[n-1]
}

//最小路径之和
/*
在一个m×n（m、n均大于0）的格子中，每个位置都有一个数字。一个机器人每步只能向下或向右，请计算它从格子的左上角到达右下角的路径的数字之和的最小值。
例如，从图14.8中3×3的格子的左上角到达右下角的路径的数字之和的最小值是8，图中数字之和最小的路径用灰色背景表示
*/

func minPathSum(grid [][]int) int {
	row := len(grid)
	column := len(grid[0])
	dp := make([][]int, row)
	for i := 0; i < row; i++ {
		dp[i] = make([]int, column)
	}
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
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
	return dp[row-1][column-1]
}

//三角形中最小路径之和
/*
在一个由数字组成的三角形中，第1行有1个数字，第2行有2个数字，以此类推，第n行有n个数字。
例如，图14.9是一个包含4行数字的三角形。如果每步只能前往下一行中相邻的数字，请计算从三角形顶部到底部的路径经过的数字之和的最小值。
如图14.9所示，从三角形顶部到底部的路径数字之和的最小值为11，对应的路径经过的数字用阴影表示
*/

func minimumTotal(triangle [][]int) int {
	row := len(triangle)
	dp := make([][]int, row)
	for i := 0; i < row; i++ {
		dp[i] = make([]int, row)
	}
	for i := 0; i < row; i++ {
		for j := 0; j <= i; j++ {
			dp[i][j] = triangle[i][j]
			if i > 0 {
				if j == 0 {
					dp[i][j] += dp[i-1][j]
				} else if j == i {
					dp[i][j] += dp[i-1][j-1]
				} else {
					dp[i][j] += min(dp[i-1][j], dp[i-1][j-1])
				}
			}
		}
	}
	minT := dp[row-1][0]
	for i := 1; i < row; i++ {
		if dp[row-1][i] < minT {
			minT = dp[row-1][i]
		}
	}
	return minT
}

//背包问题
/*
背包问题是一类经典的可以应用动态规划来解决的问题。背包问题的基本描述如下：给定一组物品，每种物品都有其重量和价格，在限定的总重量内如何选择才能使物品的总价格最高。
由于问题是关于如何选择最合适的物品放置于给定的背包中，因此这类问题通常被称为背包问题。根据物品的特点，背包问题还可以进一步细分。如果每种物品只有一个，可以选择将之放入或不放入背包，
那么可以将这类问题称为0-1背包问题。0-1背包问题是最基本的背包问题，其他背包问题通常可以转化为0-1背包问题。如果第i种物品最多有Mi个，也就是每种物品的数量都是有限的，
那么这类背包问题称为有界背包问题（也可以称为多重背包问题）。如果每种物品的数量都是无限的，那么这类背包问题称为无界背包问题（也可以称为完全背包问题）
*/

//分割等和子集

/*
给定一个非空的正整数数组，请判断能否将这些数字分成和相等的两部分。例如，如果输入数组为[3，4，1]，
将这些数字分成[3，1]和[4]两部分，它们的和相等，因此输出true；如果输入数组为[1，2，3，5]，则不能将这些数字分成和相等的两部分，因此输出false
*/

func canPartition(nums []int) bool {
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
	dp := make([][]bool, lth+1)
	for i := 0; i <= lth; i++ {
		dp[i] = make([]bool, half+1)
		dp[i][0] = true
	}
	for i := 1; i <= lth; i++ {
		for j := 1; j <= half; j++ {
			dp[i][j] = dp[i-1][j]
			if !dp[i][j] && j >= nums[i-1] {
				dp[i][j] = dp[i-1][j-nums[i-1]]
			}
		}
	}
	return dp[lth][half]
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

//加减的目标值
/*
给定一个非空的正整数数组和一个目标值S，如果为每个数字添加“+”或“-”运算符，请计算有多少种方法可以使这些整数的计算结果为S。
例如，如果输入数组[2，2，2]并且S等于2，有3种添加“+”或“-”的方法使结果为2，它们分别是2+2-2=2、2-2+2=2及-2+2+2=2。
*/
func findTargetSumWays(nums []int, target int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if (sum+target)%2 == 1 || sum < target {
		return 0
	}
	return subSetSum(nums, (sum+target)/2)

}

func subSetSum(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for _, num := range nums {
		for j := target; j >= num; j-- {
			dp[j] += dp[j-num]
		}
	}
	return dp[target]
}

//最少的硬币数目
/*
给定正整数数组coins表示硬币的面额和一个目标总额t，请计算凑出总额t至少需要的硬币数目。每种硬币可以使用任意多枚。如果不能用输入的硬币凑出给定的总额，则返回-1。
例如，如果硬币的面额为[1，3，9，10]，总额t为15，那么至少需要3枚硬币，即2枚面额为3的硬币及1枚面额为9的硬币
*/

func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 0; i <= amount; i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for _, coin := range coins {
		for j := amount; j >= 1; j-- {
			for k := 1; k*coin <= j; k++ {
				dp[j] = min(dp[j], dp[j-k*coin]+k)
			}
		}
	}
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

func coinChangeV2(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 0; i <= amount; i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
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

//排列的数目
/*
给定一个非空的正整数数组nums和一个目标值t，数组中的所有数字都是唯一的，请计算数字之和等于t的所有排列的数目。数组中的数字可以在排列中出现任意次。
例如，输入数组[1，2，3]，目标值t为3，那么总共有4个组合的数字之和等于3，它们分别为{1，1，1}、{1，2}、{2，1}及{3}。
*/

func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i <= target; i++ {
		for _, num := range nums {
			if i >= num {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}
