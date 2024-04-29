package algorithm_template

/************-背包模版-***************/
/*
0-1背包问题  https://blog.csdn.net/xxc97/article/details/123798317
二维遍历顺序不重要
一维
	先遍历物品，后遍历背包（顺序要是反的话，会多放物品），且背包容量从大到小遍历
完全背包问题
	对于纯完全背包问题，先遍历物品再遍历背包和先遍历背包，后遍历物品都可以。
	01背包内嵌的循环 j（一维数组） 是从大到小遍历，为了保证每个物品仅被添加一次。而完全背包的物品是可以添加多次的，所以要从小到大去遍历。

完全背包的排列组合问题
组合问题 leetcode518 先遍历物品，后遍历背包
排列问题 leetcode377 先遍历背包，后遍历物品

个数问题 求排列组合中元素最小个数而非求排列组合的数量，此时先遍历背包还是先遍历物品都可以。


*/

/*
组合问题
*/
//#0-1背包，不可重复
func _templateBackPack01(nums []int, target int) int {
	return 0
}

// #完全背包，可重复，无序，算重量
// #完全背包，可重复，有序，算次数
func _templateBackPackTotalDup(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i <= target; i++ {
		for _, num := range nums {
			if i >= num && dp[i-num] > 0 {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}

/*相关题目
377 组合总和 Ⅳ
494 目标和
518 零钱兑换 II
*/

// 377  完全背包，可重复，无序，算重量
func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i <= target; i++ {
		for _, num := range nums {
			if i >= num && dp[i-num] > 0 {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}

// 494  0-1背包，不可重复
func findTargetSumWays(nums []int, target int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	//+和为A，-和为B A+B=target A-B=sum A=(target+sum)/2
	t := (sum + target) / 2
	if (sum+target)%2 == 1 || sum < target || t < 0 {
		return 0
	}
	dp := make([][]int, len(nums)+1)
	for i := 0; i < len(nums)+1; i++ {
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

//true、false问题
//最大最小问题
