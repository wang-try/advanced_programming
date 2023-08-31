package dynamic_programming

import (
	"fmt"
	"testing"
)

func TestMinCostClimbingStairs(t *testing.T) {
	fmt.Println(minCostClimbingStairs([]int{1, 100, 1, 1, 100, 1}))
	fmt.Println(minCostClimbingStairs([]int{10, 15, 20}))
}

func TestRob(t *testing.T) {
	fmt.Println(rob([]int{2, 1, 1, 2}))
	fmt.Println(robV2([]int{2, 1, 1, 2}))
}

func TestRobCircle(t *testing.T) {
	fmt.Println(robCircle([]int{1, 2, 3, 1}))
	fmt.Println(robCircleV2([]int{1, 2, 3, 1}))
}

// [[17，2，16]，[15，14，5]，[13，3，1]]
func TestMinCosts(t *testing.T) {
	fmt.Println(minCost([][]int{{17, 2, 16}, {15, 14, 5}, {13, 3, 1}}))
}

func TestMinFlipsMonoIncr(t *testing.T) {
	fmt.Println(minFlipsMonoIncr("010110"))
	fmt.Println(minFlipsMonoIncrV2("010110"))
	fmt.Println(minFlipsMonoIncrV3("010110"))
}

func TestLenLongestFibSubSeq(t *testing.T) {
	fmt.Println(lenLongestFibSubSeq([]int{1, 2, 3, 4, 5, 6, 7, 8}))
}

func TestMinCut(t *testing.T) {
	fmt.Println(minCut("aab"))
}

func TestLongestCommonSubsequence(t *testing.T) {
	fmt.Println(longestCommonSubsequence("bl", "yby"))
}

// 字符串"aadbbcbcac"可以由字符串"aabcc"和"dbbca"交织而成
func TestIsInterleave(t *testing.T) {
	fmt.Println(isInterleave("aabcc", "dbbca", "aadbbcbcac"))
}

func TestNumDistinct(t *testing.T) {
	fmt.Println(numDistinct("babgbag", "bag"))
}

func TestUniquePaths(t *testing.T) {
	fmt.Println(uniquePaths(3, 3))
}

func TestMiniMumTotal(t *testing.T) {
	//[[2],[3,4],[6,5,7],[4,1,8,3]]
	fmt.Println(minimumTotal([][]int{{2}, {3, 4}, {6, 5, 7}, {4, 1, 8, 3}}))
}

func TestCanPartition(t *testing.T) {
	fmt.Println(canPartition([]int{1, 5, 11, 5}))
}
