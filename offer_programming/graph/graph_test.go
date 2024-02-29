package graph

import (
	"fmt"
	"testing"
)

func TestUpdateMatrix(t *testing.T) {
	fmt.Println(updateMatrix([][]int{{0, 0, 0}, {0, 1, 0}, {1, 1, 1}}))
}

func TestLadderLength(t *testing.T) {
	fmt.Println(ladderLength("hit", "cog", []string{"hot", "dot", "dog", "lot", "log", "cog"}))
	fmt.Println(ladderLengthV2("hit", "cog", []string{"hot", "dot", "dog", "lot", "log"}))
}

func TestOpenLock(t *testing.T) {
	fmt.Println(openLock([]string{"0201", "0101", "0102", "1212", "2002"}, "0202"))
}

func TestAllPathsSourceTarget(t *testing.T) {
	//[[1，2]，[3]，[3]，[]]
	//[4,3,1],[3,2,4],[3],[4],[]
	fmt.Println(allPathsSourceTarget([][]int{{4, 3, 1}, {3, 2, 4}, {3}, {4}, {}}))
}

func TestCalcEquation(t *testing.T) {

	//[["a","b"],["b","c"]]
	//values =
	//[2.0,3.0]
	//queries =
	//[["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]

	fmt.Println(calcEquation([][]string{{"a", "b"}, {"b", "c"}}, []float64{2.0, 3.0}, [][]string{{"a", "c"}, {"b", "a"}, {"a", "e"}, {"a", "a"}, {"x", "x"}}))
}

func TestLongestIncreasingPath(t *testing.T) {
	//[[9,9,4],[6,6,8],[2,1,1]]
	fmt.Println(longestIncreasingPath([][]int{{9, 9, 4}, {6, 6, 8}, {2, 1, 1}}))
}

func TestAlienOrder(t *testing.T) {
	//["ac"，"ab"，"bc"，"zc"，"zb"]
	fmt.Println(alienOrder([]string{"ac", "ab", "bc", "zc", "zb"}))
}

func TestSequenceReconstruction(t *testing.T) {
	fmt.Println(sequenceReconstruction([]int{1, 2, 3}, [][]int{{1, 2}, {1, 3}}))
}

func TestFindRedundantConnection(t *testing.T) {
	fmt.Println(findRedundantConnection([][]int{{1, 2}, {1, 3}, {2, 3}}))
}

func TestLongestConsecutive(t *testing.T) {
	fmt.Println(longestConsecutive([]int{10, 5, 9, 2, 4, 3}))
	//[9,1,4,7,3,-1,0,5,8,-1,6]
	fmt.Println(longestConsecutiveV2([]int{9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6}))
}

func TestMaxAreaOfIslandV2(t *testing.T) {
	fmt.Println(maxAreaOfIslandV2([][]int{{1, 1}}))
}
