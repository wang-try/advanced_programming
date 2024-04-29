package high_frequency_summary

import (
	"fmt"
	"testing"
)

func TestMaximalRectangle(t *testing.T) {
	fmt.Println(maximalRectangle([][]byte{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {'1', '1', '1', '1', '1'}, {'1', '0', '0', '1', '0'}}))
}

func TestSpiralOrder(t *testing.T) {
	fmt.Println(spiralOrder([][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}))
}

func TestFindDisappearedNumbersV2(t *testing.T) {
	fmt.Println(findDisappearedNumbersV2([]int{4, 3, 2, 7, 8, 2, 3, 1}))
}

func TestFindClosestElements(t *testing.T) {
	fmt.Println(findClosestElements([]int{1, 25, 35, 45, 50, 59}, 1, 30))
}

func TestReverseInt(t *testing.T) {
	fmt.Println(reverseInt(123))
}

func TestFractionToDecimal(t *testing.T) {
	fmt.Println(fractionToDecimal(4, 333))
}

func TestSplitIntoFibonacci(t *testing.T) {
	fmt.Println(splitIntoFibonacci("1101111"))
}

func TestFindPeakElement(t *testing.T) {
	fmt.Println(findPeakElementV2([]int{1, 2}))
}

func TestSubSets(t *testing.T) {
	fmt.Println(subsets([]int{1, 2, 3}))
}

func TestNextGreaterElements(t *testing.T) {
	fmt.Println(nextGreaterElements([]int{2, 3, 1}))
}

func TestCrackPassword(t *testing.T) {
	fmt.Println(crackPassword([]int{0, 3, 30, 34, 5, 9}))
}

func TestLongestLine(t *testing.T) {
	fmt.Println(longestLine([][]int{{}, {}}))
}

func TestCombinationSum2(t *testing.T) {
	fmt.Println(combinationSum2([]int{10, 1, 2, 7, 6, 1, 5}, 8))
}
