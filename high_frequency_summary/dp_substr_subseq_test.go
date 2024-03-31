package high_frequency_summary

import (
	"fmt"
	"testing"
)

func TestLengthOfLISV2(t *testing.T) {
	fmt.Println(lengthOfLISV2([]int{0, 3, 1, 6, 2, 2, 7}))
}

func TestFindNumberOfLIS(t *testing.T) {
	fmt.Println(findNumberOfLIS([]int{1, 2, 4, 3, 5, 4, 7, 2}))
}

func TestFindTargetSumWays(t *testing.T) {
	fmt.Println(findTargetSumWays([]int{100}, -200))
}

func TestMinWindow(t *testing.T) {
	fmt.Println(minWindow("ADOBECODEBANC", "ABC"))
}

func TestRemoveDuplicateLetters(t *testing.T) {
	fmt.Println(removeDuplicateLetters("cbacdcbc"))
}
