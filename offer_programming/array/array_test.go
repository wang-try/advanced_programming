package array

import (
	"fmt"
	"testing"
)

func TestThreeSum(t *testing.T) {
	fmt.Println(ThreeSum([]int{-1, 0, 1, 2, -1, -4}))
}

func TestMinSubArrayLen(t *testing.T) {
	fmt.Println(MinSubArrayLen(7, []int{5, 1, 4, 3}))
}

func TestNumSubarrayProductLessThanK(t *testing.T) {
	fmt.Println(NumSubarrayProductLessThanK([]int{10, 5, 2, 6}, 100))
}

func TestFindMaxLength01Same(t *testing.T) {
	fmt.Println(FindMaxLength01Same([]int{0, 1, 0, 1}))
}
