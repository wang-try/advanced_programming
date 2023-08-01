package backtrack

import (
	"fmt"
	"testing"
)

func TestSubsets(t *testing.T) {
	fmt.Println(Subsets([]int{1, 2, 3, 4}))
}

func TestCombine(t *testing.T) {
	fmt.Println(Combine(4, 2))
}

func TestCombinationSum(t *testing.T) {
	fmt.Println(CombinationSum([]int{2, 3, 5}, 8))
}

func TestCombinationSum2(t *testing.T) {
	fmt.Println(combinationSum2([]int{2, 2, 2, 4, 3, 3}, 8))
}

func TestPermute(t *testing.T) {
	fmt.Println(permute([]int{1, 2, 3}))
}

func TestPermuteUnique(t *testing.T) {
	fmt.Println(permuteUnique([]int{2, 2, 1, 1}))
}
