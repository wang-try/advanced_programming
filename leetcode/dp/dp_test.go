package dp

import (
	"fmt"
	"testing"
)

func TestGetMaximumGenerated(t *testing.T) {
	fmt.Println(getMaximumGenerated(7))
}

func TestJump(t *testing.T) {
	fmt.Println(jump([]int{2, 3, 1, 1, 4}))
	fmt.Println(jumpV2([]int{2, 3, 1, 1, 4}))
}

func TestRob(t *testing.T) {
	fmt.Println(rob([]int{4, 1, 2, 7, 5, 3, 1}))
}

func TestDiffWaysToCompute(t *testing.T) {
	fmt.Println(diffWaysToCompute("2-1-1"))
}
