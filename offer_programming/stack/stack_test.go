package stack

import (
	"fmt"
	"testing"
)

func TestEvalRPN(t *testing.T) {
	fmt.Println(EvalRPN([]string{"4", "13", "5", "/", "+"}))
}

func TestAsteroidCollision(t *testing.T) {
	fmt.Println(AsteroidCollision([]int{4, 5, -6, 4, 8, -5}))
}

func TestDailyTemperatures(t *testing.T) {
	fmt.Println(DailyTemperatures([]int{35, 31, 33, 36, 34}))
}

func TestLargestRectangleArea(t *testing.T) {
	fmt.Println(LargestRectangleAreaV4([]int{9, 0}))
}
