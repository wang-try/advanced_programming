package binary_search

import (
	"fmt"
	"testing"
)

func TestSearchInsert(t *testing.T) {
	fmt.Println(SearchInsert([]int{1, 3, 6, 8}, 9))
}

func TestPeakIndexInMountainArray(t *testing.T) {
	fmt.Println(PeakIndexInMountainArray([]int{1, 3, 5, 4, 2}))
}

func TestSingleNonDuplicate(t *testing.T) {
	fmt.Println(SingleNonDuplicate([]int{1, 1, 2, 2, 3, 4, 4, 5, 5}))
}

func TestSingleNonDuplicate2(t *testing.T) {
	fmt.Println(SingleNonDuplicateV2([]int{1, 1, 2, 2, 3, 4, 4, 5, 5}))
}

func TestConstructor(t *testing.T) {
	s := Constructor([]int{3, 4, 1, 7})

	fmt.Println(s.PickIndex())
	fmt.Println(s.PickIndex())
	fmt.Println(s.PickIndex())
	fmt.Println(s.PickIndex())
}

func TestMySqrt(t *testing.T) {
	fmt.Println(MySqrt(4))
	fmt.Println(MySqrt(18))
	fmt.Println(MySqrt(9))
	fmt.Println(MySqrt(6))
}

func TestMinEatingSpeed(t *testing.T) {
	fmt.Println(MinEatingSpeed([]int{3, 6, 7, 11}, 8))
}
