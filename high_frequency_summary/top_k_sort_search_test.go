package high_frequency_summary

import (
	"fmt"
	"testing"
)

func TestQuickSort(t *testing.T) {
	nums := []int{2, 2, 1, 3, 9, 6, 10, 7}
	quickSort(nums, 0, 7)
	fmt.Println(nums)
}

func TestHeapSort(t *testing.T) {
	nums := []int{2, 2, 1, 3, 9, 6, 10, 7}
	heapSort(nums)
	fmt.Println(nums)
}

func TestSmallestK(t *testing.T) {
	fmt.Println(smallestK([]int{2, 2, 1, 3, 9, 6, 10, 7}, 7))
}

func TestFindKthLargest(t *testing.T) {
	fmt.Println(findKthLargest([]int{3, 2, 1, 5, 6, 4}, 2))
}
