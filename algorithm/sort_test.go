package algorithm

import (
	"fmt"
	"testing"
)

func TestHeapSort(t *testing.T) {
	nums := []int{4, 2, 3, 66, 20, 6, 8, 5, 5}
	//HeapSortV2(nums)
	//nums := []int{9, 8, 7, 6, 5, 5, 3, 2, 1}
	HeapSort(nums)
	fmt.Println(nums)
}
