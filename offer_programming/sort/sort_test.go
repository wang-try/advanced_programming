package sort

import (
	"fmt"
	"testing"
)

func TestMerge(t *testing.T) {
	fmt.Println(Merge([][]int{{1, 3}, {4, 5}, {8, 10}, {2, 6}, {9, 12}, {15, 18}}))
}

func TestSortArray(t *testing.T) {
	fmt.Println(SortArray([]int{2, 3, 4, 2, 3, 2, 1}))
}

func TestRelativeSortArray(t *testing.T) {
	fmt.Println(RelativeSortArray([]int{2, 3, 3, 7, 3, 9, 2, 1, 7, 2}, []int{3, 2, 1}))
}

func TestQuickSort(t *testing.T) {
	arr := []int{99, 1, 23, 2, 6, 5, 10, 7, 23}
	QuickSort(arr, 0, 8)
	fmt.Println(arr)

}

func TestFindKthLargest(t *testing.T) {
	fmt.Println(FindKthLargest([]int{99, 1, 23, 2, 6, 5, 10, 7, 23}, 3))
}

func TestMergeSort(t *testing.T) {
	arr := []int{99, 1, 23, 2, 6, 5, 10, 7, 23}
	MergeSort(arr, 0, 9)
	fmt.Println(arr)
}

func TestSortListV2(t *testing.T) {
	head := &ListNode{
		Val: 3,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val:  1,
				Next: nil,
			},
		},
	}
	cur := SortListV2(head)
	for cur != nil {
		fmt.Println(cur.Val)
		cur = cur.Next
	}
}
