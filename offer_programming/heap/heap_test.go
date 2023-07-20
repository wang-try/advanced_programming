package heap

import (
	"container/heap"
	"fmt"
	"testing"
)

func TestMyHeap_Push(t *testing.T) {
	nums := []int{0, 45, 7, 6, 9, 2, 3}
	h := new(intHeap)
	for _, num := range nums {
		h.Push(num)
	}
	heap.Init(h)
	heap.Push(h, 23)
	heap.Pop(h)

	fmt.Println(h)
}

func TestKthLargest_Add1(t *testing.T) {
	k := Constructor(2, []int{4, 2, 6})
	fmt.Println(k.Add(45))
	fmt.Println(k.Add(1))
	fmt.Println(k.Add(7))
	fmt.Println(k.Add(9))
}

func TestKthLargest_Add2(t *testing.T) {
	k := Constructor(1, []int{})
	fmt.Println(k.Add(45))
	fmt.Println(k.Add(1))
	fmt.Println(k.Add(7))
	fmt.Println(k.Add(9))
}

func TestTopKFrequent(t *testing.T) {
	fmt.Println(TopKFrequent([]int{3, 0, 1, 0}, 1))
}

func TestKSmallestPairs(t *testing.T) {
	fmt.Println(KSmallestPairs([]int{1, 1, 2}, []int{1, 2, 3}, 10))
}
