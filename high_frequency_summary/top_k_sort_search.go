package high_frequency_summary

import "container/heap"

// 排序算法 & 查找 & top k
// 快排
func quickSort(nums []int, start, end int) {
	if start < end {
		p := partition(nums, start, end)
		quickSort(nums, start, p-1)
		quickSort(nums, p+1, end)
	}
}

func partition(nums []int, start, end int) int {
	pivot := nums[end]
	smallIndex := start
	for i := start; i < end; i++ {
		if nums[i] < pivot {
			nums[smallIndex], nums[i] = nums[i], nums[smallIndex]
			smallIndex++
		}
	}
	nums[smallIndex], nums[end] = nums[end], nums[smallIndex]
	return smallIndex

}

// 堆排序
func heapSort(nums []int) {
	lth := len(nums)
	for i := (lth - 2) / 2; i >= 0; i-- {
		adjustHeap(nums, i, lth)
	}

	for i := lth - 1; i >= 0; i-- {
		nums[i], nums[0] = nums[0], nums[i]
		adjustHeap(nums, 0, i)
	}

}

func adjustHeap(nums []int, index, lth int) {
	father := index
	child := 2*father + 1
	for child < lth {
		if child+1 < lth && nums[child+1] > nums[child] {
			child++
		}
		if nums[father] < nums[child] {
			nums[father], nums[child] = nums[child], nums[father]
			father = child
			child = 2*father + 1
		} else {
			break
		}
	}

}

// leetcode 面试题 17.14. 最小K个数
func smallestK(arr []int, k int) []int {
	smallestKHelp(arr, k, 0, len(arr)-1)
	return arr[:k]
}

func smallestKHelp(arr []int, k, start, end int) {
	if start < end {
		p := partition(arr, start, end)
		if p == k-1 {
			return
		} else if p < k-1 {
			smallestKHelp(arr, k, p+1, end)
		} else {
			smallestKHelp(arr, k, start, p-1)
		}
	}
}

type intHeap []int

func (h *intHeap) Less(i, j int) bool {
	return (*h)[i] < (*h)[j]
}

func (h *intHeap) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

func (h *intHeap) Len() int {
	return len(*h)
}

func (h *intHeap) Pop() (v any) {
	*h, v = (*h)[:h.Len()-1], (*h)[h.Len()-1]
	return
}

func (h *intHeap) Push(v any) {
	*h = append(*h, v.(int))
}

func findKthLargest(nums []int, k int) int {
	h := new(intHeap)
	lth := len(nums)

	for i := 0; i < k; i++ {
		h.Push(nums[i])
	}
	heap.Init(h)
	for i := k; i < lth; i++ {
		heap.Push(h, nums[i])
		heap.Pop(h)
	}
	return (*h)[0]
}

// 347. 前 K 个高频元素
type mapHeap struct {
	nums    []int
	num2feq map[int]int
}

//type mp mapHeap

func (mh *mapHeap) Less(i, j int) bool {
	return mh.num2feq[mh.nums[i]] < mh.num2feq[mh.nums[j]]
}

func (mh *mapHeap) Swap(i, j int) {
	mh.nums[i], mh.nums[j] = mh.nums[j], mh.nums[i]
}

func (mh *mapHeap) Len() int {
	return len(mh.nums)
}

func (mh *mapHeap) Pop() (v any) {
	mh.nums, v = mh.nums[:mh.Len()-1], mh.nums[mh.Len()-1]
	return
}

func (mh *mapHeap) Push(v any) {
	mh.nums = append(mh.nums, v.(int))
}

//出现频率最高的k个数字
/*
请找出数组中出现频率最高的k个数字。例如，当k等于2时，输入数组[1，2，2，1，3，1]，由于数字1出现了3次，数字2出现了2次，数字3出现了1次，因此出现频率最高的2个数字是1和2。
*/
func TopKFrequent(nums []int, k int) []int {
	mh := mapHeap{
		nums:    make([]int, 0, k),
		num2feq: make(map[int]int),
	}
	for _, num := range nums {
		mh.num2feq[num]++
	}
	cnt := 0
	isInit := false
	for num, _ := range mh.num2feq {
		if cnt < k {
			mh.Push(num)
			cnt++
		} else {
			if !isInit {
				heap.Init(&mh)
				isInit = true
			}
			heap.Push(&mh, num)
			heap.Pop(&mh)
		}
	}
	return mh.nums
}
