package heap

import "container/heap"

/*如果用数组表示堆，那么数组中的每个元素对应堆的一个节点。如果数组中的一个元素的下标为i，那么它在堆中对应节点的父节点在数组的
下标为(i-1)/2，而它的左右节点在数组中的下标分别为2i+1和2i+2
*/
//数据流的第k大数字
/*
请设计一个类型KthLargest，它每次从一个数据流中读取一个数字，并得出数据流已经读取的数字中第k（k≥1）大的数字。该类型的构造函数有两个参数：
一个是整数k，另一个是包含数据流中最开始数字的整数数组nums（假设数组nums的长度大于k）。该类型还有一个函数add，用来添加数据流中的新数字并返回数据流中已经读取的数字的第k大数字。
*/

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

func (h *intHeap) Root() int {
	if h.Len() > 0 {
		return (*h)[0]
	}
	return -1
}

type KthLargest struct {
	Heap *intHeap
	K    int
}

func Constructor(k int, nums []int) KthLargest {
	h := new(intHeap)
	i := 0
	for ; i < len(nums) && i < k; i++ {
		h.Push(nums[i])
	}
	heap.Init(h)
	for i = k; i < len(nums); i++ {
		heap.Push(h, nums[i])
		heap.Pop(h)
	}
	return KthLargest{
		Heap: h,
		K:    k,
	}

}

func (this *KthLargest) Add(val int) int {
	heap.Push(this.Heap, val)
	if this.Heap.Len() <= this.K {
		return this.Heap.Root()
	}
	heap.Pop(this.Heap)
	return this.Heap.Root()
}

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

//和最小的k个数对
/*
给定两个递增排序的整数数组，从两个数组中各取一个数字u和v组成一个数对（u，v），请找出和最小的k个数对。例如，输入两个数组[1，5，13，21]和[2，4，9，15]，和最小的3个数对为（1，2）、（1，4）和（2，5）。
*/

type pairHeap [][]int

func (ph *pairHeap) Less(i, j int) bool {
	return ((*ph)[i][0] + (*ph)[i][1]) > ((*ph)[j][0] + (*ph)[j][1])
}

func (ph *pairHeap) Swap(i, j int) {
	(*ph)[i], (*ph)[j] = (*ph)[j], (*ph)[i]
}

func (ph *pairHeap) Len() int {
	return len(*ph)
}

func (ph *pairHeap) Pop() (v any) {
	*ph, v = (*ph)[:ph.Len()-1], (*ph)[ph.Len()-1]
	return
}

func (ph *pairHeap) Push(v any) {
	*ph = append(*ph, v.([]int))
}

func KSmallestPairs(nums1 []int, nums2 []int, k int) [][]int {
	cnt := 0
	var ph pairHeap
	isInit := false
	for i := 0; i < min(k, len(nums1)); i++ {
		for j := 0; j < min(k, len(nums2)); j++ {
			pair := []int{nums1[i], nums2[j]}
			if cnt < k {
				ph.Push(pair)
			} else {
				if !isInit {
					heap.Init(&ph)
					isInit = true
				}
				heap.Push(&ph, pair)
				heap.Pop(&ph)
			}
			cnt++
		}
	}
	return ph
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
