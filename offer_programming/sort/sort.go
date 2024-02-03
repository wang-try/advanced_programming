package sort

import (
	"math"
	"sort"
)

//合并区间
/*
输入一个区间的集合，请将重叠的区间合并。每个区间用两个数字比较，分别表示区间的起始位置和结束位置。例如，输入区间[[1，3]，[4，5]，[8，10]，[2，6]，[9，12]，[15，18]]，
合并重叠的区间之后得到[[1，6]，[8，12]，[15，18]]。
*/
func Merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	var res [][]int
	start := intervals[0][0]
	end := intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] <= end {
			if intervals[i][1] > end {
				end = intervals[i][1]
			}
		} else {
			res = append(res, []int{start, end})
			start = intervals[i][0]
			end = intervals[i][1]
		}
	}
	res = append(res, []int{start, end})
	return res
}

/*
计数排序是一种线性时间的整数排序算法。如果数组的长度为n，整数范围（数组中最大整数与最小整数的差值）为k，
对于k远小于n的场景（如对某公司所有员工的年龄排序），那么计数排序的时间复杂度优于其他基于比较的排序算法（如归并排序、快速排序等）。
*/
// 计数排序
func SortArray(nums []int) []int {
	min := math.MaxInt
	max := math.MinInt
	for _, num := range nums {
		if num < min {
			min = num
		}
		if num > max {
			max = num
		}
	}
	counts := make([]int, max-min+1)
	for _, num := range nums {
		counts[num-min]++
	}
	i := 0
	for num := min; num <= max; num++ {
		for counts[num-min] > 0 {
			nums[i] = num
			i++
			counts[num-min]--
		}
	}
	return nums
}

//数组相对排序
/*
输入两个数组arr1和arr2，其中数组arr2中的每个数字都唯一，并且都是数组arr1中的数字。请将数组arr1中的数字按照数组arr2中的数字的相对顺序排序。
如果数组arr1中的数字在数组arr2中没有出现，那么将这些数字按递增的顺序排在后面。假设数组中的所有数字都在0到1000的范围内。
例如，输入的数组arr1和arr2分别是[2，3，3，7，3，9，2，1，7，2]和[3，2，1]，则数组arr1排序之后为[3，3，3，2，2，2，1，7，7，9
*/

func RelativeSortArray(arr1 []int, arr2 []int) []int {
	var counts [1001]int
	for _, num := range arr1 {
		counts[num]++
	}
	i := 0
	for _, num := range arr2 {
		for counts[num] > 0 {
			arr1[i] = num
			i++
			counts[num]--
		}
	}
	for num, cnt := range counts {
		for cnt > 0 {
			arr1[i] = num
			i++
			cnt--
		}
	}
	return arr1
}

// 快速排序
func QuickSort(nums []int, start, end int) {
	if start < end {
		p := partition(nums, start, end)
		QuickSort(nums, start, p-1)
		QuickSort(nums, p+1, end)
	}
}

func partition(nums []int, start, end int) int {
	pivot := nums[end]
	smallIndex := start
	for i := start; i < end; i++ {
		if nums[i] < pivot {
			nums[i], nums[smallIndex] = nums[smallIndex], nums[i]
			smallIndex++
		}
	}
	nums[smallIndex], nums[end] = nums[end], nums[smallIndex]
	return smallIndex
}

//数组中第k大的数字
/*
请从一个乱序数组中找出第k大的数字。例如，数组[3，1，2，4，5，5，6]中第3大的数字是5。
*/
func FindKthLargest(nums []int, k int) int {
	target := len(nums) - k
	start := 0
	end := len(nums) - 1
	index := partition(nums, start, end)
	for index != target {
		if index > target {
			end = index - 1
		} else {
			start = index + 1
		}
		index = partition(nums, start, end)
	}
	return nums[index]
}

// 归并排序
func MergeSort(nums []int, start int, end int) {
	if start < end-1 {
		mid := (start + end) / 2
		MergeSort(nums, start, mid)
		MergeSort(nums, mid, end)
		mergeArr(nums, start, mid, end)
	}
}

func mergeArr(nums []int, start, mid, end int) {
	var tmp []int

	i, j := start, mid
	for i < mid || j < end {
		if j == end || i < mid && nums[i] < nums[j] {
			tmp = append(tmp, nums[i])
			i++
		} else {
			tmp = append(tmp, nums[j])
			j++
		}

	}
	for i, j = 0, start; i < len(tmp); i, j = i+1, j+1 {
		nums[j] = tmp[i]
	}
}

//链表排序
/*
输入一个链表的头节点，请将该链表排序
*/

// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

// 超出时间限制
func sortList(head *ListNode) *ListNode {
	lth := 0
	cur := head
	for cur != nil {
		lth++
		cur = cur.Next
	}
	for i := 0; i < lth; i++ {
		for j, cmp := 0, head; j < lth-i-1 && cmp != nil && cmp.Next != nil; j, cmp = j+1, cmp.Next {
			if cmp.Val > cmp.Next.Val {
				cmp.Val, cmp.Next.Val = cmp.Next.Val, cmp.Val
			}
		}
	}
	return head
}

func SortListV2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	rightHead := slow.Next
	slow.Next = nil
	return mergeList(SortListV2(head), SortListV2(rightHead))

}

func mergeList(l1, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	dummyHead := dummy
	for l1 != nil || l2 != nil {
		if l2 == nil || l1 != nil && l1.Val < l2.Val {
			dummy.Next = l1
			l1 = l1.Next
		} else {
			dummy.Next = l2
			l2 = l2.Next
		}
		dummy = dummy.Next
	}
	return dummyHead.Next
}

//合并排序链表
/*
输入k个排序的链表，请将它们合并成一个排序的链表。例如，输入3个排序的链表
*/
func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	return recMergeLists(lists, 0, len(lists))
}

func recMergeLists(lists []*ListNode, start, end int) *ListNode {
	if start+1 == end {
		return lists[start]
	}
	mid := (start + end) / 2
	head1 := recMergeLists(lists, start, mid)
	head2 := recMergeLists(lists, mid, end)
	return mergeList(head1, head2)
}
