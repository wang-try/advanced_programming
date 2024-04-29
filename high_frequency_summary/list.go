package high_frequency_summary

type ListNode struct {
	Val  int
	Next *ListNode
}

// leetcode 25. K 个一组翻转链表
func reverseKGroup(head *ListNode, k int) *ListNode {
	lth := 0
	cur := head

	for cur != nil {
		lth++
		cur = cur.Next
	}
	dummy := new(ListNode)
	start := head
	iter := dummy
	for lth >= k {
		h, t := reverseHelp(start, k)
		iter.Next = h
		iter = start
		start = t
		lth -= k
	}
	iter.Next = start
	return dummy.Next
}

func reverseHelp(start *ListNode, k int) (head, tail *ListNode) {
	pre := start
	cur := start.Next
	step := 1
	for cur != nil && step < k {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
		step++
	}
	head = pre
	tail = cur
	return
}

// leetcode141. 环形链表
func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

// leetcode142. 环形链表 II
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			//meet
			meet := slow
			cur := head
			for cur != nil {
				if meet == cur {
					return meet
				}
				meet = meet.Next
				cur = cur.Next
			}
		}

	}

	return nil
}

// leetcode160. 相交链表
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	curA, curB := headA, headB
	lthA, lthB := 0, 0
	for curA != nil {
		curA = curA.Next
		lthA++
	}

	for curB != nil {
		curB = curB.Next
		lthB++
	}

	short, long := headA, headB
	step := lthB - lthA
	if lthA >= lthB {
		long, short = headA, headB
		step = lthA - lthB
	}

	for step > 0 {
		long = long.Next
		step--
	}

	for short != nil && long != nil {
		if long == short {
			return short
		}
		short = short.Next
		long = long.Next
	}
	return nil

}

// leetcode234. 回文链表
func isPalindromeList(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	reverseHead := slow.Next
	var pre *ListNode = nil
	cur := reverseHead
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	reverseHead = pre

	for reverseHead != nil {
		if head.Val != reverseHead.Val {
			return false
		}
		reverseHead = reverseHead.Next
		head = head.Next
	}
	return true
}

// leetcode876. 链表的中间结点
func middleNode(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

// leetcode19. 删除链表的倒数第 N 个结点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	slow, fast := head, head
	for i := 1; i <= n && fast != nil; i++ {
		fast = fast.Next
	}

	if fast == nil {
		return head.Next
	}

	for fast != nil && fast.Next != nil {
		fast = fast.Next
		slow = slow.Next
	}

	slow.Next = slow.Next.Next
	return head
}

// leetcode面试题 02.02. 返回倒数第 k 个节点
func kthToLast(head *ListNode, k int) int {
	slow, fast := head, head
	for i := 1; i <= k-1 && fast != nil; i++ {
		fast = fast.Next
	}
	for fast != nil && fast.Next != nil {
		fast = fast.Next
		slow = slow.Next
	}
	return slow.Val
}

// leetcode206. 反转链表
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode = nil
	cur := head
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

// leetcode92. 反转链表 II TODO 不理解？？？
func reverseBetween(head *ListNode, left, right int) *ListNode {
	// 设置 dummyNode 是这一类问题的一般做法
	dummyNode := new(ListNode)
	dummyNode.Next = head
	pre := dummyNode
	for i := 0; i < left-1; i++ {
		pre = pre.Next
	}
	cur := pre.Next
	for i := 0; i < right-left; i++ {
		next := cur.Next
		cur.Next = next.Next
		next.Next = pre.Next
		pre.Next = next
	}
	return dummyNode.Next
}

func reverseBetweenV2(head *ListNode, left int, right int) *ListNode {
	var pre *ListNode = nil
	cur := head
	var spiltPre, splitCur *ListNode

	for i := 1; i <= right && cur != nil; i++ {
		if i == left {
			spiltPre = pre
			splitCur = cur
			pre = cur
			cur = cur.Next
		} else if i > left && i <= right {
			next := cur.Next
			cur.Next = pre
			pre = cur
			cur = next
		} else {
			pre = cur
			cur = cur.Next
		}
	}
	splitCur.Next = cur
	if spiltPre != nil {
		spiltPre.Next = pre
	} else {
		return pre
	}

	return head
}

// leetcode82. 删除排序链表中的重复元素 II
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	dummy := new(ListNode)
	iter := dummy
	var pre *ListNode = nil
	cur := head
	for cur != nil {
		next := cur.Next

		if (pre == nil && cur.Val != next.Val) ||
			(pre != nil && next != nil && cur.Val != pre.Val && cur.Val != next.Val) ||
			(pre != nil && next == nil && cur.Val != pre.Val) {
			iter.Next = cur
			iter = iter.Next
		}
		pre = cur
		cur = next
	}
	iter.Next = nil
	return dummy.Next
}

// leetcode83 删除排序链表中的重复元素
func deleteDuplicatesII(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	pre, cur := head, head.Next
	for cur != nil {
		if cur.Val != pre.Val {
			pre.Next = cur
			pre = cur
		}
		cur = cur.Next
	}
	pre.Next = nil
	return head
}

// leetcode2. 两数相加
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	carry := 0
	dummy := new(ListNode)
	cur := dummy
	for l1 != nil || l2 != nil {
		sum := 0
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}

		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}

		sum += carry
		num := sum % 10
		carry = sum / 10
		cur.Next = &ListNode{
			Val:  num,
			Next: nil,
		}
		cur = cur.Next
	}

	if carry > 0 {
		cur.Next = &ListNode{
			Val:  carry,
			Next: nil,
		}
	}

	return dummy.Next

}

// leetcode445. 两数相加 II
func addTwoNumbersII(l1 *ListNode, l2 *ListNode) *ListNode {
	l1 = reverseList(l1)
	l2 = reverseList(l2)
	var cur *ListNode = nil
	carry := 0
	for l1 != nil || l2 != nil {
		sum := 0
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}

		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}

		sum += carry
		num := sum % 10
		carry = sum / 10
		node := &ListNode{
			Val:  num,
			Next: cur,
		}

		cur = node
	}

	head := cur
	if carry > 0 {
		head = &ListNode{
			Val:  carry,
			Next: cur,
		}
	}
	return head
}

// leetcode21. 合并两个有序链表
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := new(ListNode)
	cur := dummy
	for list1 != nil || list2 != nil {
		if list1 == nil || list2 != nil && list2.Val <= list1.Val {
			cur.Next = list2
			list2 = list2.Next
			cur = cur.Next
		} else {
			cur.Next = list1
			list1 = list1.Next
			cur = cur.Next
		}
	}
	return dummy.Next
}

// leetcode23
func mergeKLists(lists []*ListNode) *ListNode {
	dummy := new(ListNode)
	cur := dummy
	for {
		var minNode *ListNode = nil
		minIndex := -1
		for i, list := range lists {
			if list != nil {
				if minNode == nil || (minNode != nil && minNode.Val > list.Val) {
					minNode = list
					minIndex = i
				}
			}
		}

		if minIndex == -1 {
			break
		}

		cur.Next = minNode
		cur = cur.Next
		lists[minIndex] = lists[minIndex].Next
	}
	return dummy.Next
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

func mergeKListsV2(lists []*ListNode) *ListNode {
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

func reverseKGroupV2(head *ListNode, k int) *ListNode {
	lth := 0
	cur := head
	for cur != nil {
		lth++
		cur = cur.Next
	}

	start := head
	dummy := new(ListNode)
	dummyHead := dummy
	for lth >= k {
		h, t := reverseKGroupHelp(start, k)
		dummy.Next = h
		dummy = start
		start = t
		lth -= k
	}
	dummy.Next = start
	return dummyHead.Next
}

func reverseKGroupHelp(start *ListNode, k int) (head, tail *ListNode) {
	pre, cur := start, start.Next
	step := 1
	for cur != nil && step < k {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
		step++
	}
	head, tail = pre, cur
	return
}

// leetcode143. 重排链表
func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}

	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	tail := slow.Next
	slow.Next = nil
	rHead := reverseList(tail)

	cur := head
	for rHead != nil {
		next := cur.Next
		cur.Next = rHead
		rNext := rHead.Next
		rHead.Next = next
		rHead = rNext
		cur = next
	}
}

// leetcode328. 奇偶链表
func oddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	oddDummy, evenDummy := new(ListNode), new(ListNode)
	cur := head
	evenHead := evenDummy
	for i := 1; cur != nil; i, cur = i+1, cur.Next {
		if i&1 == 1 {
			oddDummy.Next = cur
			oddDummy = oddDummy.Next
		} else {
			evenDummy.Next = cur
			evenDummy = evenDummy.Next
		}
	}
	evenDummy.Next = nil
	oddDummy.Next = evenHead.Next

	return head
}

// leetcode86. 分隔链表
func partitionList(head *ListNode, x int) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	smallDummy, bigDummy := new(ListNode), new(ListNode)
	cur := head
	bigHead, smallHead := bigDummy, smallDummy
	for cur != nil {
		if cur.Val < x {
			smallDummy.Next = cur
			smallDummy = smallDummy.Next
		} else {
			bigDummy.Next = cur
			bigDummy = bigDummy.Next
		}
		cur = cur.Next
	}

	bigDummy.Next = nil
	smallDummy.Next = bigHead.Next
	return smallHead.Next

}
