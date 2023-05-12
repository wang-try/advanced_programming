package list

import "fmt"

type ListNode struct {
	Val  int
	Next *ListNode
}

//删除倒数第k个节点
/*
如果给定一个链表，请问如何删除链表中的倒数第k个节点？假设链表中节点的总数为n，那么1≤k≤n。要求只能遍历链表一次
*/

func RemoveNthFromEnd(head *ListNode, n int) *ListNode {
	if head == nil {
		return head
	}
	slow := head
	fast := head
	for step := 1; step <= n && fast != nil; step++ {
		fast = fast.Next
	}
	if fast == nil {
		return head.Next
	}

	for fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return head
}

func RemoveNthFromEndV2(head *ListNode, n int) *ListNode {
	dummy := new(ListNode)
	dummy.Next = head
	front := head
	back := dummy
	for i := 1; i <= n; i++ {
		front = front.Next
	}
	for front != nil {
		front = front.Next
		back = back.Next
	}
	back.Next = back.Next.Next
	return dummy.Next
}

//链表中环的入口节点
/*
如果一个链表中包含环，那么应该如何找出环的入口节点？从链表的头节点开始顺着next指针方向进入环的第1个节点为环的入口节点。例如，在如图4.3所示的链表中，环的入口节点是节点3。
*/

func DetectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil {
		slow = slow.Next
		if fast.Next == nil {
			return nil
		}
		fast = fast.Next.Next
		if slow == fast {
			p := head
			for p != slow {
				p = p.Next
				slow = slow.Next
			}
			return p
		}
	}
	return nil
}

//两个链表的第1个重合节点
/*
输入两个单向链表，请问如何找出它们的第1个重合节点。例如，图4.5中的两个链表的第1个重合节点的值是4
*/

func GetIntersectionNode(headA, headB *ListNode) *ListNode {
	lth1 := 0
	lth2 := 0
	cur := headA
	for cur != nil {
		lth1++
		cur = cur.Next
	}
	cur = headB
	for cur != nil {
		lth2++
		cur = cur.Next
	}

	if lth2 > lth1 {
		for i := 0; i < lth2-lth1; i++ {
			headB = headB.Next
		}

	}
	if lth1 > lth2 {
		for i := 0; i < lth1-lth2; i++ {
			headA = headA.Next
		}
	}
	for headA != nil && headB != nil {
		if headB == headA {
			return headA
		}
		headB = headB.Next
		headA = headA.Next
	}
	return nil
}

//反转链表
/*
定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
*/

func ReverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	pre := head
	cur := head.Next
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	head.Next = nil
	return pre
}

//链表中的数字相加
/*
题目：给定两个表示非负整数的单向链表，请问如何实现这两个整数的相加并且把它们的和仍然用单向链表表示？链表中的每个节点表示整数十进制的一位，并且头节点对应整数的最高位数而尾节点对应整数的个位数。
例如，在图4.10（a）和图4.10（b）中，两个链表分别表示整数123和531，它们的和为654，对应的链表如图4.10（c）所示。
*/

func AddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	cur := dummy
	carry := 0
	for l1 != nil || l2 != nil {
		var num1 int
		if l1 != nil {
			num1 = l1.Val
			l1 = l1.Next
		}
		var num2 int
		if l2 != nil {
			num2 = l2.Val
			l2 = l2.Next
		}
		sum := num1 + num2 + carry
		val := sum % 10
		carry = sum / 10
		node := new(ListNode)
		node.Val = val
		cur.Next = node
		cur = cur.Next
	}
	if carry > 0 {
		node := new(ListNode)
		node.Val = carry
		cur.Next = node
		cur = cur.Next
	}
	return dummy.Next

}

//重排链表
/*
给定一个链表，链表中节点的顺序是L0→L1→L2→…→Ln-1→Ln，请问如何重排链表使节点的顺序变成L0→Ln→L1→Ln-1→L2→Ln-2→…？
例如，输入图4.12（a）中的链表，重排之后的链表如图4.12（b）所示
*/
func ReOrderList(head *ListNode) {
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	mid := slow.Next
	slow.Next = nil
	reverseHead := ReverseList(mid)
	cur1 := head
	cur2 := reverseHead
	for cur2 != nil {
		cur1Next := cur1.Next
		cur2Next := cur2.Next
		cur1.Next = cur2
		cur2.Next = cur1Next
		cur2 = cur2Next
		cur1 = cur1Next
	}
}

//回文链表
/*
如何判断一个链表是不是回文？要求解法的时间复杂度是O（n），并且不得使用超过O（1）的辅助空间。如果一个链表是回文，那么链表的节点序列从前往后看和从后往前看是相同的。
例如，图4.13中的链表的节点序列从前往后看和从后往前看都是1、2、3、3、2、1，因此这是一个回文链表。
*/

func IsPalindrome(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	midHead := ReverseList(slow.Next)
	for midHead != nil {
		if midHead.Val != head.Val {
			return false
		}
		midHead = midHead.Next
		head = head.Next
	}
	return true
}

//展平多级双向链表
/*
在一个多级双向链表中，节点除了有两个指针分别指向前后两个节点，还有一个指针指向它的子链表，并且子链表也是一个双向链表，它的节点也有指向子链表的指针。
请将这样的多级双向链表展平成普通的双向链表，即所有节点都没有子链表。
*/

type Node struct {
	Val   int
	Prev  *Node
	Next  *Node
	Child *Node
}

func Flatten(root *Node) *Node {
	cur := root
	for cur != nil {
		next := cur.Next
		if cur.Child != nil {
			sub := Flatten(cur.Child)
			cur.Next = sub
			sub.Prev = cur
			cur.Child = nil
			tail := sub
			for tail.Next != nil {
				tail = tail.Next
			}
			tail.Next = next
			if next != nil {
				next.Prev = tail
			}
		}
		cur = next
	}
	return root
}

func FlattenPrint(root *Node) {
	for root != nil {
		fmt.Println(root.Val)
		next := root.Next
		if root.Child != nil {
			FlattenPrint(root.Child)
		}
		root = next
	}
}

//排序的循环链表
/*
在一个循环链表中节点的值递增排序，请设计一个算法在该循环链表中插入节点，并保证插入节点之后的循环链表仍然是排序的。
*/

func Insert(node *ListNode, val int) *ListNode {
	insertNode := &ListNode{
		Val:  val,
		Next: nil,
	}

	if node == nil {
		return insertNode
	}
	if node.Next == nil {
		if node.Val < val {
			node.Next = insertNode
		} else {
			insertNode.Next = node
		}
	}

	cur := node
	for {
		if cur.Val <= val && cur.Next.Val >= val {
			next := cur.Next
			cur.Next = insertNode
			insertNode.Next = next
			return node
		}
		cur = cur.Next
	}

}

func insert(aNode *Node, x int) *Node {
	insertNode := &Node{
		Val:  x,
		Next: nil,
	}

	if aNode == nil {
		insertNode.Next = insertNode
		return insertNode
	}
	if aNode.Next == aNode {
		aNode.Next = insertNode
		insertNode.Next = aNode
		return aNode
	}

	cur := aNode
	bigest := aNode
	next := aNode.Next
	for !(cur.Val <= x && next.Val >= x) && (next != aNode) {
		cur = next
		next = next.Next
		if cur.Val >= bigest.Val {
			bigest = cur
		}
	}

	if cur.Val <= x && next.Val >= x {
		cur.Next = insertNode
		insertNode.Next = next
	} else {
		insertNode.Next = bigest.Next
		bigest.Next = insertNode
	}
	return aNode
}
