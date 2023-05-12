package list

import (
	"fmt"
	"testing"
)

func TestReOrderList(t *testing.T) {
	head := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 3,
				Next: &ListNode{
					Val: 4,
					Next: &ListNode{
						Val:  5,
						Next: nil,
					},
				},
			},
		},
	}
	ReOrderList(head)
}

func TestIsPalindrome(t *testing.T) {
	head := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val:  2,
			Next: nil,
		},
	}

	fmt.Println(IsPalindrome(head))
}

func TestFlatten(t *testing.T) {
	list := make([]*Node, 10)
	for i := 1; i <= 9; i++ {
		list[i] = &Node{
			Val:   i,
			Prev:  nil,
			Next:  nil,
			Child: nil,
		}
	}
	head := list[1]
	list[1].Next = list[2]
	list[2].Prev = list[1]
	list[2].Next = list[3]
	list[3].Prev = list[2]
	list[3].Next = list[4]
	list[4].Prev = list[3]

	list[2].Child = list[5]
	list[5].Next = list[6]
	list[6].Prev = list[5]
	list[6].Next = list[7]
	list[7].Prev = list[6]

	list[6].Child = list[8]
	list[8].Next = list[9]
	list[9].Next = list[8]

	FlattenPrint(head)
}
