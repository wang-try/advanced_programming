package main

import (
	"fmt"
	"testing"
)

func TestLowestCommonAncestor(t *testing.T) {
	root := &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val: 5,
			Left: &TreeNode{
				Val:   6,
				Left:  nil,
				Right: nil,
			},
			Right: &TreeNode{
				Val: 2,
				Left: &TreeNode{
					Val:   7,
					Left:  nil,
					Right: nil,
				},
				Right: &TreeNode{
					Val:   4,
					Left:  nil,
					Right: nil,
				},
			},
		},
		Right: &TreeNode{
			Val: 1,
			Left: &TreeNode{
				Val:   0,
				Left:  nil,
				Right: nil,
			},
			Right: &TreeNode{
				Val:   8,
				Left:  nil,
				Right: nil,
			},
		},
	}
	p := root.Left.Left
	q := root.Left.Right.Right
	res := LowestCommonAncestor(root, p, q)
	fmt.Println(res.Val)
}

func TestLongestCommonPrefix(t *testing.T) {
	str := LongestCommonPrefix([]string{"flower", "flow", "flight"})
	fmt.Println(str)

}
