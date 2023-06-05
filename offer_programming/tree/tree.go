package tree

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func PreOrderTreeRec(root *TreeNode) {
	if root != nil {
		fmt.Println(root.Val)
		PreOrderTreeRec(root.Left)
		PreOrderTreeRec(root.Right)
	}
}

func PreOrderTreeIteration(root *TreeNode) []int {
	cur := root
	var ret []int
	var stack []*TreeNode
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			ret = append(ret, cur.Val)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		cur = cur.Right
	}
	return ret
}

func MidOrderTreeRec(root *TreeNode) {
	if root != nil {
		MidOrderTreeRec(root.Left)
		fmt.Println(root.Val)
		MidOrderTreeRec(root.Right)
	}
}

func MidOrderTreeIteration(root *TreeNode) []int {
	cur := root
	var ret []int
	var stack []*TreeNode
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		ret = append(ret, cur.Val)
		cur = cur.Right
	}
	return ret
}

func PostOrderTreeRec(root *TreeNode) {
	if root != nil {
		PostOrderTreeRec(root.Left)
		PostOrderTreeRec(root.Right)
		fmt.Println(root.Val)
	}
}

func PostOrderTreeIteration(root *TreeNode) []int {
	var ret []int
	var stack []*TreeNode
	cur := root
	var prev *TreeNode = nil
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		if cur.Right != nil && cur.Right != prev {
			cur = cur.Right
		} else {
			stack = stack[:len(stack)-1]
			ret = append(ret, cur.Val)
			prev = cur
			cur = nil
		}
	}
	return ret
}
