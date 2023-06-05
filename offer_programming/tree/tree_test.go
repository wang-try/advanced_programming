package tree

import (
	"fmt"
	"testing"
)

var root = &TreeNode{
	Val: 1,
	Left: &TreeNode{
		Val: 2,
		Left: &TreeNode{
			Val:   4,
			Left:  nil,
			Right: nil,
		},
		Right: &TreeNode{
			Val:   5,
			Left:  nil,
			Right: nil,
		},
	},
	Right: &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val:  6,
			Left: nil,
			Right: &TreeNode{
				Val:   8,
				Left:  nil,
				Right: nil,
			},
		},
		Right: &TreeNode{
			Val:   7,
			Left:  nil,
			Right: nil,
		},
	},
}

func TestPreTree(t *testing.T) {
	PreOrderTreeRec(root)
	fmt.Println(PreOrderTreeIteration(root))
}

func TestMidTree(t *testing.T) {
	MidOrderTreeRec(root)
	fmt.Println(MidOrderTreeIteration(root))
}

func TestPostOrderTreeRec(t *testing.T) {
	PostOrderTreeRec(root)
	fmt.Println(PostOrderTreeIteration(root))
}
