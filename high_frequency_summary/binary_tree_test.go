package high_frequency_summary

import (
	"fmt"
	"testing"
)

func TestVerifyTreeOrder(t *testing.T) {
	fmt.Println(verifyTreeOrder([]int{5, 2, -17, -11, 25, 76, 62, 98, 92, 61}))
}

func TestPathSumIII(t *testing.T) {
	node := &TreeNode{
		Left: &TreeNode{
			Left: &TreeNode{
				Left: &TreeNode{
					Left:  nil,
					Right: nil,
					Val:   3,
				},
				Right: &TreeNode{
					Left:  nil,
					Right: nil,
					Val:   -2,
				},
				Val: 3,
			},
			Right: &TreeNode{
				Left: nil,
				Right: &TreeNode{
					Left:  nil,
					Right: nil,
					Val:   1,
				},
				Val: 2,
			},
			Val: 5,
		},
		Right: &TreeNode{
			Left: nil,
			Right: &TreeNode{
				Left:  nil,
				Right: nil,
				Val:   11,
			},
			Val: -3,
		},
		Val: 10,
	}
	fmt.Println(pathSumIII(node, 8))
}

func TestTreeToDoublyList(t *testing.T) {
	root := &Node{
		Left: &Node{
			Left:  nil,
			Right: nil,
			Val:   1,
		},
		Right: &Node{
			Left: &Node{
				Left:  nil,
				Right: nil,
				Val:   4,
			},
			Right: &Node{
				Left:  nil,
				Right: nil,
				Val:   5,
			},
			Val: 3,
		},
		Val: 2,
	}
	treeToDoublyList(root)
}

func TestDeleteNode(t *testing.T) {
	root := &TreeNode{
		Left: &TreeNode{
			Left: &TreeNode{
				Left:  nil,
				Right: nil,
				Val:   2,
			},
			Right: &TreeNode{
				Left:  nil,
				Right: nil,
				Val:   4,
			},
			Val: 3,
		},
		Right: &TreeNode{
			Left: nil,
			Right: &TreeNode{
				Left:  nil,
				Right: nil,
				Val:   7,
			},
			Val: 6,
		},
		Val: 5,
	}
	fmt.Println(deleteNode(root, 3))
}

func TestRecoverTree(t *testing.T) {
	root := &TreeNode{
		Left: &TreeNode{
			Left: &TreeNode{
				Left: nil,
				Right: &TreeNode{
					Left:  nil,
					Right: nil,
					Val:   2,
				},
				Val: 3,
			},
			Right: nil,
			Val:   0,
		},
		Right: nil,
		Val:   1,
	}
	recoverTree(root)

}
