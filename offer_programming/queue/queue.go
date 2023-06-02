package queue

//滑动窗口的平均值
/*
请实现如下类型MovingAverage，计算滑动窗口中所有数字的平均值，该类型构造函数的参数确定滑动窗口的大小，每次调用成员函数next时都会在滑动窗口中添加一个整数，并返回滑动窗口中所有数字的平均值
*/
//TODO 可以用数组也可以用链表，下面的链表使用方式错误，新加的节点应该防止链表尾部，驱逐头节点。
type MovingAverage struct {
	Head *MovingNode
	Tail *MovingNode
	Sum  int
	Size int
	Lth  int
}

type MovingNode struct {
	Val      int
	NextNode *MovingNode
}

/** Initialize your data structure here. */
func Constructor(size int) MovingAverage {
	return MovingAverage{
		Head: nil,
		Tail: nil,
		Sum:  0,
		Size: size,
		Lth:  0,
	}
}

func (this *MovingAverage) Next(val int) float64 {
	node := &MovingNode{
		Val:      val,
		NextNode: nil,
	}
	if this.Lth < this.Size {
		if this.Lth == 0 {
			this.Head, this.Tail = node, node
		} else {
			node.NextNode = this.Head
			this.Head = node
		}
		this.Sum += val
		this.Lth++
		return float64(this.Sum) / float64(this.Lth)
	} else {
		if this.Head == this.Tail {
			this.Sum -= this.Tail.Val
			node.NextNode = this.Head
			this.Head = node
			this.Tail = node
			this.Sum += val
			return float64(this.Sum) / float64(this.Lth)
		}
		cur := this.Head
		for cur != nil && cur.NextNode != this.Tail {
			cur = cur.NextNode
		}
		this.Sum -= this.Tail.Val
		this.Tail = cur
		node.NextNode = this.Head
		this.Head = node
		this.Sum += val
		return float64(this.Sum) / float64(this.Lth)
	}
}

//最近请求次数
/*
请实现如下类型RecentCounter，它是统计过去3000ms内的请求次数的计数器。该类型的构造函数RecentCounter初始化计数器，请求数初始化为0；
函数ping（int t）在时间t添加一个新请求（t表示以毫秒为单位的时间），并返回过去3000ms内（时间范围为[t-3000，t]）发生的所有请求数。假设每次调用函数ping的参数t都比之前调用的参数值大
*/

type RecentCounter struct {
	queue []int
}

func ConstructorRecentCounter() RecentCounter {
	return RecentCounter{}
}

func (this *RecentCounter) Ping(t int) int {
	for len(this.queue) > 0 && this.queue[0] < t-3000 {
		this.queue = this.queue[1:]
	}
	this.queue = append(this.queue, t)
	return len(this.queue)
}

//在完全二叉树中添加节点
/*
题目：在完全二叉树中，除最后一层之外其他层的节点都是满的（第n层有2n-1个节点）。最后一层的节点可能不满，该层所有的节点尽可能向左边靠拢。
例如，图7.3中的4棵二叉树均为完全二叉树。实现数据结构CBTInserter有如下3种方法。● 构造函数CBTInserter（TreeNode root），用一棵完全二叉树的根节点初始化该数据结构。
● 函数insert（int v）在完全二叉树中添加一个值为v的节点，并返回被插入节点的父节点。例如，在如图7.3（a）所示的完全二叉树中添加一个值为7的节点之后，
二叉树如图7.3（b）所示，并返回节点3。在如图7.3（b）所示的完全二叉树中添加一个值为8的节点之后，二叉树如图7.3（c）所示，并返回节点4。
在如图7.3（c）所示的完全二叉树中添加节点9会得到如图7.3（d）所示的二叉树并返回节点4。● 函数get_root()返回完全二叉树的根节点。
*/

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type CBTInserter struct {
	Root            *TreeNode
	NotFullNodeList []*TreeNode
}

func ConstructorTree(root *TreeNode) CBTInserter {
	if root == nil {
		return CBTInserter{nil, []*TreeNode{}}
	}
	var leaves []*TreeNode
	leaves = append(leaves, root)
	for len(leaves) > 0 {
		node := leaves[0]
		if node.Left != nil {
			leaves = append(leaves, node.Left)
		} else {
			break
		}
		if node.Right != nil {
			leaves = append(leaves, node.Right)
		} else {
			break
		}
		leaves = leaves[1:]
	}
	return CBTInserter{root, leaves}
}

func (this *CBTInserter) Insert(v int) int {
	node := &TreeNode{
		Val:   v,
		Left:  nil,
		Right: nil,
	}
	if len(this.NotFullNodeList) > 0 {
		leftNode := this.NotFullNodeList[0]
		if leftNode.Left == nil {
			leftNode.Left = node
		} else if leftNode.Right == nil {
			leftNode.Right = node
			this.NotFullNodeList = this.NotFullNodeList[1:]
		}
		this.NotFullNodeList = append(this.NotFullNodeList, node)
		return leftNode.Val
	}
	return -1
}

func (this *CBTInserter) Get_root() *TreeNode {
	return this.Root
}

//二叉树中每层的最大值
/*
输入一棵二叉树，请找出二叉树中每层的最大值。例如，输入图7.4中的二叉树，返回各层节点的最大值[3，4，9]。
*/

func LargestValues(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var ret []int
	var queue []*TreeNode
	queue = append(queue, root)
	for len(queue) > 0 {
		lth := len(queue)
		max := queue[0].Val
		for i := 0; i < lth; i++ {
			if queue[i].Val > max {
				max = queue[i].Val
			}
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		ret = append(ret, max)
		queue = queue[lth:]
	}
	return ret
}

//二叉树最低层最左边的值
/*
如何在一棵二叉树中找出它最低层最左边节点的值？假设二叉树中最少有一个节点。例如，在如图7.5所示的二叉树中最低层最左边一个节点的值是5。
*/
func FindBottomLeftValue(root *TreeNode) int {
	if root == nil {
		return -1
	}
	var queue []*TreeNode
	var left int
	queue = append(queue, root)
	for len(queue) > 0 {
		lth := len(queue)
		left = queue[0].Val
		for i := 0; i < lth; i++ {
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[lth:]
	}
	return left
}

func FindBottomLeftValueV2(root *TreeNode) int {
	height := 0
	ans := root.Val
	var dfs func(*TreeNode, int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil {
			if height < depth {
				height = depth
				ans = node.Val
			}
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 0)
	return ans
}

func FindBottomLeftValueV3(root *TreeNode) int {
	height := 0
	left := root.Val
	dfs(root, 0, &height, &left)
	return left
}

func dfs(node *TreeNode, depth int, height, left *int) {
	if node == nil {
		return
	}
	if node.Left == nil && node.Right == nil {
		if *height < depth {
			*height = depth
			*left = node.Val
		}
	}
	dfs(node.Left, depth+1, height, left)
	dfs(node.Right, depth+1, height, left)
}

//二叉树的右侧视图
/*
给定一棵二叉树，如果站在该二叉树的右侧，那么从上到下看到的节点构成二叉树的右侧视图。例如，图7.6中二叉树的右侧视图包含节点8、节点10和节点7。请写一个函数返回二叉树的右侧视图节点的值。
*/

func RightSideView(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var queue []*TreeNode
	var ret []int
	queue = append(queue, root)
	for len(queue) > 0 {
		lth := len(queue)
		ret = append(ret, queue[lth-1].Val)
		for i := 0; i < lth; i++ {
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[lth:]
	}
	return ret
}

func RightSideViewV2(root *TreeNode) []int {
	ans := []int{}
	if root == nil {
		return ans
	}
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}

		if len(ans) == depth {
			ans = append(ans, node.Val)
		}
		dfs(node.Right, depth+1)
		dfs(node.Left, depth+1)
	}
	dfs(root, 0)
	return ans
}
