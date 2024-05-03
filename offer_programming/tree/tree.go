package tree

import (
	"fmt"
	"strconv"
	"strings"
)

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
		//左子树入栈
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		//当前节点右子树不为空，且没有遍历过
		if cur.Right != nil && cur.Right != prev {
			cur = cur.Right
		} else {
			//当期右子树为空，或者遍历过，次节点出栈，并且打印
			stack = stack[:len(stack)-1]
			ret = append(ret, cur.Val)
			prev = cur
			cur = nil
		}
	}
	return ret
}

//二叉树剪枝
/*
一棵二叉树的所有节点的值要么是0要么是1，请剪除该二叉树中所有节点的值全都是0的子树。例如，在剪除图8.2（a）中二叉树中所有节点值都为0的子树之后的结果如图8.2（b）所示。
*/

func PruneTree(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	root.Left = PruneTree(root.Left)
	root.Right = PruneTree(root.Right)
	if root.Left == nil && root.Right == nil && root.Val == 0 {
		return nil
	}
	return root
}

//序列化和反序列化二叉树
/*
请设计一个算法将二叉树序列化成一个字符串，并能将该字符串反序列化出原来二叉树的算法。
*/

type Codec struct {
}

func Constructor() Codec {
	return Codec{}
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	if root == nil {
		return "#"
	}
	left := this.serialize(root.Left)
	right := this.serialize(root.Right)
	return strconv.Itoa(root.Val) + "," + left + "," + right
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	nodeStrs := strings.Split(data, ",")
	index := 0
	return dfs(nodeStrs, &index)
}

func dfs(strs []string, index *int) *TreeNode {
	str := strs[*index]
	*index++
	if str == "#" {
		return nil
	}
	val, _ := strconv.Atoi(str)
	root := &TreeNode{
		Val:   val,
		Left:  nil,
		Right: nil,
	}
	root.Left = dfs(strs, index)
	root.Right = dfs(strs, index)
	return root
}

//从根节点到叶节点的路径数字之和
/*
在一棵二叉树中所有节点都在0～9的范围之内，从根节点到叶节点的路径表示一个数字。求二叉树中所有路径表示的数字之和。
例如，图8.4的二叉树有3条从根节点到叶节点的路径，它们分别表示数字395、391和302，这3个数字之和是1088。
*/

func sumNumbers(root *TreeNode) int {
	return dfsSumNumbers(root, 0)
}

func dfsSumNumbers(root *TreeNode, path int) int {
	if root == nil {
		return 0
	}
	path = path*10 + root.Val
	if root.Left == nil && root.Right == nil {
		return path
	}
	return dfsSumNumbers(root.Left, path) + dfsSumNumbers(root.Right, path)
}

//向下的路径节点值之和
/*
给定一棵二叉树和一个值sum，求二叉树中节点值之和等于sum的路径的数目。路径的定义为二叉树中顺着指向子节点的指针向下移动所经过的节点，但不一定从根节点开始，也不一定到叶节点结束。
例如，在如图8.5所示中的二叉树中有两条路径的节点值之和等于8，其中，第1条路径从节点5开始经过节点2到达节点1，第2条路径从节点2开始到节点6。
*/
func pathSum(root *TreeNode, targetSum int) int {
	amap := make(map[int]int, 0)
	amap[0] = 1
	return dfsPathSum(root, targetSum, amap, 0)
}

// 前缀和
func dfsPathSum(root *TreeNode, targetSum int, amap map[int]int, path int) int {
	if root == nil {
		return 0
	}
	path += root.Val
	//?????
	count := amap[path-targetSum]
	amap[path] += 1
	count += dfsPathSum(root.Left, targetSum, amap, path)
	count += dfsPathSum(root.Right, targetSum, amap, path)
	//函数结束时，程序将回到节点的父节点，所以要在函数结束之前将当前节点从路径中删除
	amap[path] -= 1
	return count
}

//节点值之和最大的路径
/*
在二叉树中将路径定义为顺着节点之间的连接从任意一个节点开始到达任意一个节点所经过的所有节点。路径中至少包含一个节点，不一定经过二叉树的根节点，也不一定经过叶节点。
给定非空的一棵二叉树，请求出二叉树所有路径上节点值之和的最大值。例如，在如图8.6所示的二叉树中，从节点15开始经过节点20到达节点7的路径的节点值之和为42，是节点值之和最大的路径
*/

func maxPathSum(root *TreeNode) int {
	res := root.Val
	var dfs func(t *TreeNode) int
	dfs = func(t *TreeNode) int {
		if t == nil {
			return 0
		}
		lm := max(dfs(t.Left), 0)
		rm := max(dfs(t.Right), 0)
		res = max(res, lm+rm+t.Val)
		return max(lm, rm) + t.Val
	}
	dfs(root)
	return res
}
func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

//展平二叉搜索树
/*
给定一棵二叉搜索树，请调整节点的指针使每个节点都没有左子节点。调整之后的树看起来像一个链表，但仍然是二叉搜索树。例如，把图8.8（a）中的二叉搜索树按照这个规则展平之后的结果如图8.8（b）所示。
*/
func increasingBST(root *TreeNode) *TreeNode {
	var stack []*TreeNode
	flattenRoot := new(TreeNode)
	curNode := flattenRoot
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		curNode.Right = &TreeNode{
			Val:   cur.Val,
			Left:  nil,
			Right: nil,
		}
		curNode = curNode.Right
		cur = cur.Right
	}
	return flattenRoot.Right
}

func increasingBSTV2(root *TreeNode) *TreeNode {
	var stack []*TreeNode
	cur := root
	var prev *TreeNode
	var first *TreeNode
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if prev != nil {
			prev.Right = cur
		} else {
			first = cur
		}
		prev = cur
		cur.Left = nil
		cur = cur.Right
	}
	return first
}

//二叉搜索树的下一个节点
/*
给定一棵二叉搜索树和它的一个节点p，请找出按中序遍历的顺序该节点p的下一个节点。假设二叉搜索树中节点的值都是唯一的。例如，在图8.9的二叉搜索树中，节点8的下一个节点是节点9，节点11的下一个节点是null。
*/

func inorderSuccessor(root *TreeNode, p *TreeNode) *TreeNode {
	var stack []*TreeNode
	cur := root
	isFindP := false
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		if isFindP {
			return cur
		}
		if cur == p {
			isFindP = true
		}
		stack = stack[:len(stack)-1]
		cur = cur.Right
	}
	return nil

}

func inorderSuccessorV2(root *TreeNode, p *TreeNode) *TreeNode {
	var ret *TreeNode
	cur := root
	for cur != nil {
		if cur.Val > p.Val {
			ret = cur
			cur = cur.Left
		} else {
			cur = cur.Right
		}
	}
	return ret
}

//所有大于或等于节点的值之和
/*
给定一棵二叉搜索树，请将它的每个节点的值替换成树中大于或等于该节点值的所有节点值之和。假设二叉搜索树中节点的值唯一。
例如，输入如图8.10（a）所示的二叉搜索树，由于有两个节点的值大于或等于6（即节点6和节点7），因此值为6节点的值替换成13，其他节点的值的替换过程与此类似，所有节点的值替换之后的结果如图8.10（b）所示
*/
func convertBST(root *TreeNode) *TreeNode {
	var ret []*TreeNode
	var stack []*TreeNode
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		ret = append(ret, cur)
		stack = stack[:len(stack)-1]
		cur = cur.Right
	}
	sum := 0
	for _, node := range ret {
		sum += node.Val
	}
	preSum := 0
	for i := 0; i < len(ret); i++ {
		tmp := ret[i].Val
		ret[i].Val = sum - preSum
		preSum += tmp
	}
	return root
}

func convertBSTV2(root *TreeNode) *TreeNode {
	var stack []*TreeNode
	cur := root
	sum := 0
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Right
		}
		cur = stack[len(stack)-1]
		sum += cur.Val
		cur.Val = sum
		stack = stack[:len(stack)-1]
		cur = cur.Left
	}
	return root
}

//二叉搜索树迭代器
/*
：请实现二叉搜索树的迭代器BSTIterator，它主要有如下3个函数。
● 构造函数：输入二叉搜索树的根节点初始化该迭代器。
● 函数next：返回二叉搜索树中下一个最小的节点的值。
● 函数hasNext：返回二叉搜索树是否还有下一个节点
*/

type BSTIterator struct {
	cur   *TreeNode
	stack []*TreeNode
}

func ConstructorBST(root *TreeNode) BSTIterator {
	return BSTIterator{
		cur:   root,
		stack: nil,
	}
}

func (this *BSTIterator) Next() int {
	for this.cur != nil {
		this.stack = append(this.stack, this.cur)
		this.cur = this.cur.Left
	}
	num := this.stack[len(this.stack)-1].Val
	this.stack = this.stack[:len(this.stack)-1]
	this.cur = this.cur.Right
	return num
}

func (this *BSTIterator) HasNext() bool {
	return this.cur != nil || len(this.stack) > 0
}

//二叉搜索树中两个节点的值之和
/*
给定一棵二叉搜索树和一个值k，请判断该二叉搜索树中是否存在值之和等于k的两个节点。假设二叉搜索树中节点的值均唯一。
例如，在如图8.12所示的二叉搜索树中，存在值之和等于12的两个节点（节点5和节点7），但不存在值之和为22的两个节点
*/
func findTarget(root *TreeNode, k int) bool {
	val2isExist := make(map[int]struct{})
	cur := root
	var stack []*TreeNode
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		if _, ok := val2isExist[k-cur.Val]; ok {
			return true
		}
		val2isExist[cur.Val] = struct{}{}
		stack = stack[:len(stack)-1]
		cur = cur.Right
	}
	return false
}

//值和下标之差都在给定的范围内
/*
给定一个整数数组nums和两个正数k、t，请判断是否存在两个不同的下标i和j满足i和j之差的绝对值不大于给定的k，并且两个数值nums[i]和nums[j]的差的绝对值不大于给定的t。
*/
//???????完全看不懂
func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	m := make(map[int]int)
	for i, num := range nums {
		id := getID(num, t+1)
		if _, ok := m[id]; ok {
			return true
		}
		if n, ok := m[id-1]; ok && abs(num-n) <= t {
			return true
		}
		if n, ok := m[id+1]; ok && abs(num-n) <= t {
			return true
		}
		m[id] = num
		if i >= k {
			delete(m, getID(nums[i-k], t+1))
		}
	}
	return false
}

func getID(x int, t int) int {
	if x >= 0 {
		return x / t
	}
	return (x+1)/t - 1
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
