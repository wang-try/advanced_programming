package high_frequency_summary

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

//前序遍历

type Tree struct {
	Left  *Tree
	Right *Tree
	Val   int
}

type TreeNode struct {
	Left  *TreeNode
	Right *TreeNode
	Val   int
}

func preOrder(root *Tree) {
	if root != nil {
		fmt.Println(root.Val)
		preOrder(root.Left)
		preOrder(root.Right)
	}

}

func preOrderIter(root *Tree) []int {
	var ans []int
	var stack []*Tree
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			ans = append(ans, cur.Val)
			cur = cur.Left
		}
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		cur = node.Right
	}
	return ans
}

func midOrder(root *Tree) {
	if root != nil {
		midOrder(root.Left)
		fmt.Println(root.Val)
		midOrder(root.Right)
	}

}

func midOrderIter(root *Tree) []int {
	var ans []int
	cur := root
	var stack []*Tree
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		ans = append(ans, cur.Val)
		stack = stack[:len(stack)-1]
		cur = cur.Right
	}
	return ans
}

func postOrder(root *Tree) {
	if root != nil {
		postOrder(root.Left)
		postOrder(root.Right)
		fmt.Println(root.Val)
	}
}

func PostOrderIter(root *Tree) []int {
	var ans []int
	cur := root
	var stack []*Tree
	var prev *Tree = nil
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		//当前节点右子树不为空，且没有遍历过
		if cur.Right != nil && cur.Right != prev {
			cur = cur.Right
		} else {
			//当前右子树为空，或者遍历过，此节点出栈，并且打印
			stack = stack[:len(stack)-1]
			ans = append(ans, cur.Val)
			prev = cur
			cur = nil
		}
	}
	return ans
}

// leetcodeLCR 152. 验证二叉搜索树的后序遍历序列
func verifyTreeOrder(postorder []int) bool {
	var help func(start, end int) bool
	help = func(start, end int) bool {
		if start >= end {
			return true
		}
		//找到第一个比根节点大的元素
		pivot := postorder[end]
		splitIndex := end
		for i := start; i < end; i++ {
			if postorder[i] > pivot {
				splitIndex = i
				break
			}
		}

		//start到splitindex-1是左子树，已满足左子树性质
		//splitindex-1到end-1是右子树
		for i := splitIndex; i < end; i++ {
			if postorder[i] < pivot {
				return false
			}
		}
		return help(start, splitIndex-1) && help(splitIndex, end-1)

	}

	return help(0, len(postorder)-1)
}

// leetcode105. 从前序与中序遍历序列构造二叉树
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := preorder[0]
	index := findRootIndex(root, inorder)
	if index == -1 {
		return nil
	}
	lLth := len(inorder[:index])
	return &TreeNode{
		Left:  buildTree(preorder[1:1+lLth], inorder[:index]),
		Right: buildTree(preorder[1+lLth:], inorder[index+1:]),
		Val:   root,
	}
}

func findRootIndex(root int, inorder []int) int {
	for i, val := range inorder {
		if root == val {
			return i
		}
	}
	return -1
}

// leetcode572. 另一棵树的子树
func isSubtree(s *TreeNode, t *TreeNode) bool {
	if s == nil {
		return false
	}
	return check(s, t) || isSubtree(s.Left, t) || isSubtree(s.Right, t)
}

func check(a, b *TreeNode) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Val == b.Val {
		return check(a.Left, b.Left) && check(a.Right, b.Right)
	}
	return false
}

func isSubtreeV2(root *TreeNode, subRoot *TreeNode) bool {
	//暴力匹配算法时间复杂度为O(∣s∣×∣t∣)
	//优化后的算法KMP O(∣s∣+∣t∣)
	//采用树哈希的方法为O(argπ(max{∣s∣,∣t∣})),略慢于线性复杂度
	//本次解法采用树哈希方法
	//1、求解对应原树的哈希
	//2、求解对应的子树的哈希
	//3、任意树搜索算法进行遍历
	//哈希求解采用官方题解，公式为:H(x) = val + 31(任意权值)*H(l)*p(sl) + 179(任意权值,与左子树不同即可)*H(r)*p(sr)
	getPrime()
	var ms, mr *(map[*TreeNode]status)
	mss, mrr := make(map[*TreeNode]status), make(map[*TreeNode]status)
	ms, mr = &mss, &mrr
	dfs(root, mr)
	dfs(subRoot, ms)
	hash := (*ms)[subRoot].f
	for _, v := range *mr {
		if v.f == hash {
			return true
		}
	}
	return false
}

type status struct {
	//f 记录 对应的哈希值， s记录以该node为根节点的树的大小
	f, s int
}

// 小tips：在go中的map不是一个并发安全的结构，所以，并不能修改他在结构体中的值，两种方案，一个是传结构体，一个是存指针
// 中序遍历dfs
func dfs(node *TreeNode, m *map[*TreeNode]status) {
	(*m)[node] = status{node.Val, 1}
	if node.Left == nil && node.Right == nil {
		return
	}
	if node.Left != nil {
		dfs(node.Left, m)
		//左子树遍历后的大小累加
		tmp := (*m)[node]
		tmp.s += (*m)[node.Left].s
		tmp.f = ((*m)[node].f + (31*(*m)[node.Left].f*prime[(*m)[node.Left].s])%MOD) % MOD
		(*m)[node] = tmp

	}
	if node.Right != nil {
		dfs(node.Right, m)
		tmp := (*m)[node]
		tmp.s += (*m)[node.Right].s
		tmp.f = ((*m)[node].f + (179*(*m)[node.Right].f*prime[(*m)[node.Right].s])%MOD) % MOD
		(*m)[node] = tmp
	}
}

var MOD = int(^uint(0) >> 1)
var MAX_N = 30000
var vi [30000]bool
var prime [30000]int
var cnt int

// 欧拉筛
func getPrime() {
	vi[0], vi[1] = true, true
	cnt = 0
	for i := 2; i < MAX_N; i++ {
		if !vi[i] {
			prime[cnt] = i
			cnt++
		}

		for j := 0; j < cnt && prime[j]*i < MAX_N; j++ {
			vi[i*prime[j]] = true
			//保证只会被最小质因数访问
			if i%prime[j] == 0 {
				break
			}
		}
	}
}

// leetcode104. 二叉树的最大深度
func maxDepth(root *TreeNode) int {
	ans := 0
	var dfs func(node *TreeNode, depth int)

	dfs = func(node *TreeNode, depth int) {
		if depth > ans {
			ans = depth
		}
		if node == nil {
			return
		}
		depth++
		dfs(node.Right, depth)
		dfs(node.Left, depth)
	}

	dfs(root, 0)
	return ans
}

// leetcode111. 二叉树的最小深度
func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	ans := math.MaxInt32
	var dfs func(node *TreeNode, depth int)

	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}

		if node.Left == nil && node.Right == nil {
			if depth < ans {
				ans = depth
			}
			return
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}

	dfs(root, 1)
	return ans
}

// leetcode662. 二叉树的最大宽度
// Fixme 内存超过限制
func widthOfBinaryTree(root *TreeNode) int {
	var queue []*TreeNode
	queue = append(queue, root)
	maxW := 0
	for len(queue) > 0 {
		lth := len(queue)
		start := -1
		end := 0
		for i := 0; i < lth; i++ {
			if queue[i] != nil {
				if start == -1 {
					start = i
				}
				end = i

			}
		}
		if start == -1 {
			break
		}
		for i := start; i <= end; i++ {
			if queue[i] == nil {
				queue = append(queue, nil, nil)
			} else {
				queue = append(queue, queue[i].Left, queue[i].Right)
			}
		}

		maxW = max(maxW, end-start+1)
		queue = queue[lth:]
	}
	return maxW
}

type Pair struct {
	node  *TreeNode
	index int
}

func widthOfBinaryTreeV2(root *TreeNode) int {
	var queue []Pair
	queue = append(queue, Pair{
		node:  root,
		index: 1,
	})
	maxWidth := 0
	for len(queue) > 0 {
		lth := len(queue)
		maxWidth = max(maxWidth, queue[lth-1].index-queue[0].index+1)
		for i := 0; i < lth; i++ {
			if queue[i].node.Left != nil {
				queue = append(queue, Pair{
					node:  queue[i].node.Left,
					index: queue[i].index * 2,
				})
			}
			if queue[i].node.Right != nil {
				queue = append(queue, Pair{
					node:  queue[i].node.Right,
					index: queue[i].index*2 + 1,
				})
			}
		}
		queue = queue[lth:]
	}
	return maxWidth
}

// leetcode124. 二叉树中的最大路径和
func maxPathSum(root *TreeNode) int {
	res := root.Val
	var dfs func(t *TreeNode) int
	dfs = func(t *TreeNode) int {
		if t == nil {
			return 0
		}
		lm := max(dfs(t.Left), 0)
		rm := max(dfs(t.Right), 0)
		// 更新答案
		res = max(res, t.Val+lm+rm)
		//返回节点的最大贡献值
		return max(lm, rm) + t.Val
	}
	dfs(root)
	return res
}

// leetcode543. 二叉树的直径
func diameterOfBinaryTree(root *TreeNode) int {
	ans := 1
	var dfs func(t *TreeNode) int
	dfs = func(t *TreeNode) int {
		if t == nil {
			return 0
		}
		l := dfs(t.Left)
		r := dfs(t.Right)
		ans = max(ans, l+r+1)
		return max(l, r) + 1
	}
	dfs(root)
	return ans - 1
}

// leetcode110. 平衡二叉树
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return abs(height(root.Left)-height(root.Right)) <= 1 && isBalanced(root.Left) && isBalanced(root.Right)
}

func height(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(height(root.Left), height(root.Right)) + 1
}

func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}

func isBalancedV2(root *TreeNode) bool {
	return heightV2(root) >= 0
}

func heightV2(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftHeight := heightV2(root.Left)
	rightHeight := heightV2(root.Right)
	if leftHeight == -1 || rightHeight == -1 || abs(leftHeight-rightHeight) > 1 {
		return -1
	}
	return max(leftHeight, rightHeight) + 1
}

// leetcode297. 二叉树的序列化与反序列化
type Codec struct {
}

func ConstructorII() Codec {
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
	return Dfs(nodeStrs, &index)
}

func Dfs(strs []string, index *int) *TreeNode {
	str := strs[*index]
	*index++
	if str == "#" {
		return nil
	}
	val, _ := strconv.Atoi(str)
	return &TreeNode{
		Val:   val,
		Left:  Dfs(strs, index),
		Right: Dfs(strs, index),
	}
}

// leetcode226. 翻转二叉树
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	return &TreeNode{
		Val:   root.Val,
		Left:  invertTree(root.Right),
		Right: invertTree(root.Left),
	}
}

// leetcode235. 二叉搜索树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	ancestor := root
	for {
		if p.Val < ancestor.Val && q.Val < ancestor.Val {
			ancestor = ancestor.Left
		} else if p.Val > ancestor.Val && q.Val > ancestor.Val {
			ancestor = ancestor.Right
		} else {
			return ancestor
		}
	}
}

// leetcode236. 二叉树的最近公共祖先
func lowestCommonAncestorII(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root == p || root == q {
		return root
	}

	l := lowestCommonAncestor(root.Left, p, q)
	r := lowestCommonAncestor(root.Right, p, q)

	if l != nil && r != nil {
		return root
	}
	if l != nil {
		return l
	}
	return r
}

// leetcode103. 二叉树的锯齿形层序遍历
func zigzagLevelOrder(root *TreeNode) [][]int {
	var ans [][]int
	var dfs func(node *TreeNode, level int)
	dfs = func(node *TreeNode, level int) {
		if node == nil {
			return
		}
		if len(ans) < level {
			ans = append(ans, []int{})
		}
		if level&1 == 1 {
			ans[level-1] = append(ans[level-1], node.Val)
		} else {
			ans[level-1] = append([]int{node.Val}, ans[level-1]...)
		}
		dfs(node.Left, level+1)
		dfs(node.Right, level+1)
	}
	dfs(root, 1)
	return ans
}

// leetcode814. 二叉树剪枝
func pruneTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left = pruneTree(root.Left)
	root.Right = pruneTree(root.Right)
	if root.Left == nil && root.Right == nil && root.Val == 0 {
		return nil
	}
	return root
}

func rightSideView(root *TreeNode) []int {
	var ans []int
	var dfs func(node *TreeNode, level int)
	dfs = func(node *TreeNode, level int) {
		if node == nil {
			return
		}
		if len(ans) < level {
			ans = append(ans, node.Val)
		} else {
			ans[level-1] = node.Val
		}
		dfs(node.Left, level+1)
		dfs(node.Right, level+1)

	}
	dfs(root, 1)
	return ans
}

// leetcode112. 路径总和
func hasPathSum(root *TreeNode, targetSum int) bool {
	var dfs func(node *TreeNode, sum int) bool
	dfs = func(node *TreeNode, sum int) bool {
		if node == nil {
			return false
		}
		sum += node.Val
		if node.Left == nil && node.Right == nil && sum == targetSum {
			return true
		}

		return dfs(node.Left, sum) || dfs(node.Right, sum)
	}
	return dfs(root, 0)
}

// leetcode113. 路径总和 II
func pathSum(root *TreeNode, targetSum int) [][]int {
	var ans [][]int
	var dfs func(node *TreeNode, sum int, combination []int)

	dfs = func(node *TreeNode, sum int, combination []int) {
		if node == nil {
			return
		}
		sum += node.Val
		combination = append(combination, node.Val)
		if node.Left == nil && node.Right == nil && sum == targetSum {
			ans = append(ans, append([]int{}, combination...))
			return
		}
		dfs(node.Left, sum, combination)
		dfs(node.Right, sum, combination)
	}

	dfs(root, 0, []int{})
	return ans

}

// leetcode437. 路径总和 III
func pathSumIII(root *TreeNode, targetSum int) int {
	var ans int
	preSum := make(map[int]int)
	preSum[0] = 1
	var dfs func(node *TreeNode, sum int)

	dfs = func(node *TreeNode, sum int) {
		if node == nil {
			return
		}
		sum += node.Val
		ans += preSum[sum-targetSum]
		preSum[sum]++
		dfs(node.Left, sum)
		dfs(node.Right, sum)
		preSum[sum]--
	}

	dfs(root, 0)
	return ans
}

// leetcode958. 二叉树的完全性检验
func isCompleteTree(root *TreeNode) bool {
	var queue []Pair
	queue = append(queue, Pair{
		node:  root,
		index: 1,
	})
	var nodes []int
	nodes = append(nodes, 1)
	for len(queue) > 0 {
		lth := len(queue)
		for i := 0; i < lth; i++ {
			node := queue[i]
			if node.node.Left != nil {
				queue = append(queue, Pair{
					node:  node.node.Left,
					index: node.index * 2,
				})
				nodes = append(nodes, node.index*2)
			}

			if node.node.Right != nil {
				queue = append(queue, Pair{
					node:  node.node.Right,
					index: node.index*2 + 1,
				})
				nodes = append(nodes, node.index*2+1)
			}
		}
		queue = queue[lth:]
	}
	return nodes[len(nodes)-1] == len(nodes)
}

// 牛的
func isCompleteTreeV2(root *TreeNode) bool {
	//标记层序遍历的过程中是否有遇到空节点
	empty := false
	//辅助层序遍历的队列
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		//取一个节点
		node := queue[0]
		//出队列
		queue = queue[1:]
		if node == nil {
			//遇到空节点，把标记改成true
			empty = true
		} else {
			//此时是遍历非空节点，在非空节点之前出现了空节点，就说明并不是满二叉树
			if empty == true {
				return false
			}
			//添加左右子节点，无论是否为空
			queue = append(queue, node.Left)
			queue = append(queue, node.Right)
		}
	}
	return true
}

// leetcode108. 将有序数组转换为二叉搜索树
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := len(nums) / 2
	return &TreeNode{
		Left:  sortedArrayToBST(nums[:mid]),
		Right: sortedArrayToBST(nums[mid+1:]),
		Val:   nums[mid],
	}
}

// leetcode101. 对称二叉树
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	var dfs func(left, right *TreeNode) bool

	dfs = func(left, right *TreeNode) bool {
		if left == nil && right == nil {
			return true
		}

		if left != nil && right != nil {
			if left.Val == right.Val {
				return dfs(left.Left, right.Right) && dfs(left.Right, right.Left)
			}
			return false
		}
		return false
	}
	return dfs(root.Left, root.Right)
}

// leetcodeLCR 174. 寻找二叉搜索树中的目标节点
func findTargetNode(root *TreeNode, cnt int) int {
	cur := root
	var stack []*TreeNode
	curCnt := 0
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Right
		}
		node := stack[len(stack)-1]
		curCnt++
		if curCnt == cnt {
			return node.Val
		}
		stack = stack[:len(stack)-1]
		cur = node.Left
	}
	return -1
}

func findTargetNodeV2(root *TreeNode, cnt int) int {
	i := 0 // i从0到k
	var dfs func(n *TreeNode)
	var ans int
	dfs = func(n *TreeNode) {
		if n == nil {
			return
		}
		dfs(n.Right)
		i++
		if i == cnt {
			ans = n.Val
			return
		}
		dfs(n.Left)
	}
	dfs(root)
	return ans
}

// leetcodeLCR 153. 二叉树中和为目标值的路径
func pathTarget(root *TreeNode, target int) [][]int {
	var ans [][]int
	var dfs func(node *TreeNode, sum int, combinations []int)

	dfs = func(node *TreeNode, sum int, combinations []int) {
		if node == nil {
			return
		}
		sum += node.Val
		combinations = append(combinations, node.Val)
		if node.Left == nil && node.Right == nil && sum == target {
			ans = append(ans, append([]int{}, combinations...))
			return
		}
		dfs(node.Left, sum, combinations)
		dfs(node.Right, sum, combinations)
	}
	dfs(root, 0, []int{})
	return ans
}

// leetcodeLCR 151. 彩灯装饰记录 III
func decorateRecord(root *TreeNode) [][]int {
	var ans [][]int
	var dfs func(node *TreeNode, level int)
	dfs = func(node *TreeNode, level int) {
		if node == nil {
			return
		}
		if len(ans) < level {
			ans = append(ans, []int{})
		}
		if level&1 == 1 {
			ans[level-1] = append(ans[level-1], node.Val)
		} else {
			ans[level-1] = append([]int{node.Val}, ans[level-1]...)
		}
		dfs(node.Left, level+1)
		dfs(node.Right, level+1)
	}
	dfs(root, 1)
	return ans
}

// leetcode100. 相同的树
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p != nil && q != nil {
		if p.Val != q.Val {
			return false
		}
		return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
	}
	return false
}

// leetcode515. 在每个树行中找最大值
func largestValues(root *TreeNode) []int {
	var ans []int
	var dfs func(node *TreeNode, level int)
	dfs = func(node *TreeNode, level int) {
		if node == nil {
			return
		}
		if len(ans) < level {
			ans = append(ans, node.Val)
		} else {
			if node.Val > ans[level-1] {
				ans[level-1] = node.Val
			}
		}
		dfs(node.Left, level+1)
		dfs(node.Right, level+1)
	}
	dfs(root, 1)
	return ans
}

func kthSmallest(root *TreeNode, k int) int {
	var ans int
	cnt := 0
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		cnt++
		if cnt == k {
			ans = node.Val
			return
		}
		dfs(node.Right)
	}
	dfs(root)
	return ans
}

type Node struct {
	Left  *Node
	Right *Node
	Val   int
}

func treeToDoublyList(root *Node) *Node {
	if root == nil {
		return root
	}
	dummy := new(Node)
	dummyHead := dummy
	cur := root
	var stack []*Node
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		node := stack[len(stack)-1]
		dummy.Right = node
		node.Left = dummy
		dummy = node
		stack = stack[:len(stack)-1]
		cur = node.Right
	}
	head := dummyHead.Right
	head.Left = dummy
	dummy.Right = head
	return head
}

func treeToDoublyListV2(root *Node) *Node {
	if root == nil {
		return nil
	}
	first := &Node{}
	last := first
	var dfs func(*Node)
	dfs = func(node *Node) {
		if node == nil {
			return
		}
		dfs(node.Left)
		last.Right = node
		node.Left = last
		last = last.Right
		dfs(node.Right)
	}
	dfs(root)
	//构造环
	head := first.Right
	head.Left = last
	last.Right = head
	return head
}

// leetcode129. 求根节点到叶节点数字之和
func sumNumbers(root *TreeNode) int {
	var sum int
	var dfs func(node *TreeNode, num int)
	dfs = func(node *TreeNode, num int) {
		if node == nil {
			return
		}
		num = num*10 + node.Val
		if node.Left == nil && node.Right == nil {
			sum += num
			return
		}
		dfs(node.Left, num)
		dfs(node.Right, num)
	}
	dfs(root, 0)
	return sum
}

// leetcodeLCR 124. 推理二叉树
func deduceTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := preorder[0]
	index := findI(root, inorder)
	if index == -1 {
		return nil
	}
	lth := len(inorder[:index])
	return &TreeNode{
		Left:  deduceTree(preorder[1:lth+1], inorder[:index]),
		Right: deduceTree(preorder[lth+1:], inorder[index+1:]),
		Val:   root,
	}

}

func findI(target int, inorder []int) int {
	for i, val := range inorder {
		if target == val {
			return i
		}
	}
	return -1
}

// leetcode257. 二叉树的所有路径
func binaryTreePaths(root *TreeNode) []string {
	var ans []string
	var dfs func(node *TreeNode, path []string)

	dfs = func(node *TreeNode, path []string) {
		if node == nil {
			return
		}
		path = append(path, strconv.Itoa(node.Val))
		if node.Left == nil && node.Right == nil {
			ans = append(ans, strings.Join(path, "->"))
			return
		}
		dfs(node.Left, path)
		dfs(node.Right, path)

	}

	dfs(root, []string{})
	return ans
}

// leetcode222. 完全二叉树的节点个数 TODO??????????????????
func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	level := 0
	for node := root; node.Left != nil; node = node.Left {
		level++
	}
	return sort.Search(1<<(level+1), func(k int) bool {
		if k <= 1<<level {
			return false
		}
		bits := 1 << (level - 1)
		node := root
		for node != nil && bits > 0 {
			if bits&k == 0 {
				node = node.Left
			} else {
				node = node.Right
			}
			bits >>= 1
		}
		return node == nil
	}) - 1
}

func countNodesV2(root *TreeNode) int { // 返回以root为根节点的子树的节点个数
	if root == nil { // 递归的出口
		return 0
	}
	lH, rH := 0, 0             // 两侧高度
	lNode, rNode := root, root // 两个指针

	for lNode != nil { // 计算左侧高度
		lH++
		lNode = lNode.Left
	}
	for rNode != nil { // 计算右侧高度
		rH++
		rNode = rNode.Right
	}
	if lH == rH { // 当前子树是满二叉树，返回出节点数
		return 1<<lH - 1 // 左移n位就是乘以2的n次方
	}
	// 当前子树不是完美二叉树，只是完全二叉树，递归处理左右子树
	return 1 + countNodes(root.Left) + countNodes(root.Right)
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func countNodesV3(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := heightC(root.Left)
	right := heightC(root.Right)
	if left == right {
		return (1 << left) + countNodes(root.Right)
	} else {
		return (1 << (left - 1)) + countNodes(root.Left)
	}
}

func heightC(node *TreeNode) int {
	if node == nil {
		return 0
	}
	return 1 + heightC(node.Left)
}

// leetcode98. 验证二叉搜索树
func isValidBST(root *TreeNode) bool {
	return validateHelp(root, nil, nil)
}

func validateHelp(node, min, max *TreeNode) bool {
	if node == nil {
		return true
	}
	if min != nil && node.Val <= min.Val {
		return false
	}
	if max != nil && node.Val >= max.Val {
		return false
	}
	return validateHelp(node.Left, min, node) && validateHelp(node.Right, node, max)
}

// leetcode450. 删除二叉搜索树中的节点
func deleteNode(root *TreeNode, key int) *TreeNode {
	if root == nil { //0)排错
		return nil //1)这是BST的删除，不是一般的删除；但，不用平衡，很基础
	}
	var dummy = &TreeNode{nil, root, 0}
	var p, cur = &dummy.Right, dummy.Right //2) **TreeNode, *TreeNode
	for cur != nil && cur.Val != key {
		if key < cur.Val {
			p, cur = &cur.Left, cur.Left
			continue
		}
		p, cur = &cur.Right, cur.Right
	}
	if cur != nil {
		BSTDelete(p, cur)
	}
	return dummy.Right
}

func BSTDelete(p **TreeNode, cur *TreeNode) {
	if cur.Left == nil {
		*p = cur.Right
		return
	}
	if cur.Right == nil { //2)已经处理了叶节点的情况
		*p = cur.Left
		return
	}
	*p = cur.Right
	var rightLeftMost = cur.Right
	for rightLeftMost.Left != nil {
		rightLeftMost = rightLeftMost.Left
	}
	rightLeftMost.Left = cur.Left
}

// leetcode1325. 删除给定值的叶子节点
func removeLeafNodes(root *TreeNode, target int) *TreeNode {
	if root == nil {
		return root
	}
	root.Left = removeLeafNodes(root.Left, target)
	root.Right = removeLeafNodes(root.Right, target)
	if root.Left == nil && root.Right == nil && root.Val == target {
		return nil
	}
	return root
}

// leetcode208. 实现 Trie (前缀树)
type Trie struct {
	Next    [26]*Trie
	IsWorld bool
}

/** Initialize your data structure here. */
func ConstructorTire() Trie {
	return Trie{}
}

/** Inserts a word into the trie. */
func (this *Trie) Insert(word string) {
	cur := this
	for i := 0; i < len(word); i++ {
		if cur.Next[word[i]-'a'] == nil {
			cur.Next[word[i]-'a'] = &Trie{}
		}
		cur = cur.Next[word[i]-'a']
		if i == len(word)-1 {
			cur.IsWorld = true
		}
	}
}

/** Returns if the word is in the trie. */
func (this *Trie) Search(word string) bool {
	cur := this
	for _, ch := range word {
		if cur.Next[ch-'a'] != nil {
			cur = cur.Next[ch-'a']
		} else {
			return false
		}

	}
	return cur.IsWorld
}

/** Returns if there is any word in the trie that starts with the given prefix. */
func (this *Trie) StartsWith(prefix string) bool {
	for _, alph := range prefix {
		if this.Next[alph-'a'] != nil {
			this = this.Next[alph-'a']
		} else {
			return false
		}
	}
	return true
}

func recoverTree(root *TreeNode) {
	cur := root
	var stack []*TreeNode
	var pre *TreeNode = nil
	var x, y *TreeNode
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		node := stack[len(stack)-1]
		if pre != nil && node.Val < pre.Val {
			y = node
			if x == nil {
				x = pre
			} else {
				break
			}
		}
		pre = node
		stack = stack[:len(stack)-1]
		cur = node.Right
	}
	x.Val, y.Val = y.Val, x.Val
}
