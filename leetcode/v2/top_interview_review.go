package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

func twoSum(nums []int, target int) []int {
	valueMap := make(map[int]int)
	var ask []int

	for i, num := range nums {

		if k, ok := valueMap[target-num]; ok {
			ask = append(ask, k, i)
			return ask
		} else {
			valueMap[num] = i
		}

	}

	return ask
}

func romanToInt(s string) int {
	var symbol2value = map[string]int{
		"I":  1,
		"IV": 4,
		"V":  5,
		"IX": 9,
		"X":  10,
		"XL": 40,
		"L":  50,
		"XC": 90,
		"C":  100,
		"CD": 400,
		"D":  500,
		"CM": 900,
		"M":  1000,
	}
	val := 0
	for i := 0; i < len(s); i++ {
		if i+1 < len(s) {
			symbol := s[i : i+2]
			if value, ok := symbol2value[symbol]; ok {
				val += value
				i++
				continue
			}
		}
		if value, ok := symbol2value[string(s[i])]; ok {
			val += value
		}
	}

	return val
}

// 14. Longest Common Prefix
func LongestCommonPrefix(strs []string) string {
	var sb strings.Builder
	for pivotIndex := 0; pivotIndex < len(strs[0]); pivotIndex++ {
		i := 0
		for i < len(strs) {
			if pivotIndex >= len(strs[i]) {
				break
			}
			if strs[i][pivotIndex] != strs[0][pivotIndex] {
				break
			}
			i++
		}
		if i == len(strs) {
			sb.WriteByte(strs[0][pivotIndex])
		} else {
			break
		}
	}
	return sb.String()
}

// 20. Valid Parentheses
func isValid(s string) bool {
	var stack []byte
	for i := 0; i < len(s); i++ {
		lth := len(stack)
		if lth > 0 {
			if s[i] == ')' && stack[lth-1] == '(' || s[i] == '}' && stack[lth-1] == '{' || s[i] == ']' && stack[lth-1] == '[' {
				stack = stack[:lth-1]
				continue
			}
		}
		stack = append(stack, s[i])
	}

	if len(stack) > 0 {
		return false
	}
	return true
}

// Definition for singly-linked list

type ListNode struct {
	Val  int
	Next *ListNode
}

// 21. Merge Two Sorted Lists
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil && list2 == nil {
		return nil
	}
	if list1 != nil && list2 == nil {
		return list1
	}
	if list1 == nil && list2 != nil {
		return list2
	}
	var head *ListNode
	if list1.Val < list2.Val {
		head = list1
		head.Next = mergeTwoLists(list1.Next, list2)
	} else {
		head = list2
		head.Next = mergeTwoLists(list1, list2.Next)
	}
	return head
}

func mergeTwoListsV2(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := new(ListNode)
	cur := dummy
	for list1 != nil || list2 != nil {
		if list2 == nil || list1 != nil && list1.Val < list2.Val {
			cur.Next = list1
			list1 = list1.Next
		} else {
			cur.Next = list2
			list2 = list2.Next
		}
		cur = cur.Next
	}
	return dummy.Next
}

// 26. Remove Duplicates from Sorted Array
func removeDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	cmp := nums[0]
	index := 1
	for i := 1; i < len(nums); i++ {
		if nums[i] != cmp {
			nums[index] = nums[i]
			cmp = nums[i]
			index++
		}
	}
	return index
}

func removeDuplicatesV2(nums []int) int {
	cur := 0
	for cur < len(nums)-1 {
		pivot := nums[cur]
		j := cur + 1
		for j < len(nums) && nums[j] == pivot {
			j++
		}
		nums = append(nums[:cur+1], nums[j:]...)
		cur++
	}
	return len(nums)

}

// 28. Implement strStr()
func strStr(haystack string, needle string) int {
	lthn := len(needle)
	lthh := len(haystack)
	for i := 0; i < lthh; i++ {
		if haystack[i] == needle[0] && i+lthn <= lthh {
			if haystack[i:i+lthn] == needle {
				return i
			}
		}
	}
	return -1
}

// 66. Plus One
func plusOne(digits []int) []int {
	var res []int
	lth := len(digits)

	carry := 1

	for i := lth - 1; i >= 0; i-- {
		sum := digits[i] + carry
		num := sum % 10
		carry = sum / 10
		res = append(res, num)
	}
	if carry > 0 {
		res = append(res, carry)
	}
	reverseSlice(res)
	return res
}

func reverseSlice(nums []int) {
	lhs := 0
	rhs := len(nums) - 1
	for lhs < rhs {
		nums[lhs], nums[rhs] = nums[rhs], nums[lhs]
		lhs++
		rhs--
	}
}

func plusOneV2(digits []int) []int {
	lth := len(digits)
	going := 1
	for i := lth - 1; i >= 0; i-- {
		sum := digits[i] + going
		going = sum / 10
		val := sum % 10
		digits[i] = val
	}

	if going == 1 {
		var res []int
		res = append(res, going)
		res = append(res, digits...)
		return res
	} else {
		return digits
	}
}

// 69. Sqrt(x)
func mySqrt(x int) int {
	res := 0
	for res*res < x {
		res++
	}
	if res*res == x {
		return res
	}
	return res - 1

}

func mySqrtV2(x int) int {
	if x == 1 {
		return 1
	}
	low, high := 0, x
	var mid int
	var sqr int

	for {
		mid = (low + high) / 2
		if mid == low {
			return mid
		}

		sqr = mid * mid
		if sqr == x {
			return mid
		}
		if sqr > x {
			high = mid
		}
		if sqr < x {
			low = mid
		}
	}
}

func mySqrtV3(x int) int {
	if x == 1 {
		return 1
	}
	lhs := 0
	rhs := x

	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if mid*mid == x {
			return mid
		} else if mid*mid > x {
			rhs = mid
		} else {
			if (mid+1)*(mid+1) > x {
				return mid
			} else {
				lhs = mid
			}
		}
	}
	return -1
}

// 70. Climbing Stairs
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// 88. Merge Sorted Array
func merge(nums1 []int, m int, nums2 []int, n int) {
	if n == 0 {
		return
	}
	putPosition := m + n - 1
	cmpNums1 := m - 1
	cmpNums2 := n - 1
	for cmpNums1 >= 0 && cmpNums2 >= 0 {
		if nums1[cmpNums1] > nums2[cmpNums2] {
			nums1[putPosition] = nums1[cmpNums1]
			cmpNums1--
		} else {
			nums1[putPosition] = nums2[cmpNums2]
			cmpNums2--
		}
		putPosition--
	}
	if cmpNums1 < 0 && cmpNums2 >= 0 {
		for i := cmpNums2; i >= 0; i, putPosition = i-1, putPosition-1 {
			nums1[putPosition] = nums2[i]
		}
	}

}

func mergeV2(nums1 []int, m int, nums2 []int, n int) {
	index1 := m - 1
	index2 := n - 1
	iteIndex := len(nums1) - 1
	for index1 >= 0 || index2 >= 0 {
		if index2 < 0 {
			break
		}
		if index1 < 0 || (index1 >= 0 && index2 >= 0 && nums1[index1] < nums2[index2]) {
			nums1[iteIndex] = nums2[index2]
			index2--
			iteIndex--
		} else {
			nums1[iteIndex] = nums1[index1]
			iteIndex--
			index1--
		}
	}
}

// Definition for a binary tree node.
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	var res []int
	recInorderTraversal(root, &res)
	return res
}

func recInorderTraversal(root *TreeNode, res *[]int) {
	if root == nil {
		return
	}
	recInorderTraversal(root.Left, res)
	*res = append(*res, root.Val)
	recInorderTraversal(root.Right, res)
}

// 101. Symmetric Tree
func isSymmetric(root *TreeNode) bool {
	return recIsSymmetric(root.Left, root.Right)
}

func recIsSymmetric(node1 *TreeNode, node2 *TreeNode) bool {
	if node1 == nil && node2 == nil {
		return true
	}
	if (node1 == nil && node2 != nil) || (node1 != nil && node2 == nil) {
		return false
	}
	if node1.Val != node2.Val {
		return false
	}
	return recIsSymmetric(node1.Left, node2.Right) && recIsSymmetric(node1.Right, node2.Left)
}

func isSymmetricV2(root *TreeNode) bool {

	var nodeList []*TreeNode
	nodeList = append(nodeList, root)

	for len(nodeList) > 0 {
		var valList []int
		lth := len(nodeList)
		for i := 0; i < lth; i++ {
			if nodeList[i] == nil {
				valList = append(valList, -1000)
				//nodeList = append(nodeList, nil, nil)
			} else {
				valList = append(valList, nodeList[i].Val)
				nodeList = append(nodeList, nodeList[i].Left, nodeList[i].Right)
			}
		}
		lhs := 0
		rhs := len(valList) - 1
		for lhs < rhs {
			if valList[lhs] != valList[rhs] {
				return false
			}
			lhs++
			rhs--
		}
		nodeList = nodeList[lth:]
	}
	return true
}

func isSymmetricV3(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return isSymmetricHelp(root.Left, root.Right)
}

func isSymmetricHelp(node1, node2 *TreeNode) bool {
	if node1 == nil && node2 == nil {
		return true
	}
	if node1 != nil && node2 != nil {
		if node1.Val == node2.Val {
			return isSymmetricHelp(node1.Left, node2.Right) &&
				isSymmetricHelp(node1.Right, node2.Left)
		}
	}

	return false
}

// 104. Maximum Depth of Binary Tree
func maxDepth(root *TreeNode) int {
	return recMaxDepth(root, 0)

}
func recMaxDepth(node *TreeNode, depth int) int {
	if node != nil {
		depth++
		return max(recMaxDepth(node.Left, depth), recMaxDepth(node.Right, depth))
	}
	return depth
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

//108. Convert Sorted Array to Binary Search Tree

func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := len(nums) / 2
	return &TreeNode{
		Val:   nums[mid],
		Left:  sortedArrayToBST(nums[:mid]),
		Right: sortedArrayToBST(nums[mid+1:]),
	}
}

// 118. Pascal's Triangle
func generate(numRows int) [][]int {
	res := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		lhs := 0
		rhs := i
		res[i] = append(res[i], 1)
		for j := lhs + 1; j < rhs; j++ {
			res[i] = append(res[i], res[i-1][j-1]+res[i-1][j])
		}
		if rhs > lhs {
			res[i] = append(res[i], 1)
		}
	}
	return res
}

// 121. Best Time to Buy and Sell Stock
func maxProfit(prices []int) int {
	maxP := 0
	min := prices[0]
	for i := 1; i < len(prices); i++ {
		if prices[i] < min {
			min = prices[i]
		} else {
			maxP = max(maxP, prices[i]-min)
		}
	}
	return maxP
}

// 125. Valid Palindrome
func isPalindrome(s string) bool {
	lhs := 0
	rhs := len(s) - 1
	s = strings.ToLower(s)
	for lhs < rhs {
		for lhs < len(s) && !unicode.IsNumber(rune(s[lhs])) && !unicode.IsLetter(rune(s[lhs])) {
			lhs++
		}
		for rhs >= 0 && !unicode.IsNumber(rune(s[rhs])) && !unicode.IsLetter(rune(s[rhs])) {
			rhs--
		}
		//i := s[lhs]
		//j := s[rhs]
		//fmt.Println(i, j)
		if lhs < rhs {
			if s[lhs] != s[rhs] {
				return false
			}
		}

		lhs++
		rhs--
	}
	return true
}

func isPalindromeV2(s string) bool {
	s = strings.ToLower(s)
	lhs := 0
	rhs := len(s) - 1
	for lhs <= rhs {
		condition1 := (s[lhs] >= '0' && s[lhs] <= '9') || (s[lhs] >= 'a' && s[lhs] <= 'z')
		condition2 := (s[rhs] >= '0' && s[rhs] <= '9') || (s[rhs] >= 'a' && s[rhs] <= 'z')
		if !condition1 {
			lhs++
			continue
		}
		if !condition2 {
			rhs--
			continue
		}

		if condition1 && condition2 {
			if s[lhs] != s[rhs] {
				return false
			}
			lhs++
			rhs--
		}

	}
	return true
}

// 136. Single Number
func singleNumber(nums []int) int {
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		res ^= nums[i]
	}
	return res
}

// 141. Linked List Cycle
func hasCycle(head *ListNode) bool {
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

// 160. Intersection of Two Linked Lists
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	stepA := 0
	stepB := 0
	curA := headA
	curB := headB
	for curA != nil {
		stepA++
		curA = curA.Next
	}

	for curB != nil {
		stepB++
		curB = curB.Next
	}
	curA = headA
	curB = headB
	if stepA > stepB {
		for i := 1; i <= stepA-stepB; i++ {
			curA = curA.Next
		}
	}
	if stepB > stepA {
		for i := 1; i <= stepB-stepA; i++ {
			curB = curB.Next
		}
	}
	for curB != nil && curA != nil {
		if curA == curB {
			return curA
		}
		curA = curA.Next
		curB = curB.Next
	}
	return nil
}

// 169. Majority Element
func majorityElement(nums []int) int {
	count := 0
	var candidate int
	for _, num := range nums {
		if count == 0 {
			candidate = num
			count++
			continue
		}
		if num == candidate {
			count++
		} else {
			count--
		}
	}
	return candidate
}

// 171. Excel Sheet Column Number
func titleToNumber(columnTitle string) int {
	acc := 1
	val := 0
	for i := len(columnTitle) - 1; i >= 0; i-- {
		num := acc * int(columnTitle[i]-'A'+1)
		val += num
		acc *= 26
	}
	return val

}

// 190. Reverse Bits
func reverseBits(num uint32) uint32 {
	val := uint32(0)
	for i := 1; i <= 32; i++ {
		tmp := num & 1
		num >>= 1
		val <<= 1
		val += tmp
	}
	return val
}

// 191. Number of 1 Bits
func hammingWeight(num uint32) int {
	cnt := 0
	for i := 1; i <= 32; i++ {
		if num&1 == 1 {
			cnt++
		}
		num >>= 1
	}
	return cnt
}

// 202. Happy Number
func isHappy(n int) bool {
	result := make(map[int]bool)
	sum := n
	for {
		nBytes := []byte(strconv.Itoa(sum))
		sum = 0
		for _, i := range nBytes {
			sum += int((i - '0') * (i - '0'))
		}
		if sum == 1 {
			return true
		}

		if _, exist := result[sum]; exist {
			return false
		}
		result[sum] = true
	}
}

func isHappyV2(n int) bool {
	isCompute := make(map[int]bool)
	for {
		f := 0
		for n >= 10 {
			tmp := n / 10
			num := n % 10
			f += num * num
			n = tmp
		}
		f += n * n
		n = f
		if n == 1 {
			return true
		}
		if _, ok := isCompute[n]; ok {
			return false
		}
		isCompute[n] = true
	}
}

// 206. Reverse Linked List
func reverseList(head *ListNode) *ListNode {
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

func reverseListV2(head *ListNode) *ListNode {
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

// 217. Contains Duplicate
func containsDuplicate(nums []int) bool {
	num2cnt := make(map[int]bool)
	for _, num := range nums {
		if _, ok := num2cnt[num]; ok {
			return true
		}
		num2cnt[num] = true
	}
	return false
}

// 234. Palindrome Linked List
func isPalindromeList(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	cur := head
	cnt := 0
	for cur != nil {
		cnt++
		cur = cur.Next
	}
	mid := cnt / 2
	otherHead := head
	for i := 1; i <= mid; i++ {
		otherHead = otherHead.Next
	}
	pre := otherHead
	cur = otherHead.Next
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	otherHead.Next = nil

	list1 := head
	list2 := pre
	for list2 != nil {
		if list2.Val != list1.Val {
			return false
		}
		list1 = list1.Next
		list2 = list2.Next
	}
	return true
}

func isPalindromeListV3(head *ListNode) bool {
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	l2 := reverseListV2(slow.Next)
	cur1, cur2 := head, l2
	for cur2 != nil {
		if cur2.Val != cur1.Val {
			return false
		}
		cur1 = cur1.Next
		cur2 = cur2.Next
	}
	return true
}

func isPalindromeListV2(head *ListNode) bool {
	slow := head
	fast := head
	var pre *ListNode = nil
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		next := slow.Next
		slow.Next = pre
		pre = slow
		slow = next
	}
	if fast != nil {
		slow = slow.Next
	}
	for pre != nil && slow != nil {
		if pre.Val != slow.Val {
			return false
		}
		pre = pre.Next
		slow = slow.Next
	}
	return true
}

// 237. Delete Node in a Linked Lis
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// 242. Valid Anagram
func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	lth := len(s)
	letters := make([]int, 26)
	for _, ch := range s {
		letters[ch-'a']++
	}
	for _, ch := range t {
		if letters[ch-'a'] > 0 {
			letters[ch-'a']--
			lth--
		}

	}

	if lth == 0 {
		return true
	}
	return false
}

func isAnagramV2(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	var hash [26]int
	for i := 0; i < len(s); i++ {
		hash[s[i]-'a']++
	}

	for i := 0; i < len(t); i++ {
		hash[t[i]-'a']--
		if hash[t[i]-'a'] < 0 {
			return false
		}
	}
	return true
}

//268. Missing Number

func missingNumber(nums []int) int {
	lth := len(nums)
	sum := (1 + lth) * lth / 2
	cmpSum := 0
	for _, num := range nums {
		cmpSum += num
	}
	return sum - cmpSum
}

func missingNumberV2(nums []int) int {
	index := 0
	for index < len(nums) {
		if nums[index] != index && nums[index] < len(nums) {
			nums[index], nums[nums[index]] = nums[nums[index]], nums[index]
			continue
		}
		index++
	}
	index = 0
	for index < len(nums) {
		if nums[index] != index {
			return index
		}
		index++
	}
	return index
}

// 283. Move Zeroes
func moveZeroes(nums []int) {
	swapIndex := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[i], nums[swapIndex] = nums[swapIndex], nums[i]
			swapIndex++
		}
	}
}

// 326. Power of Three
func isPowerOfThree(n int) bool {
	acc := 1
	for {
		if acc == n {
			return true
		} else if acc > n {
			return false
		} else {
			acc *= 3
		}

	}
}

func isPowerOfThreeV2(n int) bool {
	max3 := 1162261467
	if n <= 0 {
		return false
	}
	return max3%n == 0
}

func isPowerOfThreeV3(n int) bool {
	switch {
	case n == 1:
		return true
	case n == 0:
		return false
	case n%3 != 0:
		return false
	default:
		return isPowerOfThree(n / 3)
	}
}

// 344. Reverse String
func reverseString(s []byte) {
	lhs := 0
	rhs := len(s) - 1
	for lhs < rhs {
		s[lhs], s[rhs] = s[rhs], s[lhs]
		lhs++
		rhs--
	}

}

// 350. Intersection of Two Arrays II
func intersect(nums1 []int, nums2 []int) []int {
	var res []int
	num2cnt := make(map[int]int)
	for _, num := range nums1 {
		num2cnt[num]++
	}
	for _, num := range nums2 {
		if _, ok := num2cnt[num]; ok && num2cnt[num] > 0 {
			res = append(res, num)
			num2cnt[num]--
		}
	}
	return res
}

// 387. First Unique Character in a String
func firstUniqChar(s string) int {
	hash := make([]int, 26)
	for _, ch := range s {
		hash[ch-'a']++
	}
	for i, ch := range s {
		if hash[ch-'a'] == 1 {
			return i
		}
	}
	return -1
}

// 412. Fizz Buzz
// answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
// answer[i] == "Fizz" if i is divisible by 3.
// answer[i] == "Buzz" if i is divisible by 5.
// answer[i] == i (as a string) if none of the above conditions are true.
func fizzBuzz(n int) []string {
	var res []string
	for i := 1; i <= n; i++ {
		if i%3 == 0 && i%5 == 0 {
			res = append(res, "FizzBuzz")
		} else if i%3 == 0 {
			res = append(res, "Fizz")
		} else if i%5 == 0 {
			res = append(res, "Buzz")
		} else {
			res = append(res, strconv.Itoa(i))
		}
	}
	return res
}

// 2. Add Two Numbers
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	carry := 0
	mergeHead := new(ListNode)
	cur := mergeHead
	for l1 != nil && l2 != nil {
		val := (l1.Val + l2.Val + carry) % 10
		carry = (l1.Val + l2.Val + carry) / 10
		cur.Next = &ListNode{
			Val:  val,
			Next: nil,
		}
		l1 = l1.Next
		l2 = l2.Next
		cur = cur.Next
	}
	for l1 != nil {
		val := (l1.Val + carry) % 10
		carry = (l1.Val + carry) / 10
		cur.Next = &ListNode{
			Val:  val,
			Next: nil,
		}
		cur = cur.Next
		l1 = l1.Next

	}

	for l2 != nil {
		val := (l2.Val + carry) % 10
		carry = (l2.Val + carry) / 10
		cur.Next = &ListNode{
			Val:  val,
			Next: nil,
		}
		cur = cur.Next
		l2 = l2.Next

	}
	if carry > 0 {

		cur.Next = &ListNode{
			Val:  carry,
			Next: nil,
		}
	}

	return mergeHead.Next
}

func addTwoNumbersV2(l1 *ListNode, l2 *ListNode) *ListNode {
	head := new(ListNode)
	head.Next = addTwoNumbersRec(l1, l2, 0)
	return head.Next

}

func addTwoNumbersRec(l1 *ListNode, l2 *ListNode, jinWei int) *ListNode {
	node := new(ListNode)
	if l1 == nil && l2 != nil {
		node.Val = (jinWei + l2.Val) % 10
		jinWei = (jinWei + l2.Val) / 10
		node.Next = addTwoNumbersRec(l1, l2.Next, jinWei)
	} else if l1 != nil && l2 == nil {
		node.Val = (jinWei + l1.Val) % 10
		jinWei = (jinWei + l1.Val) / 10
		node.Next = addTwoNumbersRec(l1.Next, l2, jinWei)
	} else if l1 != nil && l2 != nil {
		node.Val = (jinWei + l1.Val + l2.Val) % 10
		jinWei = (jinWei + l1.Val + l2.Val) / 10
		node.Next = addTwoNumbersRec(l1.Next, l2.Next, jinWei)
	} else {
		if jinWei != 0 {
			node.Val = jinWei
		} else {
			node = nil
		}
	}

	return node
}

func addTwoNumbersV3(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	cur1 := l1
	cur2 := l2
	carry := 0
	curNode := dummy
	for cur1 != nil || cur2 != nil {
		sum := 0
		if cur1 != nil {
			sum += cur1.Val
			cur1 = cur1.Next
		}
		if cur2 != nil {
			sum += cur2.Val
			cur2 = cur2.Next
		}
		sum += carry
		num := sum % 10
		carry = sum / 10
		curNode.Next = &ListNode{
			Val: num,
		}
		curNode = curNode.Next
	}
	if carry > 0 {
		curNode.Next = &ListNode{
			Val: carry,
		}
	}
	return dummy.Next
}

// 3. Longest Substring Without Repeating Character
func lengthOfLongestSubstring(s string) int {
	if len(s) == 0 {
		return 0
	}
	ch2index := make(map[int32]int)
	maxLth := 0
	dp := make([]int, len(s))
	dp[0] = 0
	for i, ch := range s {
		if _, ok := ch2index[ch]; !ok {
			ch2index[ch] = i
			if i > 0 {
				dp[i] = dp[i-1]
			}
			if maxLth < (i - dp[i] + 1) {
				maxLth = i - dp[i] + 1
			}
		} else {
			index := ch2index[ch]
			if index < dp[i-1] {
				ch2index[ch] = i
				dp[i] = dp[i-1]
				if maxLth < (i - dp[i] + 1) {
					maxLth = i - dp[i] + 1
				}
			} else {
				dp[i] = ch2index[ch] + 1
				ch2index[ch] = i
			}

		}
	}
	return maxLth

}

func lengthOfLongestSubstringV2(s string) int {
	maxLength := 0
	left := 0
	hash := make([]int, 256)
	for i := 0; i < len(s); i++ {
		if hash[s[i]] == 0 || hash[s[i]] < left {
			if maxLength < i-left+1 {
				maxLength = i - left + 1
			}
		} else {
			left = hash[s[i]]
		}
		hash[s[i]] = i + 1
	}
	return maxLength
}

func lengthOfLongestSubstringV3(s string) int {
	lhs := 0
	rhs := 0
	ch2index := make(map[uint8]int)
	maxLth := 0
	for rhs < len(s) {
		if index, ok := ch2index[s[rhs]]; !ok || lhs > index {
			ch2index[s[rhs]] = rhs
			if rhs-lhs+1 > maxLth {
				maxLth = rhs - lhs + 1
			}
		} else {
			lhs = index + 1
			ch2index[s[rhs]] = rhs
		}
		rhs++
	}
	return maxLth
}

// 5. Longest Palindromic Substring
func longestPalindrome(s string) string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(s); i++ {
		dp[i] = make([]bool, len(s))
		dp[i][i] = true
	}
	maxStr := ""
	for i := len(s) - 1; i >= 0; i-- {
		for j := len(s) - 1; j >= i; j-- {
			if s[i] != s[j] {
				dp[i][j] = false
				continue
			}
			if j-i == 1 || i == j {
				dp[i][j] = true
			} else {
				dp[i][j] = dp[i+1][j-1]
			}
			if dp[i][j] && len(maxStr) < j-i+1 {
				maxStr = s[i : j+1]
			}
		}
	}
	return maxStr
}

func longestPalindromeV2(s string) string {
	maxSub := s[:1]
	lth := len(s)
	for i := 1; i < lth; i++ {
		lhs := i - 1
		rhs := i + 1
		for lhs >= 0 && s[lhs] == s[i] {
			lhs--
		}
		for rhs < lth && s[rhs] == s[i] {
			rhs++
		}
		for lhs >= 0 && rhs < lth {
			if s[lhs] != s[rhs] {
				break
			}
			lhs--
			rhs++
		}
		if len(s[lhs+1:rhs]) > len(maxSub) {
			maxSub = s[lhs+1 : rhs]
		}
	}
	return maxSub

}

// 7. Reverse Integer
func reverse(x int) int {
	num := 0
	for x != 0 {
		remain := x % 10
		x /= 10
		num = num*10 + remain
		if num < math.MinInt32 || num > math.MaxInt32 {
			return 0
		}
	}
	return num
}

// 8. String to Integer (atoi)
func myAtoi(s string) int {
	num := 0
	start := 0
	isPositive := 1
	for ; start < len(s) && s[start] == ' '; start++ {
	}
	if start < len(s) && (s[start] == '+' || s[start] == '-') {
		if s[start] == '-' {
			isPositive = -1
		}
		start++
	}
	for i := start; i < len(s) && s[i] >= '0' && s[i] <= '9'; i++ {
		num = num*10 + int(s[i]-'0')
		if num*isPositive > math.MaxInt32 {
			return math.MaxInt32
		}
		if num*isPositive < math.MinInt32 {
			return math.MinInt32
		}
	}
	return num * isPositive
}

// 11. Container With Most Water
func maxArea(height []int) int {
	lhs := 0
	rhs := len(height) - 1
	maxContain := 0
	for lhs < rhs {
		line := 0
		diff := rhs - lhs
		if height[lhs] < height[rhs] {
			line = height[lhs]
			lhs++
		} else {
			line = height[rhs]
			rhs--
		}
		contain := line * diff
		if contain > maxContain {
			maxContain = contain
		}

	}
	return maxContain
}

func maxAreaV2(height []int) int {
	lth := len(height)
	lhs := 0
	rhs := lth - 1
	maxContain := 0
	for lhs < rhs {
		line := 0
		if height[lhs] < height[rhs] {
			line = height[lhs]
		} else {
			line = height[rhs]
		}
		contain := line * (rhs - lhs)
		if contain > maxContain {
			maxContain = contain
		}
		for lhs < rhs && height[lhs] <= line {
			lhs++
		}
		for lhs < rhs && height[rhs] <= line {
			rhs--
		}

	}
	return maxContain
}

// 15. 3Sum
func threeSum(nums []int) [][]int {
	sort.Slice(nums, func(i, j int) bool { return nums[i] < nums[j] })
	lth := len(nums)
	var res [][]int
	for i := 0; i <= lth-3 && nums[i] <= 0; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		lhs := i + 1
		rhs := lth - 1
		target := 0 - nums[i]
		for lhs < rhs {
			if nums[rhs]+nums[lhs] == target {
				res = append(res, []int{nums[i], nums[lhs], nums[rhs]})
				lhs++
				rhs--
				for lhs < rhs && nums[lhs] == nums[lhs-1] {
					lhs++
				}
				for lhs < rhs && nums[rhs] == nums[rhs+1] {
					rhs--
				}
			} else if nums[rhs]+nums[lhs] > target {
				rhs--
			} else {
				lhs++
			}
		}
	}
	return res

}

// 17. Letter Combinations of a Phone Number
// TODO string:range方法和取下标的区别
func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	var digit2letter = map[int32]string{
		'2': "abc",
		'3': "def",
		'4': "ghi",
		'5': "jkl",
		'6': "mno",
		'7': "pqrs",
		'8': "tuv",
		'9': "wxyz",
	}
	var res = []string{""}
	for _, digit := range digits {
		res = letterCombinationsHelp(digit2letter[digit], res)
	}
	return res
}

func letterCombinationsHelp(letters string, cur []string) []string {
	var res []string
	for _, str := range cur {
		for _, letter := range letters {
			tmp := str + string(letter)
			res = append(res, tmp)
		}
	}
	return res
}

// 19. Remove Nth Node From End of List
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	slow := head
	fast := head
	for i := 1; i <= n && fast != nil; i++ {
		fast = fast.Next
	}
	if fast == nil {
		return head.Next
	}

	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}

	slow.Next = slow.Next.Next
	return head
}

// 22. Generate Parentheses
func generateParenthesis(n int) []string {
	var res []string
	generateParenthesisHelp(n, 1, 0, "(", &res)
	return res
}

func generateParenthesisHelp(n, left, right int, curParenthesis string, res *[]string) {
	if left == n && right == n {
		*res = append(*res, curParenthesis)
		return
	}
	if left > right && left < n && right < n {
		generateParenthesisHelp(n, left+1, right, curParenthesis+"(", res)
		generateParenthesisHelp(n, left, right+1, curParenthesis+")", res)
	} else if left == right && left < n {
		generateParenthesisHelp(n, left+1, right, curParenthesis+"(", res)
	} else if left == n && right < n {
		generateParenthesisHelp(n, left, right+1, curParenthesis+")", res)
	}
}

func generateParenthesisV2(n int) []string {
	var res []string
	generateParenthesisHelpV2(&res, "", 0, 0, n)
	return res
}

func generateParenthesisHelpV2(res *[]string, str string, cnt, remain2match, num int) {
	if cnt == num && remain2match == 0 {
		*res = append(*res, str)
		return
	}
	if cnt < num {
		generateParenthesisHelpV2(res, str+"(", cnt+1, remain2match+1, num)
	}
	if remain2match > 0 {
		generateParenthesisHelpV2(res, str+")", cnt, remain2match-1, num)
	}
}

func generateParenthesisV3(n int) []string {
	var ret []string
	var dfs func(leftCnt, rightCnt int, str string)

	dfs = func(leftCnt, rightCnt int, str string) {
		if leftCnt == n && rightCnt == n {
			ret = append(ret, str)
		}
		if leftCnt < n {
			dfs(leftCnt+1, rightCnt, str+"(")
		}
		if rightCnt < n && leftCnt > rightCnt {
			dfs(leftCnt, rightCnt+1, str+")")
		}
	}

	dfs(0, 0, "")
	return ret
}

// 29. Divide Two Integers
func divide(dividend int, divisor int) int {

	if dividend == math.MinInt32 && divisor == -1 {
		return math.MaxInt32
	}
	isPositive := 1

	if (dividend < 0 && divisor > 0) || (dividend > 0 && divisor < 0) {
		isPositive = -1
	}
	if dividend < 0 {
		dividend *= -1
	}
	if divisor < 0 {
		divisor *= -1
	}
	cnt := 0
	sum := 0
	for sum < dividend {
		sum += divisor
		cnt++
		if cnt*isPositive < math.MinInt32 {
			return math.MinInt32
		}
		if cnt*isPositive > math.MaxInt32 {
			return math.MaxInt32
		}
	}
	if sum == dividend {
		return cnt * isPositive
	} else {
		return (cnt - 1) * isPositive
	}
}

func divideV2(dividend int, divisor int) int {
	if dividend == math.MinInt32 && divisor == -1 {
		return math.MaxInt32
	}
	dendSigned := 1
	if dividend < 0 {
		dendSigned = 0
		dividend *= -1
	}
	sorSigned := 1
	if divisor < 0 {
		sorSigned = 0
		divisor *= -1
	}
	signed := 1
	if dendSigned^sorSigned == 1 {
		signed = -1
	}
	ret := 0

	for dividend >= divisor {
		tmp, m := divisor, 1
		for dividend >= tmp<<1 {
			tmp, m = tmp<<1, m<<1
		}
		dividend -= tmp
		ret += m
	}
	return signed * ret

}

// 33. Search in Rotated Sorted Array
func search(nums []int, target int) int {
	lhs := 0
	rhs := len(nums) - 1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if target == nums[mid] {
			return mid
		}
		//右半部分排好序
		if nums[mid] < nums[rhs] {
			if target > nums[mid] && target <= nums[rhs] {
				lhs = mid + 1
			} else {
				rhs = mid - 1
			}
		} else if nums[mid] >= nums[lhs] {
			if target >= nums[lhs] && target < nums[mid] {
				rhs = mid - 1
			} else {
				lhs = mid + 1
			}
		}
	}
	return -1
}

// 34. Find First and Last Position of Element in Sorted Array
func searchRange(nums []int, target int) []int {
	lhs := 0
	rhs := len(nums) - 1
	index := -1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if target == nums[mid] {
			index = mid
			break
		} else if nums[mid] < target {
			lhs = mid + 1
		} else {
			rhs = mid - 1
		}
	}
	if index == -1 {
		return []int{-1, -1}
	}
	start, end := index, index
	for ; start >= 0 && nums[start] == target; start-- {
	}
	for ; end < len(nums) && nums[end] == target; end++ {
	}
	return []int{start + 1, end - 1}

}

func searchRangeV2(nums []int, target int) []int {
	targetStartIndex, targetEndIndex := -1, -1
	searchRangexxxHelp(nums, target, 0, len(nums)-1, &targetStartIndex, &targetEndIndex)
	return []int{targetStartIndex, targetEndIndex}
}

func searchRangexxxHelp(nums []int, target, start, end int, targetStartIndex, targetEndIndex *int) {
	if start <= end {
		index := bSearch(nums, target, start, end)
		if index != -1 {
			if *targetStartIndex == -1 {
				*targetStartIndex = index
			}
			if *targetEndIndex == -1 {
				*targetEndIndex = index
			}
			if *targetStartIndex != -1 && index < *targetStartIndex {
				*targetStartIndex = index
			}
			if *targetEndIndex != -1 && *targetEndIndex < index {
				*targetEndIndex = index
			}
			searchRangexxxHelp(nums, target, start, index-1, targetStartIndex, targetEndIndex)
			searchRangexxxHelp(nums, target, index+1, end, targetStartIndex, targetEndIndex)
		}

	}
}

func bSearch(nums []int, target, start, end int) int {
	for start <= end {
		mid := (start + end) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > target {
			end = mid - 1
		} else {
			start = mid + 1
		}
	}
	return -1
}

func searchRangeV3(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}
	return []int{binaryLeft(nums, target), binaryRight(nums, target)}
}

func binaryLeft(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left < right {
		mid := (right + left) / 2
		if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	if nums[left] == target {
		return left
	}
	return -1
}

func binaryRight(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left < right {
		mid := (right+left)/2 + 1
		if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] <= target {
			left = mid
		}
	}
	if nums[left] == target {
		return left
	}
	return -1
}

func searchRangeV4(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}
	left := bLeft(nums, target)
	if left >= len(nums) || nums[left] != target {
		return []int{-1, -1}
	}
	right := bRight(nums, target)
	return []int{left, right}
}

func bLeft(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := (right + left) / 2
		if target == nums[mid] {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	return left
}

func bRight(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := (right + left) / 2
		if nums[mid] == target {
			left = mid + 1
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	return right
}

// 36. Valid Sudoku
func isValidSudoku(board [][]byte) bool {
	rowRecord := make([]map[byte]bool, 9)
	columnRecord := make([]map[byte]bool, 9)
	subRecord := make([]map[byte]bool, 9)
	for i := 0; i < len(rowRecord); i++ {
		rowRecord[i] = make(map[byte]bool)
		columnRecord[i] = make(map[byte]bool)
		subRecord[i] = make(map[byte]bool)
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] < '1' && board[i][j] > '9' || board[i][j] == '.' {
				continue
			}
			if _, ok := rowRecord[i][board[i][j]]; ok {
				return false
			}

			if _, ok := columnRecord[j][board[i][j]]; ok {
				return false
			}
			rowRecord[i][board[i][j]] = true
			columnRecord[j][board[i][j]] = true
			subRow := i / 3
			subColumn := j/3 + 1
			subIndex := 3*subRow + subColumn
			if _, ok := subRecord[subIndex-1][board[i][j]]; ok {
				return false
			}
			subRecord[subIndex-1][board[i][j]] = true
		}
	}

	return true

}

// 38. Count and Say
func countAndSay(n int) string {
	str := "1"
	for i := 2; i <= n; i++ {
		digit := str[0]
		cnt := 0
		var tmp string
		for j := 0; j < len(str); j++ {
			if str[j] == digit {
				cnt++
			} else {
				tmp = tmp + strconv.Itoa(cnt) + string(digit)
				digit = str[j]
				cnt = 1
			}
		}
		tmp = tmp + strconv.Itoa(cnt) + string(digit)
		str = tmp
	}
	return str
}

func countAndSayV2(n int) string {
	say := ""
	for i := 1; i <= n; i++ {
		if i == 1 {
			say = "1"
			continue
		}
		cnt := uint8(0)
		var num uint8
		var sb strings.Builder
		for k := 0; k < len(say); k++ {
			if k == 0 {
				cnt++
				num = say[k]
			} else {
				if say[k] == say[k-1] {
					cnt++
				} else {
					sb.WriteByte(cnt + '0')
					sb.WriteByte(num)
					num = say[k]
					cnt = 1
				}
			}
		}
		sb.WriteByte(cnt + '0')
		sb.WriteByte(num)
		say = sb.String()

	}
	return say
}

// 46. Permutations
func permute(nums []int) [][]int {
	var res [][]int
	recPermute(0, nums, &res)
	return res
}

func recPermute(start int, nums []int, res *[][]int) {
	if start == len(nums)-1 {
		tmp := make([]int, len(nums))
		copy(tmp, nums)
		*res = append(*res, tmp)
		return
	}
	for i := start; i < len(nums); i++ {
		nums[i], nums[start] = nums[start], nums[i]
		recPermute(start+1, nums, res)
		nums[i], nums[start] = nums[start], nums[i]
	}
}

func permuteV2(nums []int) [][]int {
	var ret [][]int
	var dfs func(start int)
	dfs = func(start int) {
		if start == len(nums) {
			ret = append(ret, append([]int{}, nums...))
		}
		for i := start; i < len(nums); i++ {
			nums[i], nums[start] = nums[start], nums[i]
			dfs(start + 1)
			nums[i], nums[start] = nums[start], nums[i]
		}
	}
	dfs(0)
	return ret
}

// 48. Rotate Image
func rotate(matrix [][]int) {
	n := len(matrix) - 1
	for x := 0; x < len(matrix)/2; x++ {
		for y := x; y < n-x; y++ {
			t := matrix[x][y]
			matrix[x][y] = matrix[n-y][x]
			matrix[n-y][x] = matrix[n-x][n-y]
			matrix[n-x][n-y] = matrix[y][n-x]
			matrix[y][n-x] = t
		}
	}
}

//49. Group Anagrams

func groupAnagrams(strs []string) [][]string {
	dict := make(map[[26]int][]string)
	for _, v := range strs {
		ana := [26]int{}
		for _, c := range v {
			ana[c-'a']++
		}
		dict[ana] = append(dict[ana], v)
	}

	res := make([][]string, 0)
	for _, v := range dict {
		res = append(res, v)
	}
	return res
}

// 50. Pow(x, n)
func myPow(x float64, n int) float64 {
	if n < 0 {
		x, n = 1/x, -n
	}
	res := 1.0
	for n > 0 {
		if n&1 == 1 {
			res *= x
		}
		x, n = x*x, n>>1
	}
	return res
}

func myPowV2(x float64, n int) float64 {
	if n < 0 {
		x, n = 1/x, -n
	}
	var ret float64 = 1.0
	for n > 0 {
		tmp, acc := x, 1
		for n >= acc<<1 {
			tmp = tmp * tmp
			acc = acc << 1
		}
		ret *= tmp
		n -= acc
	}
	return ret

}

// 53. Maximum Subarray
func maxSubArray(nums []int) int {
	maxSubArr := nums[0]
	dp := make([][]int, len(nums))
	for i := 0; i < len(nums); i++ {
		dp[i] = make([]int, len(nums))
		dp[i][i] = nums[i]
		if dp[i][i] > maxSubArr {
			maxSubArr = dp[i][i]
		}
	}
	for i := len(nums) - 1; i >= 0; i-- {
		for j := len(nums) - 1; j > i; j-- {
			if i == j {
				continue
			} else if j-i == 1 {
				dp[i][j] = nums[i] + nums[j]
			} else {
				dp[i][j] = dp[i+1][j-1] + nums[i] + nums[j]
			}
			if dp[i][j] > maxSubArr {
				maxSubArr = dp[i][j]
			}
		}
	}
	return maxSubArr
}

func maxSubArrayV2(nums []int) int {
	lth := len(nums)
	dp := make([]int, lth)
	dp[0] = nums[0]
	res := dp[0]
	for i := 1; i < lth; i++ {
		dp[i] = max(nums[i], dp[i-1]+nums[i])
		res = max(res, dp[i])
	}
	return res
}

func maxSubArrayV3(nums []int) int {
	max := nums[0]
	pre := nums[0]
	for i := 1; i < len(nums); i++ {
		cur := nums[i]
		if nums[i]+pre > nums[i] {
			cur = nums[i] + pre
		}
		if cur > max {
			max = cur
		}
		pre = cur
	}
	return max
}

// 54. Spiral Matrix
func spiralOrder(matrix [][]int) []int {
	var res []int
	row := len(matrix)
	column := len(matrix[0])
	iterRows := row / 2
	if row&1 == 0 {
		iterRows -= 1
	}
	isVisited := make([]int, row*column)
	for i := 0; i <= iterRows; i++ {
		curRow := i
		curColumn := i
		//从左向右
		for ; curColumn < column-i; curColumn++ {
			key := curRow*column + curColumn
			if isVisited[key] == 0 {
				res = append(res, matrix[curRow][curColumn])
				isVisited[key] = 1
			}

		}
		curColumn -= 1
		//从上向下
		for curRow += 1; curRow < row-i; curRow++ {
			key := curRow*column + curColumn
			if isVisited[key] == 0 {
				res = append(res, matrix[curRow][curColumn])
				isVisited[key] = 1
			}
		}
		curRow -= 1
		//从右向左
		for curColumn -= 1; curColumn >= i; curColumn-- {
			key := curRow*column + curColumn
			if isVisited[key] == 0 {
				res = append(res, matrix[curRow][curColumn])
				isVisited[key] = 1
			}

		}
		curColumn += 1
		//从下向上
		for curRow -= 1; curRow > i; curRow-- {
			key := curRow*column + curColumn
			if isVisited[key] == 0 {
				res = append(res, matrix[curRow][curColumn])
				isVisited[key] = 1
			}
		}
	}
	return res

}

func spiralOrderV2(matrix [][]int) []int {
	startRow := 0
	startColumn := 0
	endRow := len(matrix) - 1
	endColumn := len(matrix[0]) - 1
	caps := (endRow + 1) * (endColumn + 1)
	var res []int
	var cnt int
	for endRow >= startRow && endColumn >= startColumn {
		for i := startColumn; i <= endColumn && cnt < caps; i++ {
			res = append(res, matrix[startRow][i])
			cnt++
		}
		for i := startRow + 1; i <= endRow && cnt < caps; i++ {
			res = append(res, matrix[i][endColumn])
			cnt++
		}

		for i := endColumn - 1; i >= startColumn && cnt < caps; i-- {
			res = append(res, matrix[endRow][i])
			cnt++
		}

		for i := endRow - 1; i > startRow && cnt < caps; i-- {
			res = append(res, matrix[i][startColumn])
			cnt++
		}
		startRow++
		endRow--
		startColumn++
		endColumn--
	}
	return res
}

func spiralOrderV3(matrix [][]int) []int {
	var res []int
	row := len(matrix)
	column := len(matrix[0])
	iterRows := row / 2
	if row&1 == 0 {
		iterRows -= 1
	}
	caps := row * column
	cnt := 0
	for i := 0; i <= iterRows; i++ {
		curRow := i
		curColumn := i
		//从左向右
		for ; curColumn < column-i && cnt < caps; curColumn++ {
			res = append(res, matrix[curRow][curColumn])
			cnt++
		}
		curColumn -= 1
		//从上向下
		for curRow += 1; curRow < row-i && cnt < caps; curRow++ {
			res = append(res, matrix[curRow][curColumn])
			cnt++
		}
		curRow -= 1
		//从右向左
		for curColumn -= 1; curColumn >= i && cnt < caps; curColumn-- {
			res = append(res, matrix[curRow][curColumn])
			cnt++

		}
		curColumn += 1
		//从下向上
		for curRow -= 1; curRow > i && cnt < caps; curRow-- {
			res = append(res, matrix[curRow][curColumn])
			cnt++
		}
	}
	return res

}

func spiralOrderV4(matrix [][]int) []int {
	var ret []int
	i := 0
	j := 0
	cnt := 0
	row := len(matrix)
	column := len(matrix[0])

	all := 0
	for {
		//向右
		if all == row*column {
			break
		}
		for ; j < column-cnt && all < row*column; j++ {
			ret = append(ret, matrix[i][j])
			all++
		}
		j--
		//向下
		for i += 1; i < row-cnt && all < row*column; i++ {
			ret = append(ret, matrix[i][j])
			all++
		}
		//向左
		i--
		for j -= 1; j >= cnt && all < row*column; j-- {
			ret = append(ret, matrix[i][j])
			all++
		}
		cnt++
		//向上
		j++
		for i -= 1; i >= cnt && all < row*column; i-- {
			ret = append(ret, matrix[i][j])
			all++
		}
		i++
		j++

	}
	return ret
}

// 55. Jump Game
func canJump(nums []int) bool {
	target := len(nums) - 1

	isDeal := make(map[int]bool)
	var dfs func(index int) bool
	dfs = func(index int) bool {

		if index >= target {
			return true
		}
		step := nums[index]
		isDeal[index] = true
		if index+step >= target {
			return true
		}

		for i := 1; i <= step; i++ {
			if _, ok := isDeal[index+i]; !ok {
				if dfs(index + i) {
					return true
				}
			}
		}
		return false

	}
	return dfs(0)
}

func canJumpV2(nums []int) bool {
	length := len(nums)
	if length == 1 {
		return true
	} else if nums[0] == 0 && length > 1 {
		return false
	}

	maxJump := nums[0]
	for i := 1; i < length-1; i++ {
		next := nums[i] + i
		if next > maxJump {
			maxJump = next
		}

		if i >= maxJump {
			return false
		}
	}

	return true
}

func canJumpV3(nums []int) bool {
	canJumpMaxIndex := 0
	for i := 0; i < len(nums); i++ {
		if canJumpMaxIndex >= len(nums)-1 {
			return true
		}
		if canJumpMaxIndex >= i {
			step := nums[i]
			if step+i > canJumpMaxIndex {
				canJumpMaxIndex = step + i
			}
		}
	}
	return false
}

// 牛逼
func canJumpV4(nums []int) bool {
	lastPos := len(nums) - 1 // last position can reach the end index
	for i := len(nums) - 1; i >= 0; i-- {
		if i+nums[i] >= lastPos {
			lastPos = i
		}
	}
	return lastPos == 0
}

func canJumpV5(nums []int) bool {
	memo := make([]int, len(nums))
	memo[len(nums)-1] = 1 // when index == len(nums)-1, able to reach the last index
	for i := len(nums) - 2; i >= 0; i-- {
		furthestJump := min(i+nums[i], len(nums)-1)
		for j := i + 1; j <= furthestJump; j++ {
			if memo[j] == 1 {
				memo[i] = 1
				break
			}
		}
	}
	return memo[0] == 1
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 56. Merge Intervals
func mergeIntervals(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	var res [][]int
	start := intervals[0][0]
	end := intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] <= end {
			if intervals[i][1] > end {
				end = intervals[i][1]
			}
		} else {
			res = append(res, []int{start, end})
			start = intervals[i][0]
			end = intervals[i][1]
		}
	}
	return res
}

// 62. Unique Paths
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)

	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 || j == 0 {
				dp[i][j] = 1
			} else {
				dp[i][j] = dp[i][j-1] + dp[i-1][j]
			}
		}
	}
	return dp[m-1][n-1]
}

func uniquePathsV2(m int, n int) int {
	left := 0
	upRow := make([]int, n)
	path := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 || j == 0 {
				left = 1
				path = 1
				upRow[j] = 1
			} else {
				path = left + upRow[j]
				upRow[j] = path
				left = path
			}
		}
	}
	return path
}

// 73. Set Matrix Zeroes
func setZeroes(matrix [][]int) {
	row := len(matrix)
	column := len(matrix[0])
	var isCZero bool
	for i := 0; i < row; i++ {
		if matrix[i][0] == 0 {
			isCZero = true
		}
		for j := 1; j < column; j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}
	for i := 1; i < row; i++ {
		for j := 1; j < column; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}

	if matrix[0][0] == 0 {
		for i := 0; i < column; i++ {
			matrix[0][i] = 0
		}
	}

	if isCZero {
		for i := 0; i < row; i++ {
			matrix[i][0] = 0
		}
	}

}

func setZeroesV2(matrix [][]int) {
	isSetFirsColumn := false
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if matrix[i][j] == 0 {
				if j == 0 {
					isSetFirsColumn = true
					continue
				}
				//设置行
				matrix[i][0] = 0
				//设置列
				matrix[0][j] = 0
			}
		}
	}
	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[i]); j++ {
			if matrix[i][0] == 0 {
				matrix[i][j] = 0
			}
			if matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}

	if matrix[0][0] == 0 {
		for j := 0; j < len(matrix[0]); j++ {
			matrix[0][j] = 0
		}
	}

	if isSetFirsColumn {
		for i := 0; i < len(matrix); i++ {
			matrix[i][0] = 0
		}
	}

}

// 75 Sort Colors
func sortColors(nums []int) {
	lhs := 0
	rhs := len(nums) - 1
	i := 0
	for i <= rhs {
		if nums[i] == 1 {
			i++
		} else if nums[i] < 1 {
			nums[i], nums[lhs] = nums[lhs], nums[i]
			i++
			lhs++
		} else {
			nums[i], nums[rhs] = nums[rhs], nums[i]
			rhs--
		}
	}
}

func sortColorsV2(nums []int) {
	zeroIndex := 0
	twoIndex := len(nums) - 1
	curIndex := 0
	for curIndex <= twoIndex {
		for nums[curIndex] == 2 && twoIndex >= curIndex {
			nums[curIndex], nums[twoIndex] = nums[twoIndex], nums[curIndex]
			twoIndex--
		}
		if nums[curIndex] == 0 {
			nums[curIndex], nums[zeroIndex] = nums[zeroIndex], nums[curIndex]
			zeroIndex++
		}
		curIndex++
	}
}

//[78]Subsets	7,293.2%	Medium	0.0%

func subsets(nums []int) [][]int {
	var res [][]int
	subsetsHelp([]int{}, &res, 0, nums)
	return res
}

func subsetsHelp(baseNums []int, res *[][]int, next int, nums []int) {
	*res = append(*res, baseNums)
	if next == len(nums) {
		return
	}

	for i := next; i < len(nums); i++ {
		tmp := make([]int, len(baseNums))
		copy(tmp, baseNums)
		tmp = append(tmp, nums[i])
		subsetsHelp(tmp, res, i+1, nums)
	}
}

func subsetsV2(nums []int) [][]int {
	res := [][]int{[]int{}}
	for _, v := range nums {
		lth := len(res)
		for i := 0; i < lth; i++ {
			tmp := make([]int, len(res[i]))
			copy(tmp, res[i])
			tmp = append(tmp, v)
			res = append(res, tmp)
		}
	}
	return res
}

func subsetsV3(nums []int) [][]int {
	var ret [][]int
	var dfs func(index int, subNums []int)
	dfs = func(index int, subNums []int) {
		ret = append(ret, append([]int{}, subNums...))
		if index == len(nums) {
			return
		}
		for i := index; i < len(nums); i++ {
			subNums = append(subNums, nums[i])
			dfs(i+1, subNums)
			subNums = subNums[:len(subNums)-1]
		}
	}
	dfs(0, []int{})
	return ret
}

// 79. Word Search
func existV1(board [][]byte, word string) bool {
	row := len(board)
	col := len(board[0])
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			if existHelper(board, word, i, j, 0) {
				return true
			}
		}
	}
	return false
}

func existHelper(board [][]byte, word string, i, j, matchCnt int) bool {
	if i < 0 || j < 0 || i >= len(board) || j >= len(board[0]) {
		return false
	}

	if board[i][j] == word[matchCnt] {
		if matchCnt == len(word)-1 {
			return true
		}
		temp := board[i][j]
		board[i][j] = 0
		found := existHelper(board, word, i+1, j, matchCnt+1) || existHelper(board, word, i, j+1, matchCnt+1) || existHelper(board, word, i-1, j, matchCnt+1) || existHelper(board, word, i, j-1, matchCnt+1)
		board[i][j] = temp
		return found
	}
	return false
}

func existV2(board [][]byte, word string) bool {
	for x := 0; x < len(board); x++ {
		for y := 0; y < len(board[0]); y++ {
			if board[x][y] == word[0] && dfs(board, word[1:], x, y) {
				return true
			}
		}
	}
	return false
}

func dfs(board [][]byte, word string, x, y int) bool {
	if len(word) == 0 {
		return true
	}

	tmp := board[x][y]
	board[x][y] = '0'
	if x-1 >= 0 && board[x-1][y] == word[0] && dfs(board, word[1:], x-1, y) {
		return true
	}
	if y-1 >= 0 && board[x][y-1] == word[0] && dfs(board, word[1:], x, y-1) {
		return true
	}
	if x+1 < len(board) && board[x+1][y] == word[0] && dfs(board, word[1:], x+1, y) {
		return true
	}
	if y+1 < len(board[0]) && board[x][y+1] == word[0] && dfs(board, word[1:], x, y+1) {
		return true
	}
	board[x][y] = tmp
	return false
}

func existV3(board [][]byte, word string) bool {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == word[0] {
				if len(word) == 1 {
					return true
				}
				if existHelp(board, i, j, word) {
					return true
				}
			}
		}
	}
	return false

}

func existHelp(board [][]byte, row, column int, word string) bool {
	if len(word) == 0 {
		return true
	}
	tmp := board[row][column]
	if board[row][column] == word[0] {
		board[row][column] = '0'
		//上
		if row > 0 && existHelp(board, row-1, column, word[1:]) {
			return true
		}
		//下
		if row < len(board)-1 && existHelp(board, row+1, column, word[1:]) {
			return true
		}
		//左
		if column > 0 && existHelp(board, row, column-1, word[1:]) {
			return true
		}
		//右
		if column < len(board[0])-1 && existHelp(board, row, column+1, word[1:]) {
			return true
		}
		board[row][column] = tmp
	}
	return false
}

func exist(board [][]byte, word string) bool {
	var dfs func(i, j, searchIndex int) bool
	dfs = func(i, j, searchIndex int) bool {
		if searchIndex == len(word) {
			return true
		}
		if i < 0 || j < 0 || i >= len(board) || j >= len(board[0]) {
			return false
		}
		if board[i][j] == word[searchIndex] {
			tmp := board[i][j]
			board[i][j] = '0'
			up := dfs(i-1, j, searchIndex+1)
			down := dfs(i+1, j, searchIndex+1)
			left := dfs(i, j-1, searchIndex+1)
			right := dfs(i, j+1, searchIndex+1)
			board[i][j] = tmp
			return up || down || left || right
		}
		return false
	}

	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] == word[0] {
				//iter := make(map[int]bool)
				if dfs(i, j, 0) {
					return true
				}
			}
		}
	}
	return false
}

// 91. Decode Ways
func numDecodings(s string) int {
	if s[0] == '0' {
		return 0
	}
	lth := len(s)
	dp := make([]int, lth+1)
	dp[0] = 1
	dp[1] = 1
	for i := 1; i < lth; i++ {
		val := (s[i-1]-'0')*10 + (s[i] - '0')
		if s[i] >= '1' && s[i] <= '9' && val >= 10 && val <= 26 {
			dp[i+1] = dp[i] + dp[i-1]
		} else if s[i] >= '1' && s[i] <= '9' && (val < 10 || val > 26) {
			dp[i+1] = dp[i]
		} else if s[i] == '0' && val >= 10 && val <= 26 {
			dp[i+1] = dp[i-1]
		} else {
			return 0
		}
	}
	return dp[lth]

}

func numDecodingsV2(s string) int {
	lth := len(s)
	if s[0] == '0' {
		return 0
	}
	dp := make([]int, lth+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i < lth+1; i++ {
		dp[i] = 0
		if s[i-1] > '0' {
			dp[i] = dp[i-1]
		}
		if s[i-2] == '1' || (s[i-2] == '2' && s[i-1] < '7') {
			dp[i] += dp[i-2]
		}
	}
	return dp[lth]
}

func numDecodingsV3(s string) int {
	if s[0] == '0' {
		return 0
	}
	dp := make([]int, len(s)+1)
	dp[0] = 1
	dp[1] = 1
	for i := 1; i < len(s); i++ {
		if s[i] != '0' {
			dp[i+1] += dp[i]
		}
		if s[i-1] != '0' {
			num, _ := strconv.Atoi(s[i-1 : i+1])
			if num <= 26 {
				dp[i+1] += dp[i-1]
			}
		}
	}
	return dp[len(s)]
}

func numDecodingsV4(s string) int {
	if s[0] == '0' {
		return 0
	}
	preTwo := 1
	preOne := 1
	cur := 1
	for i := 1; i < len(s); i++ {
		cur = 0
		if s[i] != '0' {
			cur += preOne
		}
		if s[i-1] != '0' {
			num, _ := strconv.Atoi(s[i-1 : i+1])
			if num <= 26 {
				cur += preTwo
			}
		}
		preTwo, preOne = preOne, cur

	}
	return cur
}

// 98. Validate Binary Search Tree
func isValidBST(root *TreeNode) bool {
	return RecValidate(root, nil, nil)
}

func RecValidate(node, min, max *TreeNode) bool {
	if node == nil {
		return true
	}
	if min != nil && node.Val <= min.Val {
		return false
	}
	if max != nil && node.Val >= max.Val {
		return false
	}
	return RecValidate(node.Left, min, node) && RecValidate(node.Right, node, max)
}

// 102. Binary Tree Level Order Traversal
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int
	var stack []*TreeNode
	stack = append(stack, root)

	for len(stack) > 0 {
		lth := len(stack)
		var tmp []int
		for i := 0; i < lth; i++ {
			tmp = append(tmp, stack[i].Val)
			if stack[i].Left != nil {
				stack = append(stack, stack[i].Left)
			}
			if stack[i].Right != nil {
				stack = append(stack, stack[i].Right)
			}
		}
		res = append(res, tmp)
		stack = stack[lth:]
	}

	return res
}

func levelOrderV2(root *TreeNode) [][]int {
	result := [][]int{}
	return traceTree(root, 0, result)
}

func traceTree(root *TreeNode, level int, result [][]int) [][]int {
	if root == nil {
		return result
	}
	if len(result) < level+1 {
		result = append(result, []int{})
	}
	result[level] = append(result[level], root.Val)
	result = traceTree(root.Left, level+1, result)
	result = traceTree(root.Right, level+1, result)
	return result
}

func levelOrderV3(root *TreeNode) [][]int {
	var ret [][]int
	var dfs func(node *TreeNode, level int)
	dfs = func(node *TreeNode, level int) {
		if node == nil {
			return
		}
		if len(ret) < level+1 {
			ret = append(ret, []int{})
		}
		ret[level] = append(ret[level], node.Val)
		dfs(node.Left, level+1)
		dfs(node.Right, level+1)
	}
	dfs(root, 0)
	return ret
}

// 103. Binary Tree Zigzag Level Order Traversal
func zigzagLevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int
	var stack []*TreeNode
	stack = append(stack, root)
	cnt := 0
	for len(stack) > 0 {
		lth := len(stack)
		var levelVals []int
		for i := 0; i < lth; i++ {
			levelVals = append(levelVals, stack[i].Val)
			if stack[i].Left != nil {
				stack = append(stack, stack[i].Left)
			}
			if stack[i].Right != nil {
				stack = append(stack, stack[i].Right)
			}
		}
		if cnt&1 == 1 {
			lhs := 0
			rhs := len(levelVals) - 1
			for lhs < rhs {
				levelVals[lhs], levelVals[rhs] = levelVals[rhs], levelVals[lhs]
				lhs++
				rhs--
			}
		}
		res = append(res, levelVals)
		stack = stack[lth:]
		cnt++
	}
	return res
}

func zigzagLevelOrderV2(root *TreeNode) [][]int {
	res := [][]int{}
	return zigzagLevelOrderHelp(root, 0, res)
}

func zigzagLevelOrderHelp(root *TreeNode, level int, res [][]int) [][]int {
	if root == nil {
		return res
	}
	if len(res) < level+1 {
		res = append(res, []int{})
	}
	if level&1 == 1 {
		res[level] = append([]int{root.Val}, res[level]...)
	} else {
		res[level] = append(res[level], root.Val)
	}
	res = zigzagLevelOrderHelp(root.Left, level+1, res)
	res = zigzagLevelOrderHelp(root.Right, level+1, res)
	return res
}

func zigzagLevelOrderV3(root *TreeNode) [][]int {
	var ret [][]int
	var dfs func(node *TreeNode, level int)
	dfs = func(node *TreeNode, level int) {
		if node == nil {
			return
		}
		if len(ret) < level+1 {
			ret = append(ret, []int{})
		}
		if level&1 == 1 {
			ret[level] = append([]int{node.Val}, ret[level]...)
		} else {
			ret[level] = append(ret[level], node.Val)
		}
		dfs(node.Left, level+1)
		dfs(node.Right, level+1)
	}
	dfs(root, 0)
	return ret
}

// 105. Construct Binary Tree from Preorder and Inorder Traversal
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	index := targetIndex(inorder, preorder[0])
	leftLength := len(inorder[:index])
	//rightLength := len(inorder[index+1:])
	root := &TreeNode{
		Val:   preorder[0],
		Left:  buildTree(preorder[1:leftLength+1], inorder[:index]),
		Right: buildTree(preorder[leftLength+1:], inorder[index+1:]),
	}
	return root
}

func targetIndex(nums []int, target int) int {
	for i := 0; i < len(nums); i++ {
		if target == nums[i] {
			return i
		}
	}
	return -1
}

// Definition for a Node.
type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

// 116. Populating Next Right Pointers in Each Node
func connect(root *Node) *Node {
	if root == nil {
		return nil
	}
	var stack []*Node
	stack = append(stack, root)
	for len(stack) > 0 {
		lth := len(stack)
		for i := 0; i < lth; i++ {
			if i < lth-1 {
				stack[i].Next = stack[i+1]
			}

			if stack[i].Left != nil {
				stack = append(stack, stack[i].Left)
			}
			if stack[i].Right != nil {
				stack = append(stack, stack[i].Right)
			}
		}
		stack = stack[lth:]
	}
	return root
}

func connectV2(root *Node) *Node {
	if root == nil {
		return nil
	}

	if root.Left != nil {
		root.Left.Next = root.Right
		if root.Next != nil {
			root.Right.Next = root.Next.Left
		}
	}

	connectV2(root.Left)
	connectV2(root.Right)

	return root
}

// 122. Best Time to Buy and Sell Stock II
func maxProfitII(prices []int) int {
	maxP := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			maxP += prices[i] - prices[i-1]
		}
	}
	return maxP
}

func maxProfitIIV2(prices []int) int {
	lth := len(prices)
	dp := make([][]int, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]int, 2)
	}
	//持有
	dp[0][0] = -prices[0]
	//未持有
	dp[0][1] = 0
	for i := 1; i < lth; i++ {
		dp[i][0] = max(dp[i-1][1]-prices[i], dp[i-1][0])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return max(dp[lth-1][0], dp[lth-1][1])
}

func maxProfitIIV3(prices []int) int {
	lth := len(prices)
	hold := -prices[0]
	noHold := 0
	for i := 1; i < lth; i++ {
		hold = max(hold, noHold-prices[i])
		noHold = max(noHold, hold+prices[i])
	}
	return noHold
}

func longestConsecutive(nums []int) int {
	child2father := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		child2father[nums[i]] = nums[i]
	}
	for k, _ := range child2father {
		if _, ok := child2father[k+1]; ok {
			child2father[k] = child2father[k+1]
		}
	}

	for c, f := range child2father {
		father := f
		for {
			if v, ok := child2father[father]; ok && v != father {
				child2father[c] = child2father[father]
				father = v
			} else {
				break
			}
		}
	}
	maxC := 0
	for c, f := range child2father {
		if f-c+1 > maxC {
			maxC = f - c + 1
		}
	}
	return maxC
}

func longestConsecutiveV2(nums []int) int {
	record := make(map[int]bool)
	for _, num := range nums {
		record[num] = true
	}
	maxC := 0
	for _, num := range nums {
		if _, ok := record[num-1]; !ok {
			curNum := num + 1
			tmpCon := 1
			for {
				if _, tmpOk := record[curNum]; tmpOk {
					curNum++
					tmpCon++
				} else {
					break
				}
			}
			if tmpCon > maxC {
				maxC = tmpCon
			}
		}

	}
	return maxC
}

type Dsu struct {
	parent map[int]int
}

func NewDsu(nums []int) *Dsu {
	parent := make(map[int]int)
	for _, num := range nums {
		parent[num] = num
	}
	return &Dsu{parent: parent}
}

func (d *Dsu) find(child int) int {
	if father, ok := d.parent[child]; ok && child != father {
		d.parent[child] = d.find(father)
	}
	return d.parent[child]
}

func (d *Dsu) connect(child int, father int) {
	d.parent[child] = d.find(father)
}

func longestConsecutiveV3(nums []int) int {
	dsu := NewDsu(nums)
	for _, num := range nums {
		if _, ok := dsu.parent[num-1]; ok {
			dsu.connect(num-1, num)
		}
		if _, ok := dsu.parent[num+1]; ok {
			dsu.connect(num, num+1)
		}
	}
	for _, num := range nums {
		dsu.find(num)
	}
	max := 0
	for child, father := range dsu.parent {
		if father-child+1 > max {
			max = father - child + 1
		}
	}
	return max
}

// 130. Surrounded Regions
func solve(board [][]byte) {
	row := len(board)
	column := len(board[0])
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if i == 0 || j == 0 || i == row-1 || j == column-1 {
				if board[i][j] == 'O' {
					solveHelp(board, i, j)
				}
			}
		}
	}

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if board[i][j] == '#' {
				board[i][j] = 'O'
			} else if board[i][j] == 'O' {
				board[i][j] = 'X'
			}

		}
	}
}

func solveHelp(board [][]byte, i, j int) {
	if i < 0 || j < 0 || i >= len(board) || j >= len(board[0]) {
		return
	}
	if board[i][j] == 'O' {
		board[i][j] = '#'
		solveHelp(board, i-1, j)
		solveHelp(board, i+1, j)
		solveHelp(board, i, j-1)
		solveHelp(board, i, j+1)
	}
	return
}

type UF struct {
	Parent []int
	Rank   []int //rank[i] = rank of subtree rooted at i
	Count  int
}

func NewUf(n int) *UF {
	parent := make([]int, n)
	rank := make([]int, n)
	for i := 0; i < n; i++ {
		parent[i] = i
		rank[i] = 0
	}
	count := n
	return &UF{
		Parent: parent,
		Rank:   rank,
		Count:  count,
	}
}

func (u *UF) find(p int) int {
	for p != u.Parent[p] {
		u.Parent[p] = u.Parent[u.Parent[p]]
		p = u.Parent[p]
	}
	return p
}

func (u *UF) connect(p int, q int) {
	pp := u.find(p)
	qp := u.find(q)
	if pp == qp {
		return
	}
	if u.Rank[pp] < u.Rank[qp] {
		u.Parent[pp] = qp
	} else if u.Rank[pp] > u.Rank[qp] {
		u.Parent[qp] = pp
	} else {
		u.Parent[qp] = pp
		u.Rank[pp]++
	}
	u.Count--
}

func (u *UF) isConnect(p int, q int) bool {
	return u.find(p) == u.find(q)
}

func solveV2(board [][]byte) {
	row := len(board)
	if row == 0 {
		return
	}
	column := len(board[0])
	uf := NewUf(row*column + 1)
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if (i == 0 || j == 0 || i == row-1 || j == column-1) && board[i][j] == 'O' {
				uf.connect(i*column+j, row*column)
			} else if board[i][j] == 'O' {
				if board[i-1][j] == 'O' {
					uf.connect(i*column+j, (i-1)*column+j)
				}
				if board[i+1][j] == 'O' {
					uf.connect(i*column+j, (i+1)*column+j)
				}
				if board[i][j-1] == 'O' {
					uf.connect(i*column+j, i*column+j-1)
				}
				if board[i][j+1] == 'O' {
					uf.connect(i*column+j, i*column+j+1)
				}

			}
		}
	}

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if board[i][j] == 'O' && !uf.isConnect(i*column+j, row*column) {
				board[i][j] = 'X'
			}
		}
	}
}

func solveV3(board [][]byte) {
	row := len(board)
	column := len(board[0])

	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || j < 0 || i >= row || j >= column {
			return
		}
		if board[i][j] == 'O' {
			board[i][j] = '#'
			dfs(i-1, j)
			dfs(i+1, j)
			dfs(i, j-1)
			dfs(i, j+1)
		}

	}

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if i == 0 || j == 0 || j == column-1 || i == row-1 {
				if board[i][j] == 'O' {
					dfs(i, j)
				}
			}
		}
	}

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if board[i][j] == '#' {
				board[i][j] = 'O'
			} else if board[i][j] == 'O' {
				board[i][j] = 'X'
			}
		}
	}

}

// 131. Palindrome Partitioning
func partition(s string) [][]string {
	var list []string
	var res [][]string
	start := 0
	for i := start; i < len(s); i++ {
		if isPal(s[start : i+1]) {
			partitionRec(s, s[start:i+1], i, list, &res)
		}

	}

	return res
}

func partitionRec(s string, substr string, end int, list []string, res *[][]string) {
	if !isPal(substr) {
		return
	}
	list = append(list, substr)
	if end == len(s)-1 {
		*res = append(*res, list)
		return
	}

	for i := end + 1; i < len(s); i++ {
		tmp := make([]string, len(list))
		copy(tmp, list)
		partitionRec(s, s[end+1:i+1], i, tmp, res)
	}

}

func partitionV4(s string) [][]string {
	var ret [][]string
	var dfs func(index int, combination []string)

	dfs = func(index int, combination []string) {
		if index == len(s) {
			ret = append(ret, append([]string{}, combination...))
			return
		}

		if index < len(s) {
			for i := index; i < len(s); i++ {
				str := s[index : i+1]
				if isPal(str) {
					combination = append(combination, str)
					dfs(i+1, combination)
					combination = combination[:len(combination)-1]
				}
			}
		}
	}
	dfs(0, []string{})
	return ret
}

func partitionV2(s string) [][]string {
	dp := make([][][]string, len(s))
	dp[0] = append(dp[0], []string{string(s[0])})
	for i := 1; i < len(s); i++ {
		for j := i; j >= 0; j-- {
			if isPal(s[j : i+1]) {
				if j-1 >= 0 {
					for _, list := range dp[j-1] {
						temp := make([]string, len(list))
						copy(temp, list)
						temp = append(temp, s[j:i+1])
						dp[i] = append(dp[i], temp)
					}
				}
				if j == 0 {
					dp[i] = append(dp[i], []string{s[j : i+1]})
				}

			}
		}

	}
	return dp[len(s)-1]
}

func partitionV3(s string) [][]string {
	var res [][]string
	partitionV3Help(&res, 0, []string{}, s)
	return res
}

func partitionV3Help(res *[][]string, index int, palindromeList []string, s string) {
	if index == len(s) {
		tmp := make([]string, len(palindromeList))
		copy(tmp, palindromeList)
		*res = append(*res, tmp)
		return
	}
	for i := index; i < len(s); i++ {
		if isPal(s[index : i+1]) {
			palindromeList = append(palindromeList, s[index:i+1])
			partitionV3Help(res, i+1, palindromeList, s)
			palindromeList = palindromeList[:len(palindromeList)-1]
		}
	}
}

func isPal(str string) bool {
	lhs := 0
	rhs := len(str) - 1
	for lhs < rhs {
		if str[lhs] != str[rhs] {
			return false
		}
		lhs++
		rhs--
	}
	return true
}

// 134. Gas Station
func canCompleteCircuit(gas []int, cost []int) int {
	curSum := 0
	totalSum := 0
	start := 0

	for i := 0; i < len(gas); i++ {
		curSum += gas[i] - cost[i]
		totalSum += gas[i] - cost[i]

		if curSum < 0 {
			start = i + 1
			curSum = 0
		}
	}

	if totalSum < 0 {
		return -1
	}

	return start
}

type NodeRandom struct {
	Val    int
	Next   *NodeRandom
	Random *NodeRandom
}

// 138. Copy List with Random Pointer
func copyRandomList(head *NodeRandom) *NodeRandom {
	if head == nil {
		return nil
	}
	src2copy := make(map[*NodeRandom]*NodeRandom)
	copyHead := new(NodeRandom)
	copyNode := copyHead
	cur := head
	for cur != nil {
		copyNode.Next = &NodeRandom{
			Val:    cur.Val,
			Next:   nil,
			Random: nil,
		}
		src2copy[cur] = copyNode.Next
		cur = cur.Next
		copyNode = copyNode.Next
	}
	cur = head
	for cur != nil {
		src2copy[cur].Random = src2copy[cur.Random]
		cur = cur.Next
	}
	return copyHead.Next
}

// 139. Word Break
func wordBreak(s string, wordDict []string) bool {
	lth := len(s)
	dp := make([]bool, lth)
	word2exist := make(map[string]bool)
	for _, word := range wordDict {
		word2exist[word] = true
	}
	for i := 0; i < lth; i++ {
		for j := i; j >= 0; j-- {
			substr := s[j : i+1]
			if word2exist[substr] {
				if j == 0 {
					dp[i] = true
				} else {
					if dp[j-1] {
						dp[i] = true
					}

				}
			}
		}
	}
	return dp[lth-1]

}

// 146. LRU Cache
type LRUCache struct {
	key2Node [10000]*LRUNode
	len      int
	cap      int
	head     *LRUNode
	tail     *LRUNode
}

type LRUNode struct {
	key  int
	val  int
	pre  *LRUNode
	next *LRUNode
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		key2Node: [10000]*LRUNode{},
		len:      0,
		cap:      capacity,
		head:     nil,
		tail:     nil,
	}

}

func (this *LRUCache) Get(key int) int {
	node := this.key2Node[key]
	if node == nil {
		return -1
	}

	val := this.key2Node[key].val
	//将此节点移到链表末尾
	if this.len > 1 {
		this.Put(key, val)
	}
	return val
}

func (this *LRUCache) Put(key int, value int) {
	if this.len == 0 {
		node := &LRUNode{
			key:  key,
			val:  value,
			pre:  nil,
			next: nil,
		}
		this.key2Node[key] = node
		this.head = node
		this.tail = node
		this.len++
		return
	}
	node := this.key2Node[key]
	if node != nil {
		node.val = value
		if node != this.tail {
			if node == this.head {
				next := node.next
				next.pre = nil
				this.head = next
			} else {
				//此节点拆开，放到尾部
				preNode := node.pre
				nextNode := node.next
				preNode.next = nextNode
				nextNode.pre = preNode
			}
			//添加到尾部
			this.tail.next = node
			node.pre = this.tail
			node.next = nil
			this.tail = node
		}
	} else {
		newNode := &LRUNode{
			key:  key,
			val:  value,
			pre:  this.tail,
			next: nil,
		}
		this.key2Node[key] = newNode
		if this.len == this.cap {
			//驱逐表头
			this.key2Node[this.head.key] = nil
			if this.cap == 1 {
				this.head = newNode
				this.tail = newNode
				newNode.pre = nil
				return
			}
			nextNode := this.head.next
			nextNode.pre = nil
			this.head = nextNode
			//放在表尾
			this.tail.next = newNode
			this.tail = newNode
			return
		}
		//将此节点放在表尾
		this.tail.next = newNode
		this.tail = newNode
		this.len++
	}

}

type LRUCacheV2 struct {
	head     *LRUNodeV2
	tail     *LRUNodeV2
	key2node map[int]*LRUNodeV2
	capacity int
	lth      int
}

type LRUNodeV2 struct {
	pre  *LRUNodeV2
	next *LRUNodeV2
	val  int
	key  int
}

func ConstructorLRU(capacity int) LRUCacheV2 {
	return LRUCacheV2{
		head:     nil,
		tail:     nil,
		key2node: make(map[int]*LRUNodeV2),
		capacity: capacity,
		lth:      0,
	}
}

func (this *LRUCacheV2) Get(key int) int {
	if node, ok := this.key2node[key]; ok {
		this.Put(key, node.val)
		return node.val
	}
	return -1
}

func (this *LRUCacheV2) Put(key int, value int) {
	//初始化
	if this.lth == 0 || (this.capacity == 1 && this.lth == 1) {
		lruNode := &LRUNodeV2{
			pre:  nil,
			next: nil,
			key:  key,
			val:  value,
		}
		this.head, this.tail = lruNode, lruNode
		this.key2node[key] = lruNode
		this.lth++
		return
	}

	//key 存在
	if node, ok := this.key2node[key]; ok {
		node.val = value
		//是表头
		if node == this.head {
			return
		}
		//是表尾
		if node == this.tail {
			this.tail = node.pre
			node.pre.next = nil
			node.pre = nil
			this.head.pre = node
			node.next = this.head
			this.head = node
			return
		}
		//在中间移动到表头
		node.pre.next = node.next
		node.next.pre = node.pre
		node.pre = nil
		this.head.pre = node
		node.next = this.head
		this.head = node
	} else {
		//不存在
		lruNode := &LRUNodeV2{
			pre:  nil,
			next: nil,
			val:  value,
			key:  key,
		}
		lruNode.pre = nil
		this.head.pre = lruNode
		lruNode.next = this.head
		this.head = lruNode
		this.key2node[key] = lruNode
		if this.lth < this.capacity {
			this.lth++
		} else {
			//驱逐表尾
			delete(this.key2node, this.tail.key)
			this.tail = this.tail.pre
			this.tail.next = nil
		}
	}

}

// 超时 148. Sort List
func sortList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	lth := 0
	cur := head
	for cur != nil {
		lth++
		cur = cur.Next
	}
	for i := 0; i <= lth; i++ {
		for j, cur := 0, head; j < lth-i && cur != nil; j, cur = j+1, cur.Next {
			if cur.Val > cur.Next.Val {
				cur.Val, cur.Next.Val = cur.Next.Val, cur.Val
			}
		}
	}
	return head
}

func sortListV2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	slow, fast := head, head.Next
	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	rightHead := slow.Next
	slow.Next = nil

	return mergeSortList(sortListV2(head), sortListV2(rightHead))
}

func mergeSortList(list1 *ListNode, list2 *ListNode) *ListNode {
	result := &ListNode{Val: 0}
	current := result
	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			current.Next = list1
			list1 = list1.Next
		} else {
			current.Next = list2
			list2 = list2.Next
		}
		current = current.Next
	}

	if list1 != nil {
		current.Next = list1
	}
	if list2 != nil {
		current.Next = list2
	}

	return result.Next
}

func sortListV3(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	rightHead := slow.Next
	slow.Next = nil

	return mergeList(sortList(head), sortList(rightHead))

}

func sortListV4(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := spiltList(head)
	return mergeList(sortList(slow), sortList(fast))
}

func mergeList(l1, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	cur := dummy
	for l1 != nil || l2 != nil {
		if (l1 == nil) || (l2 != nil && l2.Val < l1.Val) {
			cur.Next = l2
			l2 = l2.Next
		} else {
			cur.Next = l1
			l1 = l1.Next
		}
		cur = cur.Next
	}
	return dummy.Next

}

func spiltList(l *ListNode) (*ListNode, *ListNode) {
	slow, fast := l, l.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	rightHead := slow.Next
	slow.Next = nil
	return l, rightHead
}

// 150. Evaluate Reverse Polish Notation
func evalRPN(tokens []string) int {
	var numStack []int
	for i := 0; i < len(tokens); i++ {
		if tokens[i] != "+" && tokens[i] != "-" && tokens[i] != "*" && tokens[i] != "/" {
			num, _ := strconv.Atoi(tokens[i])
			numStack = append(numStack, num)
		} else {
			lth := len(numStack)
			num1 := numStack[lth-1]
			num2 := numStack[lth-2]
			var tmp int
			if tokens[i] == "+" {
				tmp = num1 + num2
			} else if tokens[i] == "-" {
				tmp = num2 - num1
			} else if tokens[i] == "*" {
				tmp = num1 * num2
			} else if tokens[i] == "/" {
				tmp = num2 / num1
			}
			numStack = numStack[:lth-2]
			numStack = append(numStack, tmp)
		}
	}
	return numStack[0]

}

// 152. Maximum Product Subarray
func maxProduct(nums []int) int {
	res, maxP, minP := nums[0], nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		mxp := maxP
		mnp := minP
		maxP = max(max(mxp*nums[i], mnp*nums[i]), nums[i])
		minP = min(min(mxp*nums[i], mnp*nums[i]), nums[i])
		res = max(maxP, res)
	}
	return res
}

// 155. Min Stack
type MinStack struct {
	data []int
	mins []int
}

func ConstructorMinStack() MinStack {
	return MinStack{
		data: []int{},
		mins: []int{},
	}
}

func (this *MinStack) Push(x int) {
	this.data = append(this.data, x)
	newmin := x
	if len(this.mins) > 0 {
		if oldmin := this.GetMin(); oldmin < x {
			newmin = oldmin
		}
	}
	this.mins = append(this.mins, newmin)
}

func (this *MinStack) Pop() {
	this.data = this.data[:len(this.data)-1]
	this.mins = this.mins[:len(this.mins)-1]
}

func (this *MinStack) Top() int {
	return this.data[len(this.data)-1]
}

func (this *MinStack) GetMin() int {
	return this.mins[len(this.mins)-1]
}

// 162. Find Peak Element
func findPeakElement(nums []int) int {
	maxIndex := 0
	maxNum := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] > maxNum {
			maxNum = nums[i]
			maxIndex = i

		}
	}
	return maxIndex

}

func findPeakElementV2(nums []int) int {
	lhs, rhs := 0, len(nums)-1
	for lhs < rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] > nums[mid+1] {
			rhs = mid
		} else {
			lhs = mid + 1
		}
	}
	return lhs

}

// 300. Longest Increasing Subsequencemj
func lengthOfLIS(nums []int) int {
	lth := len(nums)
	dp := make([][]int, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]int, lth)
	}
	dp[0][0] = 1
	maxSub := 1
	for i := 1; i < lth; i++ {
		dp[i][i] = 1
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i][j] = dp[j][j] + 1
			} else if nums[i] == nums[j] {
				dp[i][j] = dp[j][j]
			} else {
				dp[i][j] = 1
			}
			dp[i][i] = max(dp[i][i], dp[i][j])
		}
		maxSub = max(maxSub, dp[i][i])
	}
	return maxSub

}

func lengthOfLISV2(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	incNums := make([]int, len(nums))
	incNums[0] = nums[0]
	incLth := 0
	for i := 1; i < len(nums); i++ {
		idx := biSearchIdx(incNums, nums[i], 0, incLth)
		incNums[idx] = nums[i]
		if idx == incLth+1 {
			incLth++
		}
	}

	return incLth + 1
}

func biSearchIdx(incNums []int, target, start, incLth int) int {
	for start <= incLth {
		mid := (incLth + start) / 2
		if incNums[mid] > target {
			incLth = mid - 1
		} else if incNums[mid] < target {
			start = mid + 1
		} else {
			return mid
		}
	}
	return start
}

func lengthOfLISV3(nums []int) int {
	dp := make([]int, len(nums))
	maxL := 1
	for i := 0; i < len(nums); i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)
			}

		}
		if dp[i] > maxL {
			maxL = dp[i]
		}
	}
	return maxL
}

// 674. Longest Continuous Increasing Subsequence
func findLengthOfLCIS(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 1
	maxIncLth := 1
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			dp[i] = dp[i-1] + 1
		} else {
			dp[i] = 1
		}
		maxIncLth = max(maxIncLth, dp[i])
	}
	return maxIncLth
}

// 166. Fraction to Recurring Decimal
func fractionToDecimal(numerator int, denominator int) string {
	res := ""
	if numerator < 0 && denominator > 0 {
		res += "-"
		numerator *= -1
	}
	if numerator > 0 && denominator < 0 {
		res += "-"
		denominator *= -1
	}
	if numerator < 0 && denominator < 0 {
		numerator *= -1
		denominator *= -1
	}

	preFraction := numerator / denominator
	remain := numerator % denominator
	if remain == 0 {
		return res + strconv.Itoa(preFraction)
	}
	var afterFractionList []int
	remain2index := make(map[int]int)
	cnt := 0
	res += strconv.Itoa(preFraction) + "."
	for {
		remain *= 10
		if index, ok := remain2index[remain]; ok {
			for i := 0; i < len(afterFractionList); i++ {
				if i < index {
					res += strconv.Itoa(afterFractionList[i])
					continue
				} else if i == index {
					res += "("
				}
				res += strconv.Itoa(afterFractionList[i])
			}
			res += ")"
			return res
		} else {
			remain2index[remain] = cnt
			afterFraction := remain / denominator
			remain = remain % denominator
			afterFractionList = append(afterFractionList, afterFraction)
			cnt++
			if remain == 0 {
				break
			}
		}

	}

	for i := 0; i < len(afterFractionList); i++ {
		res += strconv.Itoa(afterFractionList[i])
	}
	return res

}

func fractionToDecimalV2(numerator int, denominator int) string {
	if numerator == 0 {
		return "0"
	}
	sign := 1
	nSign := 1
	if numerator < 0 {
		nSign = 0
		numerator *= -1
	}
	dSign := 1
	if denominator < 0 {
		denominator *= -1
		dSign = 0
	}

	if nSign^dSign == 1 {
		sign = -1
	}

	remain := numerator % denominator
	integerNum := numerator / denominator
	var numsList []int
	numsList = append(numsList, integerNum)
	remain2index := make(map[int]int)
	index := 1
	cycleIndex := 0
	for remain != 0 {
		remain *= 10
		if i, ok := remain2index[remain]; !ok {
			remain2index[remain] = index
			num := remain / denominator
			numsList = append(numsList, num)
			remain %= denominator
			index++
		} else {
			cycleIndex = i
			break
		}
	}

	var sb strings.Builder
	if sign == -1 {
		sb.WriteString("-")
	}
	for i := 0; i < len(numsList); i++ {
		if i == 1 {
			sb.WriteString(".")
		}
		if cycleIndex > 0 && i == cycleIndex {
			sb.WriteString("(")
		}
		sb.WriteString(strconv.Itoa(numsList[i]))
	}

	if cycleIndex > 0 {
		sb.WriteString(")")
	}

	return sb.String()
}

// 172. Factorial Trailing Zeroes
func trailingZeroes(n int) int {
	res := 0
	for n != 0 {
		res += n / 5
		n /= 5
	}
	return res
}

// 179. Largest Number
func largestNumber(nums []int) string {
	var strList []string
	for _, num := range nums {
		str := strconv.Itoa(num)
		strList = append(strList, str)
	}
	sort.Slice(strList, func(i, j int) bool {
		return strList[i]+strList[j] > strList[j]+strList[i]
	})
	if strList[0] == "0" {
		return "0"
	}

	return strings.Join(strList, "")
}

func maxStr(a, b string) string {
	if a > b {
		return a
	}
	return b
}

// 189. Rotate Array
func rotateArr(nums []int, k int) {
	lth := len(nums)
	k %= lth
	help(nums, 0, lth-1)
	help(nums, 0, k-1)
	help(nums, k, lth-1)

}

func rotateArrV2(nums []int, k int) {
	lth := len(nums)
	k %= lth
	for i := 1; i <= k; i++ {
		tmp := nums[len(nums)-1]
		for j := len(nums) - 1; j > 0; j-- {
			nums[j] = nums[j-1]
		}
		nums[0] = tmp
	}

}

func help(nums []int, start, end int) {
	for start < end {
		nums[start], nums[end] = nums[end], nums[start]
		end--
		start++
	}

}

// 236. Lowest Common Ancestor of a Binary Tree
func LowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root == p || root == q {
		return root
	}
	left := LowestCommonAncestor(root.Left, p, q)
	right := LowestCommonAncestor(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left == nil {
		return right
	}
	return left
}

func LowestCommonAncestorV2(root, p, q *TreeNode) *TreeNode {

	var twoNodes []*TreeNode
	child2father := make(map[*TreeNode]*TreeNode)
	var stack []*TreeNode
	stack = append(stack, root)
	child2father[root] = root
	for len(stack) > 0 {
		node := stack[0]
		if node == p || node == q {
			twoNodes = append(twoNodes, node)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
			child2father[node.Left] = node
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
			child2father[node.Right] = node
		}
		stack = stack[1:]
	}
	fatherHash := make(map[*TreeNode]bool)
	high := twoNodes[0]
	fatherHash[high] = true
	for high != child2father[high] {
		fatherHash[child2father[high]] = true
		high = child2father[high]
	}
	low := twoNodes[1]
	var res *TreeNode
	for low != child2father[low] {
		if _, ok := fatherHash[child2father[low]]; ok {
			res = child2father[low]
			break
		}
		low = child2father[low]
	}

	return res

}

// 31. Next Permutation
func nextPermutation(nums []int) {
	i := len(nums) - 2
	for ; i >= 0 && nums[i+1] <= nums[i]; i-- {
	}
	if i >= 0 {
		j := len(nums) - 1
		for ; nums[j] <= nums[i]; j-- {
		}
		nums[i], nums[j] = nums[j], nums[i]
	}
	lhs := i + 1
	rhs := len(nums) - 1
	for lhs < rhs {
		nums[lhs], nums[rhs] = nums[rhs], nums[lhs]
		lhs++
		rhs--
	}
}

// 198. House Robber
func rob(nums []int) int {
	dp := make([][]int, len(nums)+1)
	for i := 0; i <= len(nums); i++ {
		dp[i] = make([]int, len(nums)+1)
	}
	maxMoney := nums[0]
	dp[0][0] = nums[0]
	for i := 1; i < len(nums); i++ {
		dp[i][i] = nums[i]
		for j := 0; j < i-1; j++ {
			dp[i][j] = dp[j][j] + nums[i]
			dp[i][i] = max(dp[i][i], dp[i][j])
		}
		maxMoney = max(maxMoney, dp[i][i])
	}
	return maxMoney
}

func robV2(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	maxMoney := nums[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = nums[i]
		for j := 0; j < i-1; j++ {
			dp[i] = max(dp[i], dp[j]+nums[i])
		}
		maxMoney = max(maxMoney, dp[i])
	}
	return maxMoney
}

func robV3(nums []int) int {
	lth := len(nums)
	if lth == 0 {
		return 0
	}
	if lth == 1 {
		return nums[0]
	}
	dp := make([]int, lth)
	dp[0], dp[1] = nums[0], max(nums[0], nums[1])
	for i := 2; i < lth; i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i])
	}
	return dp[lth-1]
}

// 200. Number of Islands
func numIslands(grid [][]byte) int {
	row := len(grid)
	column := len(grid[0])
	cnt := 0
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if grid[i][j] == '1' {
				cnt++
				numsIslandsHelp(grid, i, j)
			}
		}
	}
	return cnt
}

func numsIslandsHelp(grid [][]byte, i, j int) {
	if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) {
		return
	}
	if grid[i][j] == '1' {
		grid[i][j] = '0'
		//上下
		numsIslandsHelp(grid, i-1, j)
		numsIslandsHelp(grid, i+1, j)
		//左右
		numsIslandsHelp(grid, i, j-1)
		numsIslandsHelp(grid, i, j+1)
	}

}

func numIslandsV2(grid [][]byte) int {
	row := len(grid)
	column := len(grid[0])
	visited := make([]bool, row*column)
	cnt := 0
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if grid[i][j] == '1' && !visited[i*column+j] {
				dfsIsLands(grid, i, j, visited)
				cnt++
			}
		}
	}
	return cnt

}

func dfsIsLands(grid [][]byte, i, j int, visited []bool) {
	row := len(grid)
	column := len(grid[0])

	if i >= 0 && i < row && j >= 0 && j < column && !visited[i*column+j] && grid[i][j] == '1' {
		visited[i*column+j] = true
		dfsIsLands(grid, i-1, j, visited)
		dfsIsLands(grid, i+1, j, visited)
		dfsIsLands(grid, i, j-1, visited)
		dfsIsLands(grid, i, j+1, visited)
	}

}

func numIslandsV3(grid [][]byte) int {
	row := len(grid)
	column := len(grid[0])
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || j < 0 || i >= row || j >= column {
			return
		}
		if grid[i][j] == '1' {
			grid[i][j] = 'X'
			dfs(i+1, j)
			dfs(i-1, j)
			dfs(i, j+1)
			dfs(i, j-1)
		}
	}
	var cnt int
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if grid[i][j] == '1' {
				cnt++
				dfs(i, j)
			}
		}
	}
	return cnt
}

func countPrimes(n int) int {
	if n == 0 || n == 1 {
		return 0
	}
	notPrime := make([]bool, n)
	cnt := 0
	for i := 2; i < n; i++ {
		if notPrime[i] {
			continue
		}
		cnt++
		for j := 2; i*j < n; j++ {
			notPrime[i*j] = true
		}
	}
	return cnt
}

// 1143. Longest Common Subsequence
func longestCommonSubsequence(text1 string, text2 string) int {
	dp := make([][]int, len(text1)+1)
	for i := 0; i <= len(text1); i++ {
		dp[i] = make([]int, len(text2)+1)
	}
	for i := 1; i <= len(text1); i++ {
		for j := 1; j <= len(text2); j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[len(text1)][len(text2)]
}

func canFinish(numCourses int, prerequisites [][]int) bool {
	nodeInNum := make([]int, numCourses)
	father2childes := make(map[int][]int)
	for _, prerequisite := range prerequisites {
		father := prerequisite[1]
		child := prerequisite[0]
		father2childes[father] = append(father2childes[father], child)
		nodeInNum[child]++
	}
	cnt := 0
	var stack []int
	for node, inNum := range nodeInNum {
		if inNum == 0 {
			stack = append(stack, node)
			cnt++
		}
	}

	for len(stack) > 0 {
		node := stack[0]
		for _, child := range father2childes[node] {
			nodeInNum[child]--
			if nodeInNum[child] == 0 {
				stack = append(stack, child)
				cnt++
			}
		}
		stack = stack[1:]
	}
	if cnt == len(nodeInNum) {
		return true
	}

	return false
}

type Trie struct {
	Next   []*Trie
	IsWord bool
}

func Constructorxx() Trie {
	return Trie{
		Next:   make([]*Trie, 26),
		IsWord: false,
	}
}

func (this *Trie) Insert(word string) {
	cur := this
	for i, ch := range word {
		if cur.Next[ch-'a'] == nil {
			cur.Next[ch-'a'] = &Trie{
				Next:   make([]*Trie, 26),
				IsWord: false,
			}
		}
		cur = cur.Next[ch-'a']
		if i == len(word)-1 {
			cur.IsWord = true
		}
	}

}

func (this *Trie) Search(word string) bool {
	cur := this
	for _, ch := range word {
		if cur.Next[ch-'a'] != nil {
			cur = cur.Next[ch-'a']
		} else {
			return false
		}

	}
	return cur.IsWord
}

func (this *Trie) StartsWith(prefix string) bool {
	cur := this
	for _, ch := range prefix {
		if cur.Next[ch-'a'] != nil {
			cur = cur.Next[ch-'a']
		} else {
			return false
		}

	}
	return true
}

//10. Regular Expression Matching

func isMatch(s string, p string) bool {
	s = " " + s
	p = " " + p

	m := len(p)
	n := len(s)

	dp := make([][]bool, m)
	for i := range dp {
		dp[i] = make([]bool, n)
	}

	dp[0][0] = true
	for i := 1; i < m; i++ {
		if p[i] == '*' {
			dp[i][0] = dp[i-2][0]
		}
		for j := 1; j < n; j++ {
			if p[i] == s[j] || p[i] == '.' {
				dp[i][j] = dp[i-1][j-1]
			} else if p[i] == '*' {
				dp[i][j] = dp[i-2][j] || ((p[i-1] == s[j] || p[i-1] == '.') && dp[i][j-1])
			}
		}
	}

	return dp[m-1][n-1]
}

// 41. First Missing Positive
func firstMissingPositive(nums []int) int {
	var numslen = len(nums)
	var i = 0
	for i < numslen {
		if nums[i] > numslen || nums[i] <= 0 || nums[nums[i]-1] == nums[i] {
			i = i + 1
			continue
		}
		nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
	}

	for i := 0; i < numslen; i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return numslen + 1
}

func trap(height []int) int {
	left, right := 0, len(height)-1
	res := 0
	leftMax, rightMax := 0, 0

	for left < right {
		if height[left] < height[right] {
			if height[left] >= leftMax {
				leftMax = height[left]
			} else {
				res += leftMax - height[left]
			}
			left++
		} else {
			if height[right] >= rightMax {
				rightMax = height[right]
			} else {
				res += rightMax - height[right]
			}
			right--
		}
	}

	return res
}

// leetcode296 最佳碰头地点
func minTotalDistance(grid [][]int) int {
	x, y := []int{}, []int{}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				x = append(x, i)
				y = append(y, j)
			}
		}
	}
	sort.Ints(x)
	sort.Ints(y)

	d := 0
	for i := 0; i < len(x)/2; i++ {
		d += x[len(x)-1-i] - x[i]
	}
	for i := 0; i < len(y)/2; i++ {
		d += y[len(y)-1-i] - y[i]
	}
	return d
}

/**
 * Your Trie object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Insert(word);
 * param_2 := obj.Search(word);
 * param_3 := obj.StartsWith(prefix);
 */

func main() {
	tire := Constructorxx()
	tire.Insert("apple")
	fmt.Println(tire)
}
