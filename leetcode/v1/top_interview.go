package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
)

//7. Reverse Integer

func reverse(x int) int {
	num := 0
	for x != 0 {
		mod := x % 10
		num = num*10 + mod
		if num < math.MinInt32 || num > math.MaxInt32 {
			return 0
		}
		x = x / 10
	}

	return num

}

/*
leetcode Top Interview questions
*/

//part 1  easy

// 1. Two Sum
func twoSum(nums []int, target int) []int {
	res := make([]int, 2)
	for i := 0; i < len(nums)-1; i++ {
		for j := i + 1; j < len(nums); j++ {
			if target == nums[i]+nums[j] {
				res[0] = i
				res[1] = j
				return res
			}
		}
	}

	return nil
}

//13. Roman to Integer

func romanToInt(s string) int {
	var res int

	symbol2value := map[string]int{
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
	lth := len(s)
	for i := 0; i < lth; i++ {
		if i < lth-1 {
			cmb := string(s[i]) + string(s[i+1])
			if val, ok := symbol2value[cmb]; ok {
				res += val
				i++
				continue
			}
		}
		if val, ok := symbol2value[string(s[i])]; ok {
			res += val
		}
	}

	return res
}

func romanToIntV2(s string) int {
	var res int

	symbol2value := map[uint8]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}

	lth := len(s)

	for i := 0; i < lth; i++ {
		if i < lth-1 {
			if symbol2value[s[i]] < symbol2value[s[i+1]] {
				res += symbol2value[s[i+1]] - symbol2value[s[i]]
				i++
				continue
			}
		}
		res += symbol2value[s[i]]
	}

	return res
}

//14. Longest Common Prefix

func longestCommonPrefix(strs []string) string {
	var res string

	lth := len(strs)
	for i := 0; i < len(strs[0]); i++ {
		pivot := strs[0][i]
		for j := 1; j < lth; j++ {
			if len(strs[j])-1 < i {
				return res
			}
			if pivot != strs[j][i] {
				return res
			}
		}
		res += string(pivot)
	}

	return res
}

//20. Valid Parentheses

func isValid(s string) bool {
	if len(s) == 0 {
		return true
	}
	var chStack []uint8
	chStack = append(chStack, s[0])

	for i := 1; i < len(s); i++ {

		if len(chStack) == 0 {
			chStack = append(chStack, s[i])
		} else {
			top := chStack[len(chStack)-1]
			if (top == '(' && s[i] == ')') || (top == '[' && s[i] == ']') || (top == '{' && s[i] == '}') {
				chStack = chStack[:len(chStack)-1]
			} else {
				chStack = append(chStack, s[i])
			}
		}

	}
	if len(chStack) == 0 {
		return true
	}
	return false
}

//21. Merge Two Sorted Lists

// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil && l2 == nil {
		return nil
	}
	if l1 != nil && l2 == nil {
		return l1
	}

	if l1 == nil && l2 != nil {
		return l2
	}

	head := new(ListNode)
	cur := head
	for l1 != nil && l2 != nil {
		if l2.Val < l1.Val {
			cur.Next = l2
			cur = cur.Next
			l2 = l2.Next
			continue
		}
		cur.Next = l1
		cur = cur.Next
		l1 = l1.Next
	}
	for l1 != nil {
		cur.Next = l1
		cur = cur.Next
		l1 = l1.Next
	}
	for l2 != nil {
		cur.Next = l2
		cur = cur.Next
		l2 = l2.Next

	}
	return head.Next
}

func mergeTwoListsV2(l1 *ListNode, l2 *ListNode) *ListNode {
	var head *ListNode = nil
	if l1 == nil && l2 != nil {
		head = l2
	} else if l1 != nil && l2 == nil {
		head = l1
	} else if l1 == nil && l2 == nil {
		return head
	} else {
		if l1.Val < l2.Val {
			head = l1
			head.Next = mergeTwoListsV2(l1.Next, l2)
		} else {
			head = l2
			head.Next = mergeTwoListsV2(l1, l2.Next)
		}
	}
	return head

}

//26. Remove Duplicates from Sorted Array

func removeDuplicates(nums []int) int {
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

//28. Implement strStr()

func strStr(haystack string, needle string) int {
	if needle == "" {
		return 0
	}
	cur := 0
	for cur < len(haystack) {
		flag := false
		for i := cur; i < len(haystack); i++ {
			if haystack[i] == needle[0] {
				cur = i
				flag = true
				break
			}
		}
		if !flag {
			return -1
		}
		needleIndex := 1
		haystackIndex := cur + 1
		for ; needleIndex < len(needle); needleIndex++ {
			if haystackIndex < len(haystack) {
				if haystack[haystackIndex] == needle[needleIndex] {
					haystackIndex++
				} else {
					break
				}
			} else {
				return -1
			}
		}
		if needleIndex == len(needle) {
			return cur
		}
		cur++
	}
	return -1
}

func strStrV2(haystack string, needle string) int {
	lenH, lenN := len(haystack), len(needle)
	if lenN == 0 {
		return 0
	}
	for i := 0; i <= lenH-lenN; i++ {
		found := true
		for k, j := i, 0; k < lenH && j < lenN; k, j = k+1, j+1 {
			if haystack[k] != needle[j] {
				found = false
			}
		}
		if found {
			return i
		}

	}

	return -1
}

// 53. Maximum Subarray [dp,conquer]
func maxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i-1]+nums[i] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}
	return max

}

//66. Plus One

func plusOne(digits []int) []int {
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

//69. Sqrt(x)  [bin search]

func mySqrt(x int) int {
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

//70. Climbing Stairs  [dp]

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

func mergeArr(nums1 []int, m int, nums2 []int, n int) {
	var setArr []int

	i := 0
	j := 0
	for i < m && j < n {
		if nums1[i] < nums2[j] {
			setArr = append(setArr, nums1[i])
			i++
			continue
		}
		setArr = append(setArr, nums2[j])
		j++
	}

	for i < m {
		setArr = append(setArr, nums1[i])
		i++
	}
	for j < n {
		setArr = append(setArr, nums2[j])
		j++
	}

	copy(nums1, setArr)
}

func mergeV2(nums1 []int, m int, nums2 []int, n int) {
	var setArr []int
	i, j := 0, 0
	recMerge(nums1, m, nums2, n, &setArr, i, j)
	copy(nums1, setArr)
}

func recMerge(nums1 []int, m int, nums2 []int, n int, setArr *[]int, i int, j int) {
	if i < m && j == n {
		*setArr = append(*setArr, nums1[i])
		i++
	} else if i == m && j < n {
		*setArr = append(*setArr, nums2[j])
		j++
	} else if i == m && j == n {
		return
	} else {
		if nums1[i] < nums2[j] {
			*setArr = append(*setArr, nums1[i])
			i++
		} else {
			*setArr = append(*setArr, nums2[j])
			j++
		}
	}
	recMerge(nums1, m, nums2, n, setArr, i, j)
}

func mergeV3(nums1 []int, m int, nums2 []int, n int) {
	i, j := 0, 0
	for i < m && j < n {
		if nums2[j] < nums1[i] {
			nums1[i], nums2[j] = nums2[j], nums1[i]
			k := j
			for k < n-1 && nums2[k] > nums2[k+1] {
				nums2[k], nums2[k+1] = nums2[k+1], nums2[k]
				k++
			}
		}
		i++
	}

	for j < n {
		nums1[i] = nums2[j]
		i++
		j++
	}

}

//94. Binary Tree Inorder Traversal

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	var res []int
	recInorder(root, &res)
	return res
}

func recInorder(node *TreeNode, res *[]int) {
	if node != nil {
		recInorder(node.Left, res)
		*res = append(*res, node.Val)
		recInorder(node.Right, res)
	}

}

//101. Symmetric Tree

func isSymmetric(root *TreeNode) bool {

	return recIsSymmetric(root.Left, root.Right)

}

func recIsSymmetric(left *TreeNode, right *TreeNode) bool {
	if left == nil && right == nil {
		return true
	} else if left != nil && right != nil {
		if left.Val == right.Val {
			if !recIsSymmetric(left.Right, right.Left) {
				return false
			}
			if !recIsSymmetric(left.Left, right.Right) {
				return false
			}
		} else {
			return false
		}
	} else {
		return false
	}
	return true
}

func isSymmetricV2(root *TreeNode) bool {
	var nodeStack []*TreeNode
	nodeStack = append(nodeStack, root)
	for len(nodeStack) > 0 {
		var nodeVal []int
		lth := len(nodeStack)
		for i := 0; i < lth; i++ {
			if nodeStack[i] == nil {
				nodeVal = append(nodeVal, -101)
				continue
			}
			nodeVal = append(nodeVal, nodeStack[i].Val)
			nodeStack = append(nodeStack, nodeStack[i].Left)
			nodeStack = append(nodeStack, nodeStack[i].Right)
		}
		nodeStack = nodeStack[lth:]
		if !isSy(nodeVal) {
			return false
		}

	}
	return true

}

func isSy(vals []int) bool {
	lhs := 0
	rhs := len(vals) - 1
	for lhs <= rhs {
		if vals[lhs] != vals[rhs] {
			return false
		}
		lhs++
		rhs--
	}
	return true
}

// 104. Maximum Depth of Binary Tree
func maxDepth(root *TreeNode) int {
	return recDepth(root, 0)
}

func recDepth(node *TreeNode, cnt int) int {
	if node != nil {
		cnt++
		left := recDepth(node.Left, cnt)
		right := recDepth(node.Right, cnt)
		cnt = left
		if right > left {
			cnt = right
		}
	}
	return cnt
}

// 108. Convert Sorted Array to Binary Search Tree
func sortedArrayToBST(nums []int) *TreeNode {
	lth := len(nums)
	if lth == 0 {
		return nil
	}
	mid := lth / 2
	val := nums[mid]
	return &TreeNode{
		Val:   val,
		Left:  sortedArrayToBST(nums[:mid]),
		Right: sortedArrayToBST(nums[mid+1:]),
	}

}

//118. Pascal's Triangle

func generate(numRows int) [][]int {
	res := make([][]int, numRows)
	res[0] = append(res[0], 1)
	column := 1
	for column < numRows {
		row := column + 1
		res[column] = make([]int, row)
		for i := 0; i < row; i++ {
			if i == 0 || i == row-1 {
				res[column][i] = 1
				continue
			}
			res[column][i] = res[column-1][i] + res[column-1][i-1]
		}
		column++

	}

	return res

}

//121. Best Time to Buy and Sell Stock

func maxProfitV1(prices []int) int {
	lth := len(prices)

	minPrice := prices[0]
	maxRes := 0

	for i := 1; i < lth; i++ {
		profit := prices[i] - minPrice
		if profit > maxRes {
			maxRes = profit
		}
		if minPrice > prices[i] {
			minPrice = prices[i]
		}
	}
	return maxRes

}

//125. Valid Palindrome

func isPalindromeStr(s string) bool {
	sLow := strings.ToLower(s)
	var bytes []uint8

	for i := 0; i < len(sLow); i++ {
		if sLow[i] >= 'a' && sLow[i] <= 'z' || sLow[i] >= '0' && sLow[i] <= '9' {
			bytes = append(bytes, sLow[i])
		}
	}

	lhs := 0
	rhs := len(bytes) - 1

	for lhs <= rhs {
		if bytes[lhs] != bytes[rhs] {
			return false
		}
		lhs++
		rhs--
	}

	return true

}

//136. Single Number

func singleNumber(nums []int) int {
	res := 0
	for i := 0; i < len(nums); i++ {
		res ^= nums[i]
	}
	return res

}

// 141. Linked List Cycle
func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	slow := head
	fast := head
	for slow != nil && fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

func hasCycleV2(head *ListNode) bool {
	record := make(map[*ListNode]bool)
	cur := head
	for cur != nil {
		if _, ok := record[cur]; ok {
			return true
		}
		record[cur] = true
		cur = cur.Next
	}
	return false
}

//155. Min Stack

// 160. Intersection of Two Linked Lists
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	stepA := 0
	stepB := 0
	cur := headA
	for cur != nil {
		stepA++
		cur = cur.Next
	}
	cur = headB
	for cur != nil {
		stepB++
		cur = cur.Next
	}
	curA := headA
	curB := headB
	if stepA > stepB {
		for i := 1; i <= stepA-stepB; i++ {
			curA = curA.Next
		}
	} else {
		for i := 1; i <= stepB-stepA; i++ {
			curB = curB.Next
		}

	}
	for curA != nil || curB != nil {
		if curA == curB {
			return curA
		}
		curA = curA.Next
		curB = curB.Next
	}
	return nil

}

//169. Majority Element

func majorityElement(nums []int) int {
	pivot := len(nums) / 2
	record := make(map[int]int)

	for _, num := range nums {
		record[num]++
		if record[num] > pivot {
			return num
		}
	}

	return 0

}

func majorityElementV2(nums []int) int {
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
func titleToNumber(s string) int {
	lth := len(s) - 1
	acc := 1
	res := int(s[lth] - 'A' + 1)
	for i := lth - 1; i >= 0; i-- {
		acc *= 26
		val := int(s[i]-'A'+1) * acc
		res += val
	}
	return res
}

func titleToNumberV2(s string) int {
	col := 0
	for _, r := range s {
		cur := int(r - 'A' + 1)
		col = col*26 + cur
	}
	return col
}

// 190. Reverse Bits
func reverseBits(num uint32) uint32 {
	res := uint32(0)
	for i := 0; i < 32; i++ {
		res = (res << 1) | (num & 1)
		num = num >> 1
	}
	return res
}

func reverseBitsV2(num uint32) uint32 {
	res := uint32(0)
	for i := 0; i < 32; i++ {
		if num&1 == 1 {
			res = (res << 1) + 1
		} else {
			res = res << 1
		}
		num >>= 1
	}
	return res
}

// 191. Number of 1 Bits
func hammingWeight(num uint32) int {
	cnt := 0
	for i := 0; i < 32; i++ {
		if num&1 == 1 {
			cnt++
		}
		num >>= 1
	}
	return cnt
}

//202. Happy Number

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

	return false
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

//217. Contains Duplicate

func containsDuplicate(nums []int) bool {
	record := make(map[int]bool)
	for i := 0; i < len(nums); i++ {
		if _, ok := record[nums[i]]; ok {
			return true
		}
		record[nums[i]] = true
	}
	return false
}

//234. Palindrome Linked List

func isPalindrome(head *ListNode) bool {
	var nums []int
	cur := head
	for cur != nil {
		nums = append(nums, cur.Val)
		cur = cur.Next
	}
	lhs := 0
	rhs := len(nums) - 1
	for lhs <= rhs {
		if nums[lhs] != nums[rhs] {
			return false
		}
		lhs++
		rhs--
	}
	return true
}

func isPalindromeV2(head *ListNode) bool {
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

// 237. Delete Node in a Linked List
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// 242. Valid Anagram
func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	res := make([]int, 26)
	for _, ch := range s {
		res[int(ch-'a')]++
	}
	for _, ch := range t {
		res[int(ch-'a')]--
		if res[int(ch-'a')] < 0 {
			return false
		}
	}
	return true
}

//268. Missing Number

func missingNumber(nums []int) int {
	lth := len(nums)
	sum := 0
	for i := 0; i <= lth; i++ {
		sum += i
	}
	for _, num := range nums {
		sum -= num
	}
	return sum
}

//283. Move Zeroes

func moveZeroes(nums []int) {
	cur := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[cur] = nums[i]
			cur++
		}
	}
	for i := cur; i < len(nums); i++ {
		nums[i] = 0
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

// 344. Reverse String
func reverseString(s []byte) {
	lhs := 0
	rhs := len(s) - 1
	for lhs <= rhs {
		s[lhs], s[rhs] = s[rhs], s[lhs]
		lhs++
		rhs--
	}
}

// 350. Intersection of Two Arrays II
func intersect(nums1 []int, nums2 []int) []int {

	return nil
}

// 387. First Unique Character in a String
func firstUniqChar(s string) int {
	record := make([]int, 27)
	res := -1

	for i := 0; i < len(s); i++ {
		if record[s[i]-'a'+1] == 0 {
			record[s[i]-'a'+1] = int(s[i]-'a') + 1
		} else {
			record[s[i]-'a'+1] = 0
		}
	}
	for _, index := range record {
		if index != 0 {
			if res == -1 {
				res = index
				continue
			}
			if index < res {
				res = index
			}
		}
	}
	return res
}

// 412. Fizz Buzz
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

//2. Add Two Numbers

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
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

//3. Longest Substring Without Repeating Characters

func lengthOfLongestSubstring(s string) int {
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

// 5. Longest Palindromic Substring
func longestPalindrome(s string) string {
	maxString := ""
	for i := 0; i < len(s); i++ {
		//字符串长度可能为奇数也可能为偶数，两周情况都要考虑
		//奇数
		maxString = findSubLongestPalindrome(i, i, s, maxString)
		//偶数
		maxString = findSubLongestPalindrome(i, i+1, s, maxString)

	}
	return maxString
}

func findSubLongestPalindrome(lhs, rhs int, s, maxString string) string {
	for lhs >= 0 && rhs < len(s) && s[lhs] == s[rhs] {
		lhs--
		rhs++
	}
	subStr := s[lhs+1 : rhs]
	if len(subStr) > len(maxString) {
		maxString = subStr
	}
	return maxString
}

func longestPalindromeV2(s string) string {
	lth := len(s)
	dp := make([][]bool, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]bool, lth)
	}
	maxStr := ""
	for i := 0; i < lth; i++ {
		for j := 0; j < i; j++ {
			if i == j {
				dp[i][i] = true
			}
			if s[i] == s[j] {
				if i == j+1 {
					dp[j][i] = true
				}
				if i > j+1 && dp[j+1][i-1] == true {
					dp[j][i] = true
				}
			}
			if dp[j][i] == true && len(s[j:i+1]) > len(maxStr) {
				maxStr = s[j : i+1]
			}
		}
	}
	return maxStr

}

// 8. String to Integer (atoi)
func myAtoi(s string) int {
	if len(s) == 0 {
		return 0
	}
	max := math.MaxInt32
	min := math.MinInt32
	np := 1
	index := 0
	isLegal := false
	for index < len(s) {
		if s[index] == ' ' {
			index++
			continue
		} else if s[index] == '-' {
			np = -1
			index++
			if index < len(s) && (s[index] < '0' || s[index] > '9') {
				break
			}
		} else if s[index] == '+' {
			index++
			if index < len(s) && (s[index] < '0' || s[index] > '9') {
				break
			}
		} else if s[index] >= '0' && s[index] <= '9' {
			isLegal = true
			break
		} else {
			break
		}
	}
	if !isLegal {
		return 0
	}
	num := 0
	for index < len(s) {
		if s[index] >= '0' && s[index] <= '9' {
			if np*num > max/10 || np*num == max/10 && int(s[index]-'0') > 7 {
				return max
			}
			if np*num < min/10 || np*num == min/10 && int(s[index]-'0') > 8 {
				return min
			}
			num = num*10 + int(s[index]-'0')

		} else {
			break
		}
		index++
	}

	return np * num
}

//11. Container With Most Water

func maxArea(height []int) int {
	lth := len(height)
	lhs := 0
	rhs := lth - 1
	max := 0
	for lhs < rhs {
		spacing := rhs - lhs
		h := 0
		if height[lhs] < height[rhs] {
			h = height[lhs]
			lhs++
		} else {
			h = height[rhs]
			rhs--
		}
		if h*spacing > max {
			max = h * spacing
		}

	}
	return max

}

func getMinNum(a, b int) int {
	if a < b {
		return a
	}
	return b
}

//15. 3Sum
//[-1,0,1,2,-1,-4]

func threeSum(nums []int) [][]int {
	var res [][]int
	sort.Slice(nums, func(i, j int) bool { return nums[i] < nums[j] })
	lth := len(nums)
	if lth < 3 {
		return res
	}
	lhs := 0
	rhs := lth - 1
	for nums[lhs] <= 0 && lhs < rhs-1 {
		sum := nums[lhs] + nums[rhs]
		remain := 0 - sum
		if remain > nums[rhs] {
			rhs = lth - 1
			lhs++
			for lhs < rhs-1 && nums[lhs] == nums[lhs-1] {
				lhs++
			}

		} else if remain < nums[lhs] {
			rhs--
		} else {
			mid := lhs + 1
			for mid < rhs {
				if nums[mid] == remain {
					res = append(res, []int{nums[lhs], nums[mid], nums[rhs]})

					break
				}
				mid++
			}
			rhs--
			for rhs > lhs+1 && nums[rhs] == nums[rhs+1] {
				rhs--
			}
		}

	}
	return res
}

func threeSumV2(nums []int) [][]int {
	var res [][]int
	sort.SliceStable(nums, func(i, j int) bool { return nums[i] < nums[j] })
	for i := len(nums) - 1; i >= 0; i-- {
		if i < len(nums)-1 && nums[i] == nums[i+1] {
			continue
		}
		lhs := 0
		rhs := i - 1
		for lhs < rhs {
			sum := nums[lhs] + nums[rhs] + nums[i]
			if 0 == sum {
				res = append(res, []int{nums[lhs], nums[rhs], nums[i]})
				for lhs < rhs {
					lhs++
					if nums[lhs] != nums[lhs-1] {
						break
					}
				}
				for lhs < rhs {
					rhs--
					if nums[rhs] != nums[rhs+1] {
						break
					}
				}
			} else if sum > 0 {
				rhs--
			} else {
				lhs++
			}
		}
	}
	return res
}

// 17. Letter Combinations of a Phone Numbe
func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	digit2letters := map[uint8]string{
		'2': "abc",
		'3': "def",
		'4': "ghi",
		'5': "jkl",
		'6': "mno",
		'7': "pqrs",
		'8': "tuv",
		'9': "wxyz",
	}
	res := []string{""}
	for i := 0; i < len(digits); i++ {
		letters := digit2letters[digits[i]]
		res = recComb(res, letters)
	}
	return res

}

func recComb(res []string, letters string) []string {
	var spellRes []string
	for i := 0; i < len(res); i++ {
		for j := 0; j < len(letters); j++ {
			tmp := res[i] + string([]byte{letters[j]})
			spellRes = append(spellRes, tmp)
		}
	}
	return spellRes
}

// 19. Remove Nth Node From End of List
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	if head == nil {
		return head
	}
	slow := head
	fast := head
	for step := 1; step <= n; step++ {
		if fast != nil {
			fast = fast.Next
		}
	}
	if fast == nil {
		head = head.Next
	} else {
		for fast.Next != nil {
			slow = slow.Next
			fast = fast.Next
		}
		slow.Next = slow.Next.Next
	}
	return head
}

// 22. Generate Parentheses
func generateParenthesis(n int) []string {
	var res []string
	recGen("(", 1, n-1, &res)
	return res
}

func recGen(baseStr string, toMatch int, n int, res *[]string) {

	if n == 0 && toMatch == 0 {
		*res = append(*res, baseStr)
	}
	if n > 0 {
		if toMatch > 0 {
			recGen(baseStr+")", toMatch-1, n, res)
		}
		recGen(baseStr+"(", toMatch+1, n-1, res)
	} else if n == 0 {
		if toMatch > 0 {
			recGen(baseStr+")", toMatch-1, n, res)
		}
	}

}

// 29. Divide Two Integers
func divide(dividend int, divisor int) int {
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
	if (dendSigned ^ sorSigned) == 1 {
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
	return ret * signed
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

//33. Search in Rotated Sorted Array

func search(nums []int, target int) int {
	lhs := 0
	rhs := len(nums) - 1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] == target {
			return mid
		} else if nums[lhs] <= nums[mid] {
			if nums[lhs] <= target && target < nums[mid] {
				rhs = mid - 1
			} else {
				lhs = mid + 1
			}
		} else {
			//lhs>mid
			if nums[mid] < target && target <= nums[rhs] {
				lhs = mid + 1
			} else {
				rhs = mid - 1
			}
		}
	}
	return -1

}

// 如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的
func searchV2(nums []int, target int) int {
	lhs := 0
	rhs := len(nums) - 1
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < nums[rhs] {
			if target > nums[mid] && target <= nums[rhs] {
				lhs = mid + 1
			} else {
				rhs = mid - 1
			}
		} else {
			if target < nums[mid] && target >= nums[lhs] {
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
	start := binarySearch(nums, lhs, rhs, target)
	if start == -1 {
		return []int{-1, -1}
	}

	min, max := start, start
	i, j := start, start
	for i >= 0 && nums[i] == target {
		min = i
		i--
	}
	for j < len(nums) && nums[j] == target {
		max = j
		j++
	}
	return []int{min, max}

}

func binarySearch(nums []int, lhs int, rhs int, target int) int {
	for lhs <= rhs {
		mid := (lhs + rhs) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > target {
			rhs = mid - 1
		} else {
			lhs = mid + 1
		}
	}
	return -1
}

// 36. Valid Sudoku
func isValidSudoku(board [][]byte) bool {
	rowRecord := [9][9]bool{}
	columnRecord := [9][9]bool{}
	nineRecord := [3][3][9]bool{}

	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {

			if board[i][j] < '0' || board[i][j] > '9' {
				continue
			}
			val := board[i][j] - '1'
			//判断行
			if rowRecord[i][val] == true {
				return false
			}
			//判断列
			if columnRecord[j][val] == true {
				return false
			}
			//判断九宫格
			row := getIndex(i)
			column := getIndex(j)
			if nineRecord[row][column][val] == true {
				return false
			}
			rowRecord[i][val] = true
			nineRecord[row][column][val] = true
			columnRecord[j][val] = true
		}
	}
	return true
}

func getIndex(num int) int {
	index := num + 1

	if index >= 1 && index <= 3 {
		return 0
	} else if index >= 4 && index <= 6 {
		return 1
	} else if index >= 7 && index <= 9 {
		return 2
	}
	return -1
}

// 38. Count and Say
func countAndSay(n int) string {
	str := "1"
	for i := 1; i < n; i++ {
		var pivot uint8
		cnt := uint8(0)
		var sb strings.Builder
		for j := 0; j < len(str); j++ {
			if str[j] != pivot {
				if cnt != 0 {
					sb.WriteByte(cnt + '0')
					sb.WriteByte(pivot)
				}
				pivot = str[j]
				cnt = 1
				continue
			}
			cnt++
		}
		sb.WriteByte(cnt + '0')
		sb.WriteByte(pivot)
		str = sb.String()
	}
	return str
}

//46. Permutations

func permute(nums []int) [][]int {
	var ret [][]int
	recPermute(nums, 0, &ret)
	return ret
}

func recPermute(nums []int, start int, ret *[][]int) {
	if start == len(nums)-1 {
		tmp := make([]int, len(nums))
		copy(tmp, nums)
		*ret = append(*ret, tmp)
	}
	for i := start; i < len(nums); i++ {
		if i == start || nums[i] != nums[start] {
			nums[i], nums[start] = nums[start], nums[i]
			recPermute(nums, start+1, ret)
			nums[i], nums[start] = nums[start], nums[i]
		}

	}
}

//48.Rotate Image

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

// 49. Group Anagrams
func groupAnagrams(strs []string) [][]string {
	dict := make(map[[26]int][]string)
	for _, v := range strs {
		ana := [26]int{}
		for _, c := range v {
			ana[c-'a']++
		}

		if _, ok := dict[ana]; !ok {
			dict[ana] = make([]string, 0)
		}
		dict[ana] = append(dict[ana], v)
	}

	res := make([][]string, 0)
	for _, v := range dict {
		res = append(res, v)
	}
	return res
}

//50. Pow(x, n)

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

// 54. Spiral Matrix
func spiralOrder(matrix [][]int) []int {
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

// 55. Jump Game
func canJump(nums []int) bool {
	lth := len(nums) - 1
	if lth <= 0 {
		return true
	}
	res := make([]bool, lth+1)
	recCanJump(lth, 0, nums, res)
	return res[lth]
}

func recCanJump(lth int, startIndex int, nums []int, res []bool) {
	for i := 1; i <= nums[startIndex]; i++ {
		pos := i + startIndex
		if pos < lth {
			if res[pos] {
				continue
			}
			res[pos] = true
			recCanJump(lth, pos, nums, res)
		} else if pos == lth {
			res[pos] = true

		} else {
			break
		}
	}
}

// 56. Merge Intervals
func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool { return intervals[i][0] < intervals[j][0] })
	var ret [][]int
	start := intervals[0][0]
	end := intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] <= end {
			if intervals[i][1] > end {
				end = intervals[i][1]
			}
		} else {
			ret = append(ret, []int{start, end})
			start = intervals[i][0]
			end = intervals[i][1]
		}
	}
	ret = append(ret, []int{start, end})
	return ret
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
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}

// 73. Set Matrix Zeroes
func setZeroes(matrix [][]int) {
	lth := len(matrix)
	rowFlag := make(map[int]bool)
	columnFlag := make(map[int]bool)
	for i := 0; i < lth; i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if matrix[i][j] == 0 {
				rowFlag[i] = true
				columnFlag[j] = true
			}
		}
	}

	for i := 0; i < lth; i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if rowFlag[i] == true || columnFlag[j] == true {
				matrix[i][j] = 0
			}
		}
	}
}

// 75. Sort Colors
func sortColors(nums []int) {
	lhs := 0
	rhs := len(nums) - 1
	i := lhs

	for i <= rhs {
		if nums[i] == 0 {
			nums[i], nums[lhs] = nums[lhs], nums[i]
			lhs++
			i++

		} else if nums[i] == 2 {
			nums[i], nums[rhs] = nums[rhs], nums[i]
			rhs--
		} else {
			i++
		}

	}

}

// 78. Subsets
func subsets(nums []int) [][]int {
	var ret [][]int
	ret = append(ret, []int{})
	for i := 0; i < len(nums); i++ {
		subNums := []int{nums[i]}
		ret = append(ret, subNums)
		recSubSets(i+1, subNums, nums, &ret)
	}

	return ret
}

func recSubSets(index int, subNums []int, nums []int, ret *[][]int) {
	if index == len(nums) {
		return
	}
	for i := index; i < len(nums); i++ {
		tmp := make([]int, len(subNums))
		copy(tmp, subNums)
		tmp = append(tmp, nums[i])
		*ret = append(*ret, tmp)
		recSubSets(i+1, tmp, nums, ret)
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

// 79. Word Search
// TODO bugfix
func exist(board [][]byte, word string) bool {
	row := len(board)
	column := len(board[0])

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			flag := i*column + j
			strIndex := 0
			if word[strIndex] == board[i][j] {
				if strIndex == len(word)-1 {
					return true
				}
				stack := make([][]int, len(word))
				find := true
				visit := make([]bool, row*column)
				visit[flag] = true
				curRow := i
				curColumn := j
				strIndex++
				preIndex := 0
				for strIndex < len(word) && find {
					find = false
					//向上
					if curRow-1 >= 0 && !visit[(curRow-1)*column+curColumn] && word[strIndex] == board[curRow-1][curColumn] {
						stack[preIndex] = append(stack[preIndex], (curRow-1)*column+curColumn)
						find = true
					}
					//向下
					if curRow+1 < row && !visit[(curRow+1)*column+curColumn] && word[strIndex] == board[curRow+1][curColumn] {
						stack[preIndex] = append(stack[preIndex], (curRow+1)*column+curColumn)
						find = true
					}
					//向左
					if curColumn-1 >= 0 && !visit[curRow*column+curColumn-1] && word[strIndex] == board[curRow][curColumn-1] {
						stack[preIndex] = append(stack[preIndex], curRow*column+curColumn-1)
						find = true
					}
					//向右
					if curColumn+1 < column && !visit[curRow*column+curColumn+1] && word[strIndex] == board[curRow][curColumn+1] {
						stack[preIndex] = append(stack[preIndex], curRow*column+curColumn+1)
						find = true
					}
					if find == true {
						if strIndex == len(word)-1 {
							return true
						}
						//取当前stack中第一个
						val := stack[preIndex][0]
						stack[preIndex] = stack[preIndex][1:]
						curRow = val / column
						curColumn = val % column
						visit[val] = true
						preIndex++
						strIndex++
						continue
					}
					visit[curRow*column+curColumn] = false
					for preIndex >= 0 && len(stack[preIndex]) == 0 {
						preIndex--
						strIndex--
					}
					if preIndex >= 0 && len(stack[preIndex]) > 0 {
						val := stack[preIndex][0]
						stack[preIndex] = stack[preIndex][1:]
						curRow = val / column
						curColumn = val % column
						visit[val] = true
						strIndex++
						find = true
						continue
					}
					break
				}
			}
		}
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

// 91. Decode Ways
func numDecodings(s string) int {
	dp := make([]int, len(s)+1)
	if s[0] == '0' {
		return 0
	}
	dp[0] = 1
	dp[1] = 1
	for i := 1; i < len(s); i++ {
		num := (s[i-1]-'0')*10 + (s[i] - '0')
		if s[i] > '0' && s[i] <= '9' && num >= 10 && num <= 26 {
			dp[i+1] = dp[i] + dp[i-1]
		} else if s[i] > '0' && s[i] <= '9' {
			dp[i+1] = dp[i]
		} else if num >= 10 && num <= 26 {
			dp[i+1] = dp[i-1]
		} else {
			dp[i+1] = 0
		}

	}
	return dp[len(s)]
}

// 98. Validate Binary Search Tree
func isValidBST(root *TreeNode) bool {
	return RecValidate(root, nil, nil)
}

func RecValidate(n, min, max *TreeNode) bool {
	if n == nil {
		return true
	}
	if min != nil && n.Val <= min.Val {
		return false
	}
	if max != nil && n.Val >= max.Val {
		return false
	}
	return RecValidate(n.Left, min, n) && RecValidate(n.Right, n, max)
}

// 102. Binary Tree Level Order Traversal
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int

	node := root
	var stack []*TreeNode
	stack = append(stack, node)
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

func zigzagLevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int

	node := root
	var stack []*TreeNode
	stack = append(stack, node)
	cnt := 1
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
		if cnt&1 == 0 {
			lhs := 0
			rhs := len(tmp) - 1
			for lhs <= rhs {
				tmp[lhs], tmp[rhs] = tmp[rhs], tmp[lhs]
				lhs++
				rhs--
			}
		}
		cnt++
		res = append(res, tmp)
		stack = stack[lth:]
	}

	return res

}

// 105. Construct Binary Tree from Preorder and Inorder Traversal
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	index := find(preorder[0], inorder)
	left := buildTree(preorder[1:index+1], inorder[:index])
	right := buildTree(preorder[index+1:], inorder[index+1:])
	return &TreeNode{
		Val:   preorder[0],
		Left:  left,
		Right: right,
	}
}

func find(target int, nums []int) int {
	for i, x := range nums {
		if x == target {
			return i
		}
	}
	return -1
}

// 116. Populating Next Right Pointers in Each Node
type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

func connect(root *Node) *Node {
	if root == nil {
		return nil
	}
	node := root
	var stack []*Node
	stack = append(stack, node)
	for len(stack) > 0 {
		lth := len(stack)
		for i := 0; i < lth; i++ {
			if i == lth-1 {
				stack[i].Next = nil
			} else {
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

// 122. Best Time to Buy and Sell Stock II
func maxProfit(prices []int) int {
	dp := make([]int, len(prices))
	dp[0] = 0
	for i := 1; i < len(prices); i++ {
		dp[i] = dp[i-1]
		for j := i - 1; j >= 0; j-- {
			tempProfit := 0
			if j-1 < 0 {
				tempProfit = prices[i] - prices[j]
			} else {
				tempProfit = prices[i] - prices[j] + dp[j-1]
			}
			if tempProfit > dp[i] {
				dp[i] = tempProfit
			}
		}
	}
	return dp[len(prices)-1]
}

func maxProfitV2(prices []int) int {
	max := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			max += prices[i] - prices[i-1]
		}
	}
	return max
}

// 128. Longest Consecutive Sequence
func longestConsecutive(nums []int) int {
	record := make(map[int]bool)
	for _, num := range nums {
		record[num] = true
	}
	max := 0
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
			if tmpCon > max {
				max = tmpCon
			}
		}

	}
	return max
}

//1567. Maximum Length of Subarray With Positive Product

func getMaxLen(nums []int) int {
	positive, negative, negativeCnt, max := -1, -1, 0, 0
	for i, num := range nums {
		if num == 0 {
			positive, negative, negativeCnt = i, -1, 0
		}
		//首个奇数的位置
		if num < 0 {
			negativeCnt++
			if negative == -1 {
				negative = i
			}
		}
		//偶数个负数
		if negativeCnt&1 == 0 {
			tmp := i - positive
			if tmp > max {
				max = tmp
			}
		} else {
			//奇数个负数
			tmp := i - negative
			if tmp > max {
				max = tmp
			}
		}
	}
	return max
}

func getMaxLenV2(nums []int) int {
	positive := make([]int, len(nums))
	negative := make([]int, len(nums))
	if nums[0] > 0 {
		positive[0] = 1
	}
	if nums[0] < 0 {
		negative[0] = 1
	}
	maxL := positive[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] > 0 {
			positive[i] = positive[i-1] + 1
			if negative[i-1] == 0 {
				negative[i] = 0
			} else {
				negative[i] = negative[i-1] + 1
			}

		} else if nums[i] < 0 {
			if negative[i-1] == 0 {
				positive[i] = 0
			} else {
				positive[i] = negative[i-1] + 1
			}
			negative[i] = positive[i-1] + 1
		} else {
			positive[i], negative[i] = 0, 0
		}
		maxL = max(maxL, positive[i])

	}
	return maxL
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

func (d *Dsu) Find1(child int) int {
	if father, ok := d.parent[child]; ok && child != father {
		d.parent[child] = d.Find1(father)
	}
	return d.parent[child]
}

func (d *Dsu) connect(child int, father int) {
	d.parent[child] = d.Find1(father)
}

func longestConsecutiveV2(nums []int) int {
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
		dsu.Find1(num)
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
	var flippingList [][]int
	var noFlippingList [][]int
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			isFlipping := true
			var list []int
			if board[i][j] == 'O' {
				if i == 0 || i == row-1 || j == 0 || j == column-1 {
					isFlipping = false
				}
				board[i][j] = '1'
				list = append(list, i*column+j)
				recSolve(board, i, j, column, row, &list, &isFlipping)
				if isFlipping == true {
					flippingList = append(flippingList, list)
				} else {
					noFlippingList = append(noFlippingList, list)
				}
			}
		}
	}
	//filpping
	for _, list := range flippingList {
		for _, num := range list {
			r := num / column
			j := num % column
			board[r][j] = 'X'
		}
	}

	for _, list := range noFlippingList {
		for _, num := range list {
			r := num / column
			j := num % column
			board[r][j] = 'O'
		}
	}

}

func recSolve(board [][]byte, i, j, column, row int, list *[]int, isFlipping *bool) {
	//上
	if i-1 >= 0 && board[i-1][j] == 'O' {
		board[i-1][j] = '1'
		*list = append(*list, (i-1)*column+j)
		if i-1 == 0 || j == 0 || j == column-1 {
			*isFlipping = false
		}
		recSolve(board, i-1, j, column, row, list, isFlipping)
	}
	//下
	if i+1 < row && board[i+1][j] == 'O' {
		board[i+1][j] = '1'
		*list = append(*list, (i+1)*column+j)
		if i+1 == row-1 || j == 0 || j == column-1 {
			*isFlipping = false
		}
		recSolve(board, i+1, j, column, row, list, isFlipping)
	}
	//左
	if j-1 >= 0 && board[i][j-1] == 'O' {
		board[i][j-1] = '1'
		*list = append(*list, i*column+j-1)
		if j-1 == 0 || i == 0 || i == row-1 {
			*isFlipping = false

		}
		recSolve(board, i, j-1, column, row, list, isFlipping)
	}
	//右
	if j+1 < column && board[i][j+1] == 'O' {
		board[i][j+1] = '1'
		*list = append(*list, i*column+j+1)
		if j+1 == column-1 || i == 0 || i == row-1 {
			*isFlipping = false

		}
		recSolve(board, i, j+1, column, row, list, isFlipping)
	}

	return
}

func solveV2(board [][]byte) {
	if board == nil || len(board) == 0 {
		return
	}
	m := len(board)
	n := len(board[0])
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 || j == 0 || i == m-1 || j == n-1 {
				if board[i][j] == 'O' {
					dfsSolve(board, i, j)
				}
			}
		}
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if board[i][j] == '#' {
				board[i][j] = 'O'
			} else if board[i][j] == 'O' {
				board[i][j] = 'X'
			}
		}
	}
	return
}

func dfsSolve(board [][]byte, i int, j int) {
	if i < 0 || j < 0 || i >= len(board) || j >= len(board[0]) {
		return
	}
	if board[i][j] == 'O' {
		board[i][j] = '#'
		dfsSolve(board, i+1, j)
		dfsSolve(board, i-1, j)
		dfsSolve(board, i, j+1)
		dfsSolve(board, i, j-1)
	}
	return
}

// 131. Palindrome Partitioning
func partition(s string) [][]string {
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

func isPal(str string) bool {

	lhs := 0
	rhs := len(str) - 1
	for lhs <= rhs {
		if str[lhs] != str[rhs] {
			return false
		}
		lhs++
		rhs--
	}
	return true
}

func partitionV2(s string) [][]string {
	var res [][]string
	var currList []string
	dfspartitionV2(0, &res, &currList, s)
	return res
}

func dfspartitionV2(start int, res *[][]string, currList *[]string, s string) {
	if start >= len(s) {
		currListCopy := make([]string, len(*currList))
		copy(currListCopy, *currList)
		*res = append(*res, currListCopy)
	}
	for end := start; end < len(s); end++ {
		if isPald(s, start, end) {
			*currList = append(*currList, s[start:end+1])
			dfspartitionV2(end+1, res, currList, s)
			*currList = (*currList)[:len(*currList)-1]
		}
	}
}

func isPald(s string, low, high int) bool {
	for low < high {
		if s[low] != s[high] {
			return false
		}
		low++
		high--
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

type RandomNode struct {
	Val    int
	Next   *RandomNode
	Random *RandomNode
}

func copyRandomList(head *RandomNode) *RandomNode {
	// Duplicate each node.
	curr := head
	for curr != nil {
		newNode := &RandomNode{
			Val:    curr.Val,
			Next:   curr.Next,
			Random: curr.Random,
		}

		curr.Next = newNode
		curr = newNode.Next
	}

	// Make the duplicated nodes point to the correct random.
	curr = head
	for curr != nil {
		curr = curr.Next
		if curr.Random != nil {
			curr.Random = curr.Random.Next
		}
		curr = curr.Next
	}

	// Extract the duplicated nodes and recover the original list.
	dummy := &RandomNode{}
	curr2 := dummy
	curr = head
	for curr != nil {
		n := curr.Next
		curr.Next = curr.Next.Next
		curr = curr.Next
		curr2.Next = n
		curr2 = curr2.Next
	}

	return dummy.Next

}
func copyRandomListV2(head *RandomNode) *RandomNode {
	if head == nil {
		return nil
	}

	oldNode2NewNode := make(map[*RandomNode]*RandomNode)
	newHead := new(RandomNode)
	newHead.Val = head.Val
	oldNode2NewNode[head] = newHead
	cur := head.Next
	newCur := newHead
	for cur != nil {
		node := &RandomNode{
			Val:    cur.Val,
			Next:   nil,
			Random: nil,
		}
		oldNode2NewNode[cur] = node
		newCur.Next = node
		newCur = newCur.Next
		cur = cur.Next
	}
	cur = head
	for cur != nil {
		oldNode2NewNode[cur].Random = oldNode2NewNode[cur.Random]
		cur = cur.Next
	}

	return newHead

}

// 139. Word Break
func wordBreak(s string, wordDict []string) bool {
	dp := make([]bool, len(s))
	for i := 0; i < len(dp); i++ {
		for j := i; j >= 0; j-- {
			subStr := s[j : i+1]
			if findStrIsExist(wordDict, subStr) {
				if j == 0 {
					dp[i] = true
				} else {
					if dp[j-1] == true {
						dp[i] = true
						break
					}
				}
			}

		}
	}

	return dp[len(dp)-1]
}

func findStrIsExist(wordDict []string, target string) bool {
	for _, str := range wordDict {
		if str == target {
			return true
		}
	}
	return false
}

type Info struct {
	a int
	b string
}

func ab() *Info {
	return nil
}

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
		if this.len > 1 {
			if node == this.head {
				next := node.next
				next.pre = nil
				this.head = next
				//添加到尾部
				this.tail.next = node
				node.pre = this.tail
				node.next = nil
				this.tail = node
			} else {
				if node != this.tail {
					//此节点拆开，放到尾部
					preNode := node.pre
					nextNode := node.next
					preNode.next = nextNode
					nextNode.pre = preNode
					//放到尾部
					this.tail.next = node
					node.pre = this.tail
					node.next = nil
					this.tail = node
				}
			}
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
			if nextNode != nil {
				nextNode.pre = nil
			}
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

// 148. Sort List
func sortList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	//冒泡
	var tailNode *ListNode = nil
	curNode := head
	nextNode := head.Next
	for curNode != tailNode {
		for curNode != tailNode && nextNode != tailNode {
			if curNode.Val > nextNode.Val {
				curNode.Val, nextNode.Val = nextNode.Val, curNode.Val

			}
			curNode = curNode.Next
			nextNode = nextNode.Next
		}
		tailNode = curNode
		curNode = head
		nextNode = head.Next
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

	return mergeV100(sortList(head), sortList(rightHead))
}

func mergeV100(list1 *ListNode, list2 *ListNode) *ListNode {
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

// 150. Evaluate Reverse Polish Notation
func evalRPN(tokens []string) int {
	var numsStack []int
	res := 0
	for _, token := range tokens {
		if token != "+" && token != "-" && token != "*" && token != "/" {
			num, _ := strconv.Atoi(token)
			numsStack = append(numsStack, num)
		} else {
			lth := len(numsStack)
			rhs := numsStack[lth-1]
			lhs := numsStack[lth-2]
			numsStack = numsStack[:lth-2]
			if token == "+" {
				res = lhs + rhs
			} else if token == "-" {
				res = lhs - rhs
			} else if token == "*" {
				res = lhs * rhs
			} else if token == "/" {
				res = lhs / rhs
			}
			numsStack = append(numsStack, res)
		}
	}
	return numsStack[0]

}

// 152. Maximum Product Subarray
func maxProduct(nums []int) int {
	res, curMin, curMax := nums[0], nums[0], nums[0]
	for _, v := range nums[1:] {
		if v < 0 {
			curMin, curMax = curMax, curMin
		}

		curMin = min(curMin*v, v)
		curMax = max(curMax*v, v)
		res = max(res, curMax)
	}

	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}

	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}

func maxProductV2(nums []int) int {
	lth := len(nums)
	maxDp := make([]int, lth)
	minDp := make([]int, lth)
	maxDp[0], minDp[0] = nums[0], nums[0]
	res := nums[0]
	for i := 1; i < lth; i++ {
		maxDp[i] = max(max(nums[i]*maxDp[i-1], nums[i]*minDp[i-1]), nums[i])
		minDp[i] = min(min(nums[i]*maxDp[i-1], nums[i]*minDp[i-1]), nums[i])
		res = max(res, maxDp[i])
	}
	return res
}

func maxProductV3(nums []int) int {
	lth := len(nums)
	res, maxProd, minProd := nums[0], nums[0], nums[0]
	for i := 1; i < lth; i++ {
		mxp := maxProd
		mnp := minProd
		maxProd = max(max(mxp*nums[i], mnp*nums[i]), nums[i])
		minProd = min(min(mxp*nums[i], mnp*nums[i]), nums[i])
		res = max(res, maxProd)
	}
	return res
}

// 166. Fraction to Recurring Decimal
func fractionToDecimal(numerator int, denominator int) string {
	res := ""

	if (numerator > 0 && denominator < 0) || (numerator < 0 && denominator > 0) {
		res += "-"
		if numerator < 0 {
			numerator = -numerator
		}
		if denominator < 0 {
			denominator = -denominator
		}
	}

	record := make(map[int]int)
	quo := numerator / denominator
	res += strconv.Itoa(quo)
	remain := numerator % denominator
	if remain != 0 {
		res += "."
	}
	var afterPointList []string
	index := 0
	for remain != 0 {

		if pos, ok := record[remain]; ok {
			res += strings.Join(afterPointList[:pos], "")
			res += "(" + strings.Join(afterPointList[pos:], "") + ")"
			break
		}
		record[remain] = index
		remain *= 10
		quo = remain / denominator
		remain = remain % denominator
		afterPointList = append(afterPointList, strconv.Itoa(quo))
		index++
	}
	if remain == 0 {
		res += strings.Join(afterPointList, "")
	}

	return res
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
	var numStrs []string
	for _, num := range nums {
		numStr := strconv.Itoa(num)
		numStrs = append(numStrs, numStr)
	}
	sort.Slice(numStrs, func(i, j int) bool { return numStrs[i]+numStrs[j] > numStrs[j]+numStrs[i] })
	if numStrs[0] == "0" {
		return ""
	}
	res := strings.Join(numStrs, "")
	return res

}

// 189. Rotate Array
func rotateV2(nums []int, k int) {
	lth := len(nums)
	k %= lth
	help(nums, 0, lth-1)
	help(nums, 0, k-1)
	help(nums, k, lth-1)

}

func help(nums []int, start, end int) {
	for start < end {
		nums[start], nums[end] = nums[end], nums[start]
		end--
		start++
	}

}

// 198. House Robber
func rob(nums []int) int {
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

// 204. Count Primes
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

// 207. Course Schedule
// 拓扑排序
// 入度：顶点的入度是指「指向该顶点的边」的数量；
// 出度：顶点的出度是指该顶点指向其他点的边的数量。
// 最终图中所有顶点的出度都为0，那么就存在一个拓扑序，这个图也就是有向无环图。
func canFinish(numCourses int, prerequisites [][]int) bool {
	intoNode := make(map[int]int, numCourses)
	father2childes := make(map[int][]int, numCourses)
	row := len(prerequisites)
	for i := 0; i < row; i++ {
		father := prerequisites[i][1]
		child := prerequisites[i][0]
		father2childes[father] = append(father2childes[father], child)
		intoNode[child]++
		if _, ok := intoNode[father]; !ok {
			intoNode[father] = 0
		}
	}
	//获取入度为0的节点,并入栈
	cnt := 0
	var stack []int
	for k, v := range intoNode {
		if v == 0 {
			stack = append(stack, k)
			cnt++
		}
	}
	for len(stack) > 0 {
		father := stack[0]
		childes := father2childes[father]
		for _, child := range childes {
			if _, ok := intoNode[child]; ok {
				intoNode[child]--
				if intoNode[child] == 0 {
					stack = append(stack, child)
					cnt++
				}
			}
		}
		stack = stack[1:]
	}
	if cnt == len(intoNode) {
		return true
	}
	return false
}

type status uint8

// other variations use white,gray,black as indicators of state
const (
	unprocessed status = iota
	processing
	processed
)

func dfsCourses(adjList [][]int, statusArr []status, node int) error {
	statusArr[node] = processing
	for _, child := range adjList[node] {
		switch statusArr[child] {
		case processing:
			return fmt.Errorf("cycle detected")
		case unprocessed:
			err := dfsCourses(adjList, statusArr, child)
			if err != nil { //dag is false
				return fmt.Errorf("cycle detected")
			}
		case processed:
			continue
		}
	}
	statusArr[node] = processed
	return nil
}

func canFinishV2(numCourses int, prerequisites [][]int) bool {
	//res := true    //assume dag is true initially
	// bi -> ai; bi is prereq to ai
	graph := make([][]int, numCourses)
	for i := 0; i < len(prerequisites); i++ {
		prereq, nextCourse := prerequisites[i][1], prerequisites[i][0]
		graph[prereq] = append(graph[prereq], nextCourse)
	}
	// maintain a status slice as we process dfs calls
	statusArr := make([]status, numCourses)
	for i := range statusArr {
		if statusArr[i] != processed {
			if err := dfsCourses(graph, statusArr, i); err != nil {
				return false //dag is false
			}
		}
	}
	return true

}

//["Trie","insert","insert","insert","insert","insert","insert","search","search","search","search","search","search","search","search","search","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith"]
//[[],["app"],["apple"],["beer"],["add"],["jam"],["rental"],["apps"],["app"],["ad"],["applepie"],["rest"],["jan"],["rent"],["beer"],["jam"],["apps"],["app"],["ad"],["applepie"],["rest"],["jan"],["rent"],["beer"],["jam"]]

// 208. Implement Trie (Prefix Tree)
type Trie struct {
	next   []*Trie
	isWord bool
}

func ConstructorTrie() Trie {
	return Trie{isWord: false, next: make([]*Trie, 26)}
}

func (this *Trie) Insert(word string) {
	lth := len(word)
	for i := 0; i < lth; i++ {
		alph := word[i]
		if this.next[alph-'a'] == nil {
			this.next[alph-'a'] = &Trie{
				next:   nil,
				isWord: false,
			}
			this.next[alph-'a'].next = make([]*Trie, 26)
		}
		this = this.next[alph-'a']
		if i == lth-1 {
			this.isWord = true
		}
	}

}

func (this *Trie) Search(word string) bool {
	for _, alph := range word {
		if this.next[alph-'a'] != nil {
			this = this.next[alph-'a']
		} else {
			return false
		}
	}
	return this.isWord
}

func (this *Trie) StartsWith(prefix string) bool {
	for _, alph := range prefix {
		if this.next[alph-'a'] != nil {
			this = this.next[alph-'a']
		} else {
			return false
		}
	}
	return true
}

// 210. Course Schedule II
func findOrder(numCourses int, prerequisites [][]int) []int {
	var res []int
	intoNodeNums := make(map[int]int)
	father2child := make(map[int][]int)
	for _, order := range prerequisites {
		father := order[1]
		child := order[0]
		father2child[father] = append(father2child[father], child)
		intoNodeNums[child]++
		if _, ok := intoNodeNums[father]; !ok {
			intoNodeNums[father] = 0
		}
	}
	var stack []int
	for k, v := range intoNodeNums {
		if v == 0 {
			stack = append(stack, k)
		}
	}
	isdeal := make(map[int]bool)
	for len(stack) > 0 {
		node := stack[0]
		res = append(res, node)
		isdeal[node] = true
		for _, child := range father2child[node] {
			intoNodeNums[child]--
			if intoNodeNums[child] == 0 {
				stack = append(stack, child)
			}
		}
		stack = stack[1:]
	}
	if len(res) == len(intoNodeNums) {
		for i := 0; i < numCourses; i++ {
			if _, ok := isdeal[i]; !ok {
				res = append(res, i)
			}
		}
		return res
	}
	return []int{}
}

// 215. Kth Largest Element in an Array
func findKthLargest(nums []int, k int) int {
	quickSort(nums, 0, len(nums)-1, k)
	return nums[len(nums)-k]
}

func quickSort(nums []int, start, end, k int) {
	lth := len(nums)
	pos := lth - k
	if start < end {
		index := partition1(nums, start, end)
		if index == pos {
			return
		}
		quickSort(nums, start, index-1, k)
		quickSort(nums, index+1, end, k)
	}
}

func partition1(nums []int, start, end int) int {
	pivot := nums[end]
	smallIndex := start - 1
	for i := start; i < end; i++ {
		if nums[i] < pivot {
			smallIndex++
			nums[i], nums[smallIndex] = nums[smallIndex], nums[i]
		}
	}
	nums[smallIndex+1], nums[end] = nums[end], nums[smallIndex+1]
	return smallIndex + 1
}

// 230. Kth Smallest Element in a BST
func kthSmallest(root *TreeNode, k int) int {
	var res []int
	var cnt int
	midOrder(root, k, &res, &cnt)
	fmt.Println(res)
	return res[k-1]
}

func midOrder(node *TreeNode, k int, nums *[]int, cnt *int) {
	if node != nil {
		midOrder(node.Left, k, nums, cnt)
		*cnt++
		*nums = append(*nums, node.Val)
		if *cnt == k {
			return
		}
		midOrder(node.Right, k, nums, cnt)

	}

}

// 236. Lowest Common Ancestor of a Binary Tree
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {

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

func lowestCommonAncestorV2(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root == p || root == q {
		return root
	}
	left := lowestCommonAncestorV2(root.Left, p, q)
	right := lowestCommonAncestorV2(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left == nil {
		return right
	}
	return left
}

// 238. Product of Array Except Self
func productExceptSelf(nums []int) []int {
	zeroCnt := 0
	allP := 1
	for _, num := range nums {
		if num == 0 {
			zeroCnt++
			continue
		}
		allP *= num
	}
	res := make([]int, len(nums))
	if zeroCnt > 1 {
		return res
	} else if zeroCnt == 1 {
		for i := 0; i < len(nums); i++ {
			if nums[i] == 0 {
				res[i] = allP
			}
		}
	} else {
		for i := 0; i < len(nums); i++ {
			if nums[i] != 0 {
				res[i] = allP / nums[i]
			}
		}
	}
	return res

}

func productExceptSelfV2(nums []int) []int {
	result := make([]int, len(nums))
	product := 1
	for i := len(nums) - 1; i >= 0; i-- {
		result[i] = product
		product *= nums[i]
	}

	product = 1
	for i := 0; i < len(nums); i++ {
		result[i] *= product
		product *= nums[i]
	}
	return result
}

// 240. Search a 2D Matrix II
func searchMatrix(matrix [][]int, target int) bool {
	row := len(matrix)
	column := len(matrix[0])
	i := 0
	j := 0
	for i < row && j < column && j >= 0 {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] < target {
			if j+1 < column {
				if matrix[i][j+1] <= target {
					j++
				} else {
					i++
				}
			} else {
				i++
			}
		} else {

			j--

		}

	}
	return false
}

func searchMatrixV2(matrix [][]int, target int) bool {
	if len(matrix) == 0 {
		return false
	}
	row, col := len(matrix)-1, len(matrix[0])-1
	i, j := row, 0
	for i >= 0 && j <= col {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] > target {
			i--
		} else {
			j++
		}
	}
	return false
}

// 279. Perfect Squares
func numSquares(n int) int {
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = math.MaxInt32
		for j := 1; i-j*j >= 0; j++ {
			dp[i] = min(dp[i], dp[i-j*j]+1)
		}
	}
	return dp[n]
}

func numSquaresV2(n int) int {
	queue := []int{0}
	visited := make(map[int]bool)
	cnt := 0
	for len(queue) != 0 {
		curr_len := len(queue) // number of elements in current level
		for i := 0; i < curr_len; i++ {
			v := queue[i] // pop
			if v == n {
				return cnt
			}
			for j := 1; v+j*j <= n; j++ {
				if !visited[v+j*j] {
					queue = append(queue, v+j*j) // push
					visited[v+j*j] = true
				}
			}
		}
		queue = queue[curr_len:] // delete elements from previous level
		cnt++
	}
	return -1
}

// 287. Find the Duplicate Number
func findDuplicate(nums []int) int {
	duplicate := -1
	low := 0
	high := len(nums) - 1
	for low <= high {
		cur := (low + high) / 2
		cnt := 0
		for _, num := range nums {
			if num <= cur {
				cnt++
			}
		}
		if cnt > cur {
			duplicate = cur
			high = cur - 1
		} else {
			low = cur + 1
		}
	}
	return duplicate
}

// 324. Wiggle Sort II
func wiggleSort(nums []int) {
	sort.Slice(nums, func(i, j int) bool { return nums[i] < nums[j] })
	lth := len(nums)
	tmp := make([]int, lth)
	copy(tmp, nums)
	index := 1

	for i := lth - 1; i >= 0; i-- {
		if index >= lth {
			index = 0
		}
		nums[index] = tmp[i]
		index += 2
	}

}

// 328. Odd Even Linked List
func oddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil || head.Next.Next == nil {
		return head
	}
	pre := head
	cur := head.Next
	evenHead := &ListNode{
		Val:  head.Next.Val,
		Next: nil,
	}
	evenTail := evenHead
	cnt := 2
	for cur.Next != nil {
		if cnt%2 == 0 {
			//断开
			pre.Next = cur.Next
			if cnt > 2 {
				evenTail.Next = &ListNode{
					Val:  cur.Val,
					Next: nil,
				}
				evenTail = evenTail.Next
			}
		} else {
			pre = cur
		}
		cur = cur.Next
		cnt++
	}
	if cnt%2 == 0 {
		evenTail.Next = &ListNode{
			Val:  cur.Val,
			Next: nil,
		}
		pre.Next = cur.Next
		cur = pre
	}
	cur.Next = evenHead

	return head

}

func oddEvenListV2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	odd := head
	even := head.Next
	evenHead := head.Next
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = evenHead
	return head
}

// 334. Increasing Triplet Subsequence
func increasingTriplet(nums []int) bool {
	low := nums[0]
	mid := math.MaxInt32
	for _, v := range nums {
		if v > mid {
			return true
		}
		if v < low {
			low = v
		}
		if v > low && v < mid {
			mid = v
		}
	}
	return false
}

// 347. Top K Frequent Elements
func topKFrequent(nums []int, k int) []int {
	num2freq := make(map[int]int)
	for _, num := range nums {
		num2freq[num]++
	}
	var freqList []int
	for _, v := range num2freq {
		freqList = append(freqList, v)
	}
	//select top k
	lth := len(freqList)
	pos := lth - k
	quickSort2(freqList, 0, lth-1, pos)
	res := make([]int, 0, k)
	for num, v := range num2freq {
		if v >= freqList[pos] {
			res = append(res, num)
		}
	}
	return res
}

func quickSort2(nums []int, start, end, pos int) {
	if start < end {
		q := partition2(nums, start, end)
		if q == pos {
			return
		}
		quickSort2(nums, start, q-1, pos)
		quickSort2(nums, q+1, end, pos)
	}
}

func partition2(nums []int, start int, end int) int {
	pivot := nums[end]
	smallIndex := start - 1
	for i := start; i < end; i++ {
		if nums[i] < pivot {
			smallIndex++
			nums[i], nums[smallIndex] = nums[smallIndex], nums[i]
		}
	}
	nums[end], nums[smallIndex+1] = nums[smallIndex+1], nums[end]
	return smallIndex + 1
}

func topKFrequentV2(nums []int, k int) []int {
	res := []int{}
	Map := make(map[int]int, k)
	for _, e := range nums {
		Map[e]++
	}
	for ; k > 0; k-- {
		ress := 0
		index := -1
		for i, e := range Map {
			if index < e {
				index = e
				ress = i
			}
		}
		delete(Map, ress)
		res = append(res, ress)
	}
	return res
}

func topKFrequentV3(nums []int, k int) []int {
	dict := make(map[int]int)
	for _, v := range nums {
		dict[v]++
	}

	var keys []int

	for num, _ := range dict {
		keys = append(keys, num)
	}

	sort.Slice(keys, func(i int, j int) bool {
		return dict[keys[i]] > dict[keys[j]]
	})

	return keys[:k]
}

// 378. Kth Smallest Element in a Sorted Matrix
func kthSmallestV3(matrix [][]int, k int) int {
	n := len(matrix)
	lo := matrix[0][0]
	hi := matrix[n-1][n-1]
	var mid, count int

	for lo < hi {
		mid = (hi + lo) / 2
		count = countLEQ(matrix, mid)

		if count < k {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	return hi
}

func countLEQ(matrix [][]int, x int) int {
	n := len(matrix)
	count := 0
	var j int

	for _, row := range matrix {
		for j = 0; j < n && row[j] <= x; j++ {
		}
		count += j
	}

	return count
}

func main() {
	start := rand.Intn(7)
	fmt.Println(start)
}

// 384. Shuffle an Array
type Solution struct {
	nums []int
}

func ConstructorSolution(nums []int) Solution {
	return Solution{nums: nums}
}

func (this *Solution) Reset() []int {
	return this.nums
}

func (this *Solution) Shuffle() []int {
	lth := len(this.nums)
	ret := make([]int, lth)
	copy(ret, this.nums)
	for i := 0; i < lth-1; i++ {
		rand.Seed(time.Now().Unix())
		index := rand.Intn(lth - i)
		ret[lth-1-i], ret[index] = ret[index], ret[lth-1-i]
	}

	return ret
}

func permutes(nums []int, start int) []int {
	if start == len(nums)-1 {
		return nums
	}
	for i := start; i < len(nums); i++ {
		nums[i], nums[start] = nums[start], nums[i]
		permutes(nums, start+1)
		nums[i], nums[start] = nums[start], nums[i]
	}
	return nums
}

/**
 * Your Solution object will be instantiated and called as such:
 * obj := Constructor(nums);
 * param_1 := obj.Reset();
 * param_2 := obj.Shuffle();
 */
//395. Longest Substring with At Least K Repeating Characters
func longestSubstring(s string, k int) int {
	lth := len(s)
	return longestSubstringUtil(s, 0, lth, k)
}

func longestSubstringUtil(s string, start, end, k int) int {
	if end < k {
		return 0
	}
	countMap := make([]int, 26)
	for i := start; i < end; i++ {
		countMap[s[i]-'a']++
	}
	for mid := start; mid < end; mid++ {
		if countMap[s[mid]-'a'] >= k {
			continue
		}
		midNext := mid + 1
		for midNext < end && countMap[s[midNext]-'a'] < k {
			midNext++
		}
		return max(longestSubstringUtil(s, start, mid, k), longestSubstringUtil(s, midNext, end, k))
	}
	return end - start
}

//Find the number of unique characters in the string s and store the count in variable maxUnique. For s = aabcbacad, the unique characters are a,b,c,d and maxUnique = 4.
//
//Iterate over the string s with the value of currUnique ranging from 1 to maxUnique. In each iteration, currUnique is the maximum number of unique characters that must be present in the sliding window.
//
//The sliding window starts at index windowStart and ends at index windowEnd and slides over string s until windowEnd reaches the end of string s. At any given point, we shrink or expand the window to ensure that the number of unique characters is not greater than currUnique.
//
//If the number of unique character in the sliding window is less than or equal to currUnique, expand the window from the right by adding a character to the end of the window given by windowEnd
//
//Otherwise, shrink the window from the left by removing a character from the start of the window given by windowStart.
//
//Keep track of the number of unique characters in the current sliding window having at least k frequency given by countAtLeastK. Update the result if all the characters in the window have at least k frequency.

func longestSubstringV2(s string, k int) int {
	return 0
}

// 454. 4Sum II
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	dict := make(map[int]int)
	lth := len(nums1)
	res := 0

	for i := 0; i < lth; i++ {
		for j := 0; j < lth; j++ {
			dict[nums1[i]+nums2[j]]++
		}
	}

	for i := 0; i < lth; i++ {
		for j := 0; j < lth; j++ {
			res += dict[-nums3[i]-nums4[j]]
		}
	}

	return res
}
