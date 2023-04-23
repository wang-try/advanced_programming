package str

import "math"

//字符串旋转
/*
给定一个字符串，要求将字符串前面的若干个字符串移到字符串的尾部。例如，将字符串"abcdef"的迁3个字符'a','b','c'移到字符串的尾部，
那么原字符串将变成"defabc"，请写一个函数实现此功能。
*/

func LeftShiftN(str string, n int) string {
	strByte := []byte(str)
	lth := len(strByte)
	n %= lth
	reverse(strByte, 0, lth-1)
	reverse(strByte, 0, n-1)
	reverse(strByte, n, lth-1)
	return string(strByte)
}

func reverse(strByte []byte, lhs, rhs int) {
	for lhs < rhs {
		strByte[lhs], strByte[rhs] = strByte[rhs], strByte[lhs]
		lhs++
		rhs--
	}
}

//字符串包含
/*给定一长字符串a和一段字符串b。请问，如何最快地判断出短字符串b中的所有字符是否都在长字符a中？
请编写函数，实现此功能
	为简单起见，假设输入的字符串只包含大写英文字母。下面举几个例子。
	. 如果字符安串a是"ABCD",字符串b是"BAD"，答案是true，因为字符串b中的字母都在字符串a中，或者说b是a的真子集。
	. 如果字符串a是"ABCD"，字符串b是"BCE"，答案是false，因为字符串b中的字母E不在字符串a中。
	. 如果字符串a是："ABCD"，字符串是"AA"，答案是true，因为字符串b中的字母A也包含在字符串a中。
*/

func StringContain(a, b string) bool {
	var hash [26]int
	for i := 0; i < len(a); i++ {
		hash[a[i]-'A'] = 1
	}
	for i := 0; i < len(b); i++ {
		if hash[b[i]-'A'] == 0 {
			return false
		}
	}
	return true
}

func StringContainV2(a, b string) bool {
	hash := 0
	for i := 0; i < len(a); i++ {
		hash |= 1 << (a[i] - 'A')
	}
	for i := 0; i < len(b); i++ {
		if hash&(1<<(b[i]-'A')) == 0 {
			return false
		}
	}
	return true
}

//字符串的全排列
/*
输入一个字符串，打印出该字符串中字符的所有排列。例如：输入字符"abc"，则输出由字符'a','b','c'所能排列出的所有字符串"abc","acb","bac","bca","cab","cba"
*/

func AllPermutation(str string) []string {
	var res []string
	recPermutation([]byte(str), 0, &res)
	return res
}

func recPermutation(strByte []byte, start int, res *[]string) {
	if start == len(strByte)-1 {
		tmp := make([]byte, len(strByte))
		copy(tmp, strByte)
		*res = append(*res, string(strByte))
	}
	for i := start; i < len(strByte); i++ {
		strByte[i], strByte[start] = strByte[start], strByte[i]
		recPermutation(strByte, start+1, res)
		strByte[i], strByte[start] = strByte[start], strByte[i]
	}
}

// 字符串，下一个字典序排列
func CalNextLexPermutation(str string) {
	//找到最右升序位置
	strByte := []byte(str)
	i := len(strByte) - 2
	for ; i >= 0 && strByte[i+1] <= strByte[i]; i++ {
	}
	if i >= 0 {
		j := len(strByte) - 1
		for ; strByte[j] <= strByte[i]; j-- {
		}
		strByte[j], strByte[i] = strByte[i], strByte[j]
	}
	lhs := i + 1
	rhs := len(str) - 1

	for lhs < rhs {
		strByte[lhs], strByte[rhs] = strByte[rhs], strByte[lhs]
		lhs++
		rhs--
	}
}

//字符串转化成整数
/*输入一个有数字组成的字符串，请把它转化成整数并输出。
 */

func Str2Int(s string) int {
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

//回文判断
/*给定一个字符串，如何判断这个字符串是否是回文串
 */

func IsPalindrome(str string) bool {
	lhs := 0
	rhs := len(str) - 1
	for lhs < rhs {
		if str[lhs] != str[rhs] {
			return false
		}
	}
	return true
}

//最长回文串，中心扩散法
/*给定一个字符串，求它的最长回文子串的长度*/
func longestPalindrome(s string) string {
	lth := len(s)
	maxSub := s[:1]
	for i := 0; i < lth; i++ {
		lhs := i - 1
		rhs := i + 1
		for s[i] == s[lhs] && lhs >= 0 {
			lhs--
		}
		for s[rhs] == s[i] && rhs < lth {
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

// manacher 算法
func longestPalindromeManacher(s string) string {
	return ""
}

// leetcode 647. Palindromic Substrings,中心扩散法
func countSubstrings(s string) int {
	total := 0
	for mid := 0; mid < len(s); mid++ {
		//偶数个
		left := mid
		right := mid + 1
		for left >= 0 && right < len(s) && s[left] == s[right] {
			left--
			right++
			total++
		}

		//奇数个
		left = mid
		right = mid
		for left >= 0 && right < len(s) && s[left] == s[right] {
			left--
			right++
			total++
		}
	}
	return total

}

func countSubstringsV2(s string) int {
	cnt := 0
	lth := len(s)
	dp := make([][]bool, lth)
	for i := 0; i < lth; i++ {
		dp[i] = make([]bool, lth)
		dp[i][i] = true
		cnt++
	}
	for j := 1; j < lth; j++ {
		for i := j - 1; i >= 0; i-- {
			if s[i] == s[j] {
				if dp[i+1][j-1] || j-1 == i {
					dp[i][j] = true
					cnt++
				}
			}
		}
	}
	return cnt
}

//数组
//2.1 有n个数，请找出其中最小的k个数，要求时间复杂度尽可能低

// 快排
func QuickSort(nums []int, start, end int) {
	if start < end {
		pos := partition(nums, start, end)
		QuickSort(nums, start, pos-1)
		QuickSort(nums, pos+1, end)
	}
}

func partition(nums []int, start, end int) int {
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

func GetLeastNumbers(arr []int, k int) []int {
	quickSortSelect(arr, k, 0, len(arr)-1)
	return arr[:k]
}

func quickSortSelect(arr []int, k, start, end int) {
	if start < end {
		pos := partition(arr, start, end)
		if pos == k-1 {
			return
		}
		quickSortSelect(arr, k, start, pos-1)
		quickSortSelect(arr, k, pos+1, end)
	}
}
