package high_frequency_summary

import (
	"math"
	"strconv"
	"strings"
)

// 字符串
// leetcode394 字符串解码  2[abc]3[cd]ef abc3[cd]xyz 3[a2[c]]
func decodeString(s string) string {
	var stack []string
	i := 0
	for i < len(s) {
		if s[i] >= '0' && s[i] <= '9' {
			start := i
			for ; i < len(s) && s[i] >= '0' && s[i] <= '9'; i++ {
			}
			stack = append(stack, s[start:i])
		} else if s[i] == '[' {
			stack = append(stack, string(s[i]))
			i++
		} else if s[i] >= 'a' && s[i] <= 'z' {
			start := i
			for ; i < len(s) && s[i] >= 'a' && s[i] <= 'z'; i++ {
			}
			stack = append(stack, s[start:i])
		} else {
			str := ""
			for len(stack) > 0 {
				top := stack[len(stack)-1]
				if top != "[" {
					stack = stack[:len(stack)-1]
					str = top + str
				} else {
					break
				}
			}
			stack = stack[:len(stack)-1]
			repeat, _ := strconv.Atoi(stack[len(stack)-1])
			stack = stack[:len(stack)-1]
			str = strings.Repeat(str, repeat)
			stack = append(stack, str)
			i++
		}
	}
	var ret string
	for i = 0; i < len(stack); i++ {
		ret += stack[i]
	}
	return ret
}

// leetcode 8
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

// leetcode344 反转字符串
func reverseString(s []byte) {
	lhs := 0
	rhs := len(s) - 1
	for lhs <= rhs {
		s[lhs], s[rhs] = s[rhs], s[lhs]
		lhs++
		rhs--
	}
}

// leetcode796 旋转字符串
func rotateString(s string, goal string) bool {
	if s == goal {
		return true
	}
	if len(s) != len(goal) {
		return false
	}
	splitIndex := 0
	isFind := false
	for i := 0; i < len(s); i++ {
		if s[i] == goal[0] {
			subLth := len(s) - i
			if s[i:] == goal[:subLth] {
				splitIndex = i
				isFind = true
				break
			}
		}
	}

	return isFind && s[:splitIndex] == goal[len(goal)-splitIndex:]
}

// leetcode415 字符串相加
func addStrings(num1 string, num2 string) string {
	index1, index2 := len(num1)-1, len(num2)-1
	carry := 0
	ans := ""
	for index1 >= 0 || index2 >= 0 {
		sum := 0
		n1 := 0
		if index1 >= 0 {
			n1 = int(num1[index1] - '0')
			index1--
		}
		n2 := 0
		if index2 >= 0 {
			n2 = int(num2[index2] - '0')
			index2--
		}
		sum = n1 + n2 + carry
		carry = sum / 10
		ans = strconv.Itoa(sum%10) + ans
	}
	if carry > 0 {
		ans = strconv.Itoa(carry) + ans
	}
	return ans
}

// leetcode151 翻转字符串里面的单词
func reverseWords(s string) string {
	index := 0
	var strs []string
	for index < len(s) {
		if s[index] == ' ' {
			for ; index < len(s) && s[index] == ' '; index++ {
			}

		} else {
			start := index
			for ; index < len(s) && s[index] != ' '; index++ {
			}
			strs = append(strs, s[start:index])
		}

	}
	lhs, rhs := 0, len(strs)-1
	for lhs <= rhs {
		strs[lhs], strs[rhs] = strs[rhs], strs[lhs]
		lhs++
		rhs--
	}

	return strings.Join(strs, " ")
}

// leetcode557 反转字符串里的单词 III
func reverseWordsIII(s string) string {
	index := 0
	var ans string
	for index < len(s) {
		if s[index] == ' ' {
			if len(ans) > 0 {
				ans += " "
			}
			index++
		} else {
			tmp := ""
			for ; index < len(s) && s[index] != ' '; index++ {
				tmp = string(s[index]) + tmp
			}
			ans += tmp
		}
	}
	return ans
}

// leetcode14 (字符串最长公共前缀)
func longestCommonPrefix(strs []string) string {
	lp := ""
	for i := 0; i < len(strs[0]); i++ {
		pivot := strs[0][i]
		cnt := 1
		for j := 1; j < len(strs); j++ {
			if i < len(strs[j]) && strs[j][i] == pivot {
				cnt++
			} else {
				return lp
			}
		}
		if cnt == len(strs) {
			lp = strs[0][:i+1]
		}
	}
	return lp
}

// LCR 014
func checkInclusion(s1 string, s2 string) bool {
	l1 := len(s1)
	l2 := len(s2)
	if l2 < l1 {
		return false
	}
	var hash [26]int
	for i := 0; i < len(s1); i++ {
		hash[s1[i]-'a']++
		hash[s2[i]-'a']--
	}
	if isAllCntZero(hash) {
		return true
	}
	for i := len(s1); i < len(s2); i++ {
		hash[s2[i]-'a']--
		hash[s2[i-len(s1)]-'a']++
		if isAllCntZero(hash) {
			return true
		}
	}
	return false

}

func isAllCntZero(hash [26]int) bool {
	for i := 0; i < len(hash); i++ {
		if hash[i] != 0 {
			return false
		}
	}
	return true
}

// LCR083 全排列
func permute(nums []int) [][]int {
	var ret [][]int
	var dfs func(index int)
	dfs = func(index int) {
		if index == len(nums) {
			ret = append(ret, append([]int{}, nums...))
			return
		}
		for i := index; i < len(nums); i++ {
			nums[index], nums[i] = nums[i], nums[index]
			dfs(index + 1)
			nums[index], nums[i] = nums[i], nums[index]
		}
	}
	dfs(0)
	return ret
}

// LCR084 全排列
func permuteUnique(nums []int) [][]int {
	var ret [][]int
	var dfs func(int)
	dfs = func(index int) {
		if index == len(nums) {
			ret = append(ret, append([]int{}, nums...))
			return
		}
		set := make(map[int]struct{})
		for i := index; i < len(nums); i++ {
			if _, ok := set[nums[i]]; !ok {
				set[nums[i]] = struct{}{}
				nums[i], nums[index] = nums[index], nums[i]
				dfs(index + 1)
				nums[i], nums[index] = nums[index], nums[i]
			}

		}
	}
	dfs(0)
	return ret
}

// leetcode1147段式回文
// 超出时间限制
func longestDecomposition(text string) int {
	dp := make([][]int, len(text))
	for i := 0; i < len(text); i++ {
		dp[i] = make([]int, len(text))
	}
	for i := len(text) - 1; i >= 0; i-- {
		for j := len(text) - 1; j >= i; j-- {
			dp[i][j] = 1
			lhs, rhs := i, j
			for lhs < rhs {
				if text[i:lhs+1] == text[rhs:j+1] {
					if lhs+1 == lhs {
						dp[i][j] = max(dp[i][j], 2)
					} else {
						dp[i][j] = max(dp[i][j], dp[lhs+1][rhs-1]+2)
						break
					}
				}
				lhs++
				rhs--
			}
		}
	}
	return dp[0][len(text)-1]
}

func longestDecompositionV2(text string) int {
	cnt := 0
	for len(text) > 0 {
		lhs, rhs := 0, len(text)-1
		for lhs < rhs {
			if text[:lhs+1] == text[rhs:] {
				cnt += 2
				text = text[lhs+1 : rhs]
				break
			}
			lhs++
			rhs--
		}

		if lhs >= rhs {
			cnt++
			break
		}
	}
	return cnt
}

// leetcode139 单词拆分
func wordBreak(s string, wordDict []string) bool {
	word2exist := make(map[string]bool, len(wordDict))
	for _, word := range wordDict {
		word2exist[word] = true
	}
	dp := make([]bool, len(s))
	for i := 0; i < len(s); i++ {
		for j := i; j >= 0; j-- {
			if word2exist[s[j:i+1]] && (j == 0 || dp[j-1]) {
				dp[i] = true
				break
			}
		}
	}
	return dp[len(s)-1]
}

// leetcode140 单词拆分II
func wordBreakII(s string, wordDict []string) []string {
	word2exist := make(map[string]bool, len(wordDict))
	for _, word := range wordDict {
		word2exist[word] = true
	}
	var ret []string
	var dfs func(index int, split []string)

	dfs = func(index int, split []string) {
		if index == len(s) {
			ret = append(ret, strings.Join(split, " "))
			return
		}
		for i := index; i < len(s); i++ {
			word := s[index : i+1]
			if word2exist[word] {
				split = append(split, word)
				dfs(i+1, split)
				split = split[:len(split)-1]
			}
		}
	}
	dfs(0, []string{})
	return ret
}

// leetcode680 验证回文字符串 II
func validPalindrome(s string) bool {
	lhs := 0
	rhs := len(s) - 1
	for lhs <= rhs {
		if s[lhs] != s[rhs] {
			flag1, flag2 := true, true
			l1, r1 := lhs+1, rhs
			for l1 <= r1 {
				if s[l1] != s[r1] {
					flag1 = false
					break
				}
				l1++
				r1--
			}
			l2, r2 := lhs, rhs-1
			for l2 <= r2 {
				if s[l2] != s[r2] {
					flag2 = false
					break
				}
				l2++
				r2--
			}
			return flag1 || flag2

		}
		lhs++
		rhs--
	}
	return true
}
