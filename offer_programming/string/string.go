package string

//字符串中的变位词
/*
输入字符串s1和s2，如何判断字符串s2中是否包含字符串s1的某个变位词？如果字符串s2中包含字符串s1的某个变位词，则字符串s1至少有一个变位词是字符串s2的子字符串。假设两个字符串中只包含英文小写字母。
例如，字符串s1为"ac"，字符串s2为"dgcaf"，由于字符串s2中包含字符串s1的变位词"ca"，因此输出为true。如果字符串s1为"ab"，字符串s2为"dgcaf"，则输出为false
*/
func CheckInclusion(s1, s2 string) bool {
	if len(s2) < len(s1) {
		return false
	}

	var hash [26]int
	for i := 0; i < len(s1); i++ {
		hash[s1[i]-'a']++
		hash[s2[i]-'a']--
	}
	if areAllZero(hash) {
		return true
	}

	for i := len(s1); i < len(s2); i++ {
		hash[s2[i]-'a']--
		hash[s2[i-len(s1)]-'a']++
		if areAllZero(hash) {
			return true
		}
	}
	return false
}

func areAllZero(hash [26]int) bool {
	for i := 0; i < len(hash); i++ {
		if hash[i] != 0 {
			return false
		}
	}
	return true
}

/*
双指针
输入字符串s1和s2，如何找出字符串s2的所有变位词在字符串s1中的起始下标？假设两个字符串中只包含英文小写字母。
例如，字符串s1为"cbadabacg"，字符串s2为"abc"，字符串s2的两个变位词"cba"和"bac"是字符串s1中的子字符串，输出它们在字符串s1中的起始下标0和5。
*/

func FindAnagrams(s, p string) []int {
	var ret []int
	if len(s) < len(p) {
		return ret
	}

	var hash [26]int
	for i := 0; i < len(p); i++ {
		hash[p[i]-'a']++
		hash[s[i]-'a']--
	}
	if areAllZero(hash) {
		ret = append(ret, 0)
	}

	for i := len(p); i < len(s); i++ {
		hash[s[i]-'a']--
		hash[s[i-len(p)]-'a']++
		if areAllZero(hash) {
			ret = append(ret, i-len(p)+1)
		}
	}
	return ret
}

//不含重复字符的最长子字符串
/*
输入一个字符串，求该字符串中不含重复字符的最长子字符串的长度。
例如，输入字符串"babcca"，其最长的不含重复字符的子字符串是"abc"，长度为3。
*/

func LengthOfLongestSubstring(s string) int {
	maxLth := 0
	var hash [256]int
	lhs := 0
	rhs := 0
	for rhs < len(s) {
		if hash[s[rhs]] != 0 {
			for i := lhs; i < hash[s[rhs]]-1; i++ {
				hash[s[i]] = 0
			}
			lhs = hash[s[rhs]]
			hash[s[rhs]] = rhs + 1
			rhs++
			continue
		}
		hash[s[rhs]] = rhs + 1
		if (rhs - lhs + 1) > maxLth {
			maxLth = rhs - lhs + 1
		}
		rhs++
	}
	return maxLth
}

func LengthOfLongestSubstringV2(s string) int {
	maxLth := 0
	lhs := 0
	rhs := 0
	countDup := 0
	var hash [256]int
	for rhs < len(s) {
		hash[s[rhs]]++
		if hash[s[rhs]] == 2 {
			countDup++
		}
		for countDup > 0 {
			hash[s[lhs]]--
			lhs++
			if hash[s[rhs]] == 1 {
				countDup--
			}
		}
		if (rhs - lhs + 1) > maxLth {
			maxLth = rhs - lhs + 1
		}
		rhs++
	}
	return maxLth
}

//包含所有字符的最短字符串
/*
输入两个字符串s和t，请找出字符串s中包含字符串t的所有字符的最短子字符串。
例如，输入的字符串s为"ADDBANCAD"，字符串t为"ABC"，则字符串s中包含字符'A'、'B'和'C'的最短子字符串是"BANC"。
如果不存在符合条件的子字符串，则返回空字符串""。如果存在多个符合条件的子字符串，则返回任意一个。
*/
func MinWindow(s, t string) string {
	if len(s) < len(t) {
		return ""
	}
	hash := make(map[uint8]int)
	for i := 0; i < len(t); i++ {
		hash[t[i]]++
	}
	start := 0
	for ; start < len(s); start++ {
		if hash[s[start]] > 0 {
			break
		}
	}
	if start == len(s) {
		return ""
	}
	lhs := start
	rhs := start
	ret := ""
	for rhs < len(s) {
		if _, ok := hash[s[rhs]]; ok {
			hash[s[rhs]]--
			if IsContain(hash) {
				for {
					cnt, ok := hash[s[lhs]]
					if !ok {
						lhs++
					} else if ok && cnt < 0 {
						hash[s[lhs]]++
						lhs++
					} else {
						break
					}

				}
				if (rhs-lhs+1) < len(ret) || ret == "" {
					ret = s[lhs : rhs+1]
				}
			}
		}
		rhs++
	}
	return ret
}

func IsContain(hash map[uint8]int) bool {
	for _, v := range hash {
		if v > 0 {
			return false
		}
	}
	return true
}

//最多删除一个字符得到回文
/*
给定一个字符串，请判断如果最多从字符串中删除一个字符能不能得到一个回文字符串。例如，如果输入字符串"abca"，由于删除字符'b'或'c'就能得到一个回文字符串，因此输出为true。
*/

func ValidPalindrome(s string) bool {
	lhs := 0
	rhs := len(s) - 1

	for lhs < rhs {
		if s[lhs] != s[rhs] {
			lRet := validPalindromeHelp(lhs+1, rhs, s)
			rRet := validPalindromeHelp(lhs, rhs-1, s)
			return lRet || rRet
		}
		lhs++
		rhs--
	}
	return true
}

func validPalindromeHelp(lhs, rhs int, s string) bool {
	for lhs < rhs {
		if s[lhs] != s[rhs] {
			return false
		}
		lhs++
		rhs--
	}
	return true
}

//回文子字符串的个数
/*
给定一个字符串，请问该字符串中有多少个回文连续子字符串？例如，字符串"abc"有3个回文子字符串，分别为"a"、"b"和"c"；而字符串"aaa"有6个回文子字符串，分别为"a"、"a"、"a"、"aa"、"aa"和"aaa"。
*/
func CountSubstrings(s string) int {
	cnt := 0
	for i := 0; i < len(s); i++ {
		//单节点为中心
		lhs := i
		rhs := i
		for lhs >= 0 && rhs < len(s) {
			if s[lhs] == s[rhs] {
				cnt++
				lhs--
				rhs++
				continue
			}
			break

		}
		//双节点为中心
		lhs = i
		rhs = i + 1
		for lhs >= 0 && rhs < len(s) {
			if s[lhs] == s[rhs] {
				cnt++
				lhs--
				rhs++
				continue
			}
			break
		}

	}
	return cnt
}
