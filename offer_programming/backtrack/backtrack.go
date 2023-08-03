package backtrack

import (
	"sort"
	"strconv"
	"strings"
)

/*
回溯法可以看作蛮力法的升级版，它在解决问题时的每一步都尝试所有可能的选项，最终找出所有可行的解决方案。
回溯法非常适合解决由多个步骤组成的问题，并且每个步骤都有多个选项。在某一步选择了其中一个选项之后，就进入下一步，然后会面临新的选项。
就这样重复选择，直至到达最终的状态

除了可以解决与集合排列、组合相关的问题，回溯法还能解决很多算法面试题。如果解决一个问题需要若干步骤，并且每一步都面临若干选项，当在某一步做了某个选择之后前往下一步仍然面临若干选项，那么可以考虑尝试用回溯法解决。通常，回溯法可以用递归的代码实现。
适用回溯法的问题的一个特征是问题可能有很多个解，并且题目要求列出所有的解。如果题目只是要求计算解的数目，或者只需要求一个最优解（通常是最大值或最小值），那么可能需要运用动态规划

*/
//所有子集
/*
输入一个不含重复数字的数据集合，请找出它的所有子集。例如，数据集合[1，2]有4个子集，分别是[]、[1]、[2]和[1，2]。
*/

func Subsets(nums []int) [][]int {
	var ret [][]int
	base := []int{}
	recSubsets(nums, base, 0, &ret)
	return ret
}

func recSubsets(nums, base []int, index int, ret *[][]int) {
	tmp := make([]int, len(base))
	copy(tmp, base)
	*ret = append(*ret, tmp)
	if index == len(nums) {
		return
	}
	for i := index; i < len(nums); i++ {
		base = append(base, nums[i])
		recSubsets(nums, base, i+1, ret)
		base = base[:len(base)-1]
	}
}

func SubsetsV2(nums []int) [][]int {
	ans := [][]int{}
	var dfs func([]int, int, []int)
	dfs = func(nums []int, idx int, path []int) {
		if idx <= len(nums) {
			ans = append(ans, append([]int{}, path...))
		}
		for i := idx; i < len(nums); i++ {
			path = append(path, nums[i])
			dfs(nums, i+1, path)
			//回溯节点
			path = path[:len(path)-1]
		}
	}
	dfs(nums, 0, []int{})
	return ans
}

//包含k个元素的组合
/*
输入n和k，请输出从1到n中选取k个数字组成的所有组合。例如，如果n等于3，k等于2，将组成3个组合，分别是[1，2]、[1，3]和[2，3]
*/

func Combine(n int, k int) [][]int {
	var ret [][]int
	var dfs func([]int, int, int)
	dfs = func(base []int, cnt, num int) {
		if cnt == k {
			ret = append(ret, append([]int{}, base...))
			return
		}
		for i := num; i <= n; i++ {
			base = append(base, i)
			dfs(base, cnt+1, i+1)
			base = base[:len(base)-1]
		}
	}
	dfs([]int{}, 0, 1)
	return ret
}

//允许重复选择元素的组合
/*
给定一个没有重复数字的正整数集合，请找出所有元素之和等于某个给定值的所有组合。同一个数字可以在组合中出现任意次。
例如，输入整数集合[2，3，5]，元素之和等于8的组合有3个，分别是[2，2，2，2]、[2，3，3]和[3，5]。
*/
func CombinationSum(candidates []int, target int) [][]int {
	var ret [][]int
	var dfs func(int, int, []int)
	dfs = func(sum, index int, base []int) {
		if sum == target {
			ret = append(ret, append([]int{}, base...))
			return
		}

		for i := index; i < len(candidates); i++ {
			if sum > target {
				break
			}
			sum += candidates[i]
			base = append(base, candidates[i])
			dfs(sum, i, base)
			sum -= candidates[i]
			base = base[:len(base)-1]
		}
	}
	dfs(0, 0, []int{})
	return ret
}

//包含重复元素集合的组合
/*
给定一个可能包含重复数字的整数集合，请找出所有元素之和等于某个给定值的所有组合。输出中不得包含重复的组合。
例如，输入整数集合[2，2，2，4，3，3]，元素之和等于8的组合有2个，分别是[2，2，4]和[2，3，3]。
*/

func combinationSum2(candidates []int, target int) [][]int {
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i] < candidates[j]
	})
	var ret [][]int
	var dfs func(int, int, []int)
	dfs = func(index, sum int, base []int) {
		if sum == target {
			ret = append(ret, append([]int{}, base...))
			return
		}
		for i := index; i < len(candidates); i++ {
			if sum > target {
				break
			}
			if i > index && candidates[i] == candidates[i-1] {
				continue
			}
			sum += candidates[i]
			base = append(base, candidates[i])
			dfs(i+1, sum, base)
			sum -= candidates[i]
			base = base[:len(base)-1]
		}
	}
	dfs(0, 0, []int{})
	return ret
}

//没有重复元素集合的全排列
/*
给定一个没有重复数字的集合，请找出它的所有全排列。例如，集合[1，2，3]有6个全排列，分别是[1，2，3]、[1，3，2]、[2，1，3]、[2，3，1]、[3，1，2]和[3，2，1]。
*/

func permute(nums []int) [][]int {
	var ret [][]int
	var dfs func(int)
	dfs = func(index int) {
		if index == len(nums) {
			ret = append(ret, append([]int{}, nums...))
			return
		}
		for i := index; i < len(nums); i++ {
			nums[i], nums[index] = nums[index], nums[i]
			dfs(index + 1)
			nums[i], nums[index] = nums[index], nums[i]
		}
	}
	dfs(0)
	return ret
}

//包含重复元素集合的全排列
/*
给定一个包含重复数字的集合，请找出它的所有全排列。例如，集合[1，1，2]有3个全排列，分别是[1，1，2]、[1，2，1]和[2，1，1]。
*/

func permuteUnique(nums []int) [][]int {
	var ans [][]int
	sort.Ints(nums)
	var dfs func(count int)
	var temp []int
	visited := make(map[int]bool)
	dfs = func(count int) {
		if count == len(nums) {
			ans = append(ans, append([]int{}, temp...))
		}
		for i := 0; i < len(nums); i++ {
			if visited[i] == true || (i > 0 && visited[i-1] == false && nums[i] == nums[i-1]) {
				continue
			}
			visited[i] = true
			temp = append(temp, nums[i])
			dfs(count + 1)
			temp = temp[:len(temp)-1]
			visited[i] = false
		}
	}
	dfs(0)
	return ans
}

func permuteUniqueV2(nums []int) [][]int {
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

//生成匹配的括号
/*
输入一个正整数n，请输出所有包含n个左括号和n个右括号的组合，要求每个组合的左括号和右括号匹配。例如，当n等于2时，有两个符合条件的括号组合，分别是"（()）"和"()()"。
*/

func generateParenthesis(n int) []string {
	var ret []string
	var dfs func(int, int, int, string)
	dfs = func(leftCnt, rightCnt, matchCnt int, str string) {
		if matchCnt == n {
			ret = append(ret, str)
		}
		if leftCnt < n {
			dfs(leftCnt+1, rightCnt, matchCnt, str+"(")
		}
		if rightCnt < leftCnt && matchCnt < n {
			dfs(leftCnt, rightCnt+1, matchCnt+1, str+")")
		}

	}
	dfs(0, 0, 0, "")
	return ret
}

//分割回文子字符串
/*
输入一个字符串，要求将它分割成若干子字符串，使每个子字符串都是回文。请列出所有可能的分割方法。
例如，输入"google"，将输出3种符合条件的分割方法，分别是["g"，"o"，"o"，"g"，"l"，"e"]、["g"，"oo"，"g"，"l"，"e"]和["goog"，"l"，"e"]。
*/
func partition(s string) [][]string {
	var ret [][]string
	var dfs func(int, []string)
	dfs = func(start int, base []string) {
		if start == len(s) {
			ret = append(ret, append([]string{}, base...))
			return
		}
		for index := start; index < len(s); index++ {
			if isPalindrome(s, start, index) {
				base = append(base, s[start:index+1])
				dfs(index+1, base)
				base = base[:len(base)-1]
			}
		}

	}
	dfs(0, []string{})
	return ret
}

func isPalindrome(s string, start, end int) bool {
	lhs := start
	rhs := end
	for lhs <= rhs {
		if s[lhs] != s[rhs] {
			return false
		}
		lhs++
		rhs--
	}
	return true
}

//恢复IP地址
/*
输入一个只包含数字的字符串，请列出所有可能恢复出来的IP地址。例如，输入字符串"10203040"，可能恢复出3个IP地址，分别为"10.20.30.40"、"102.0.30.40"和"10.203.0.40"。
*/

func restoreIpAddresses(s string) []string {
	var ret []string
	var dfs func(int, []string)
	dfs = func(start int, base []string) {
		if len(base) > 4 {
			return
		}
		if start == len(s) && len(base) == 4 {
			ip := strings.Join(base, ".")
			ret = append(ret, ip)
			return
		}
		for i := start; i <= start+3 && i < len(s); i++ {
			if s[start] == '0' {
				base = append(base, s[start:start+1])
				dfs(i+1, base)
				break
			}
			num, _ := strconv.Atoi(s[start : i+1])
			if num > 0 && num <= 255 {
				base = append(base, s[start:i+1])
				dfs(i+1, base)
				base = base[:len(base)-1]
			}

		}
	}
	dfs(0, []string{})
	return ret
}
