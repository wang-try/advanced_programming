package backtrack

import "sort"

/*
回溯法可以看作蛮力法的升级版，它在解决问题时的每一步都尝试所有可能的选项，最终找出所有可行的解决方案。
回溯法非常适合解决由多个步骤组成的问题，并且每个步骤都有多个选项。在某一步选择了其中一个选项之后，就进入下一步，然后会面临新的选项。
就这样重复选择，直至到达最终的状态
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
