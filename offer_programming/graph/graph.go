package graph

import (
	"math"
	"strings"
)

//图的搜素
//最大的岛屿
/*
海洋岛屿地图可以用由0、1组成的二维数组表示，水平或竖直方向相连的一组1表示一个岛屿，请计算最大的岛屿的面积（即岛屿中1的数目）。例如，在图15.5中有4个岛屿，其中最大的岛屿的面积为5。
*/

func maxAreaOfIsland(grid [][]int) int {
	maxArea := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				area := 0
				dfsMaxAreaOfIsland(grid, i, j, &area)
				if area > maxArea {
					maxArea = area
				}
			}

		}
	}
	return maxArea
}

func dfsMaxAreaOfIsland(grid [][]int, i, j int, area *int) {
	row := len(grid)
	column := len(grid[0])
	if i < 0 || i >= row || j < 0 || j >= column {
		return
	}
	if grid[i][j] == 0 {
		return
	}
	grid[i][j] = 0
	*area += 1
	dfsMaxAreaOfIsland(grid, i+1, j, area)
	dfsMaxAreaOfIsland(grid, i-1, j, area)
	dfsMaxAreaOfIsland(grid, i, j+1, area)
	dfsMaxAreaOfIsland(grid, i, j-1, area)
}

//二分图
/*
如果能将一个图中的节点分成A、B两个部分，使任意一条边的一个节点属于A而另一个节点属于B，那么该图就是一个二分图。输入一个由数组graph表示的图，graph[i]中包含所有和节点i相邻的节点，请判断该图是否为二分图
*/

func isBipartite(graph [][]int) bool {
	n := len(graph)
	mark := make([]int, n)
	for i := range mark {
		mark[i] = -1
	}
	for i := 0; i < n; i++ {
		if mark[i] != -1 {
			continue
		}
		queue := []int{i}
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			for _, v := range graph[u] {
				if mark[v] != -1 {
					if mark[u]^mark[v] != 1 {
						return false
					}
				} else {
					mark[v] = mark[u] ^ 1
					queue = append(queue, v)
				}
			}
		}
	}
	return true

}

func isBipartiteV2(graph [][]int) bool {
	n := len(graph)
	colors := make([]int, n)
	for i := range colors {
		colors[i] = -1
	}
	for i := 0; i < n; i++ {
		if colors[i] == -1 {
			if !setColor(graph, colors, i, 0) {
				return false
			}
		}
	}
	return true
}

func setColor(graph [][]int, colors []int, i, color int) bool {
	if colors[i] >= 0 {
		return colors[i] == color
	}
	colors[i] = color
	for _, neighbor := range graph[i] {
		if !setColor(graph, colors, neighbor, 1-color) {
			return false
		}
	}
	return true
}

//矩阵中的距离
/*
输入一个由0、1组成的矩阵M，请输出一个大小相同的矩阵D，矩阵D中的每个格子是矩阵M中对应格子离最近的0的距离。
水平或竖直方向相邻的两个格子的距离为1。假设矩阵M中至少有一个0。
*/

func updateMatrix(mat [][]int) [][]int {
	row := len(mat)
	column := len(mat[0])
	dists := make([][]int, row)
	for i := 0; i < row; i++ {
		dists[i] = make([]int, column)
	}
	var queue [][]int
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if mat[i][j] == 1 {
				dists[i][j] = math.MaxInt
			} else {
				queue = append(queue, []int{i, j})
			}
		}
	}
	dirs := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for len(queue) > 0 {
		pos := queue[0]
		dist := dists[pos[0]][pos[1]]
		queue = queue[1:]
		for _, dir := range dirs {
			r := pos[0] + dir[0]
			c := pos[1] + dir[1]
			if r >= 0 && c >= 0 && r < row && c < column {
				if dists[r][c] > dist+1 {
					dists[r][c] = dist + 1
					queue = append(queue, []int{r, c})
				}
			}
		}
	}
	return dists
}

func updateMatrixV2(mat [][]int) [][]int {
	m, n := len(mat), len(mat[0])
	ans := make([][]int, m)
	var q [][2]int
	for i := 0; i < m; i++ {
		ans[i] = make([]int, n)
		for j := 0; j < n; j++ {
			if mat[i][j] == 0 {
				ans[i][j] = 0
				q = append(q, [2]int{i, j})
			} else {
				ans[i][j] = -1
			}
		}
	}
	dirs := [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for len(q) > 0 {
		x, y := q[0][0], q[0][1]
		q = q[1:]
		for _, d := range dirs {
			r, c := x+d[0], y+d[1]
			if r >= 0 && r < m && c >= 0 && c < n && ans[r][c] == -1 {
				ans[r][c] = ans[x][y] + 1
				q = append(q, [2]int{r, c})
			}
		}
	}
	return ans
}

//单词演变
/*
输入两个长度相同但内容不同的单词（beginWord和endWord）和一个单词列表，求从beginWord到endWord的演变序列的最短长度，要求每步只能改变单词中的一个字母，并且演变过程中每步得到的单词都必须在给定的单词列表中。
如果不能从beginWord演变到endWord，则返回0。假设所有单词只包含英文小写字母。
*/

func ladderLength(beginWord string, endWord string, wordList []string) int {
	var queue1, queue2 []string
	queue1 = append(queue1, beginWord)
	lth := 1
	isVisited := make(map[string]bool)
	for len(queue1) > 0 {
		word := queue1[0]
		if word == endWord {
			return lth
		}
		queue1 = queue1[1:]
		neighborList := neighbors(word, wordList)
		for _, neighbor := range neighborList {
			if !isVisited[neighbor] {
				queue2 = append(queue2, neighbor)
				isVisited[neighbor] = true
			}
		}
		if len(queue1) == 0 {
			lth++
			queue1 = queue2
			queue2 = []string{}
		}

	}
	return 0
}

func neighbors(str string, wordList []string) []string {
	var ret []string
	for _, word := range wordList {
		cnt := 0
		if len(word) != len(str) {
			continue
		}
		for i := 0; i < len(word); i++ {
			if word[i] != str[i] {
				cnt++
			}
		}
		if cnt == 1 {
			ret = append(ret, word)
		}
	}
	return ret
}

func ladderLengthV2(beginWord string, endWord string, wordList []string) int {
	isFind := false
	for _, word := range wordList {
		if word == endWord {
			isFind = true
			break
		}
	}
	if !isFind {
		return 0
	}
	isVisited := make(map[string]bool)
	set1 := make(map[string]bool)
	set2 := make(map[string]bool)
	length := 2
	set1[beginWord] = true
	set2[endWord] = true
	isVisited[endWord] = true
	for len(set1) > 0 && len(set2) > 0 {
		if len(set2) < len(set1) {
			set1, set2 = set2, set1
		}
		set3 := make(map[string]bool)
		for word, _ := range set1 {
			neighborList := neighbors(word, wordList)
			for _, neighbor := range neighborList {
				if set2[neighbor] {
					return length
				}
				if !isVisited[neighbor] {
					set3[neighbor] = true
					isVisited[neighbor] = true
				}

			}
		}
		length++
		set1 = set3
	}
	return 0
}

//开密码锁

/*
一个密码锁由4个环形转轮组成，每个转轮由0～9这10个数字组成。每次可以上下拨动一个转轮，如可以将一个转轮从0拨到1，也可以从0拨到9。密码锁有若干死锁状态，一旦4个转轮被拨到某个死锁状态，这个锁就不可能打开。
密码锁的状态可以用一个长度为4的字符串表示，字符串中的每个字符对应某个转轮上的数字。输入密码锁的密码和它的所有死锁状态，请问至少需要拨动转轮多少次才能从起始状态"0000"开始打开这个密码锁？如果锁不可能打开，则返回-1
*/

func openLock(deadends []string, target string) int {
	var queue1 []string
	var queue2 []string
	deadFlag := make(map[string]struct{})
	for _, dead := range deadends {
		if dead == "0000" || dead == target {
			return -1
		}
		deadFlag[dead] = struct{}{}
	}
	step := 0
	queue1 = append(queue1, "0000")
	isVisited := make(map[string]bool)
	isVisited["0000"] = true
	for len(queue1) > 0 {
		cur := queue1[0]
		if cur == target {
			return step
		}
		queue1 = queue1[1:]
		nexts := getNextStatus(cur)
		for _, next := range nexts {
			_, ok := deadFlag[next]
			if !ok && !isVisited[next] {
				queue2 = append(queue2, next)
				isVisited[next] = true
			}
		}
		if len(queue1) == 0 {
			step++
			queue1 = queue2
			queue2 = []string{}
		}

	}
	return -1
}

func getNextStatus(curStatus string) []string {
	var ret []string
	for index, _ := range curStatus {
		var sb1, sb2 strings.Builder
		sb1.WriteString(curStatus[:index])
		sb2.WriteString(curStatus[:index])
		if curStatus[index] == '0' {
			a := "9" + curStatus[index+1:]
			b := "1" + curStatus[index+1:]
			sb1.WriteString(a)
			sb2.WriteString(b)
		} else if curStatus[index] == '9' {
			a := "0" + curStatus[index+1:]
			b := "8" + curStatus[index+1:]
			sb1.WriteString(a)
			sb2.WriteString(b)
		} else {
			sb1.WriteByte(curStatus[index] - 1)
			sb1.WriteString(curStatus[index+1:])

			sb2.WriteByte(curStatus[index] + 1)
			sb2.WriteString(curStatus[index+1:])
		}

		ret = append(ret, sb1.String(), sb2.String())
	}
	return ret
}

func openLockV2(deadends []string, target string) int {
	start := "0000"
	if target == start {
		return 0
	}
	dist := make(map[string]int)
	for _, d := range deadends {
		dist[d] = -1
	}
	if dist[start] == -1 {
		return -1
	}
	queue := []string{start}
	dist[start] = 0
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for i := 0; i < 8; i++ {
			next := getNext(curr, i)
			if dist[next] == 0 || dist[next] > dist[curr]+1 {
				dist[next] = dist[curr] + 1
				queue = append(queue, next)
			}
			if next == target {
				return dist[next]
			}
		}
	}
	return -1
}

func getNext(curr string, idx int) string {
	b := []byte(curr)
	b[idx/2] = getUpOrDown(b[idx/2], idx&1)
	return string(b)
}

func getUpOrDown(num byte, needUp int) byte {
	if needUp == 0 {
		num--
		if num < '0' {
			num = '9'
		}
	} else {
		num++
		if num > '9' {
			num = '0'
		}
	}
	return num
}

//所有路径
/*
一个有向无环图由n个节点（标号从0到n-1，n≥2）组成，请找出从节点0到节点n-1的所有路径。图用一个数组graph表示，数组的graph[i]包含所有从节点i能直接到达的节点。
例如，输入数组graph为[[1，2]，[3]，[3]，[]]，则输出两条从节点0到节点3的路径，分别为0→1→3和0→2→3，如图15.12所示。
*/

// 深度优先搜索
func allPathsSourceTarget(graph [][]int) [][]int {
	ans := [][]int{}
	path := []int{0}
	n := len(graph)
	var dfs func(idx int)
	dfs = func(idx int) {
		if idx == n-1 {
			ans = append(ans, append([]int(nil), path...))
			return
		}

		for _, v := range graph[idx] {
			path = append(path, v)
			dfs(v)
			path = path[:len(path)-1]
		}
	}
	dfs(0)
	return ans
}

//计算除法
/*
题目：输入两个数组equations和values，其中，数组equations的每个元素包含两个表示变量名的字符串，数组values的每个元素是一个浮点数值。
如果equations[i]的两个变量名分别是Ai和Bi，那么Ai/Bi=values[i]。再给定一个数组queries，它的每个元素也包含两个变量名。
对于queries[j]的两个变量名Cj和Dj，请计算Cj/Dj的结果。假设任意values[i]大于0。如果不能计算，那么返回-1。
*/
//TODO ????????????what the fuck!
func calcEquation(equations [][]string, values []float64, queries [][]string) []float64 {
	//标记下标索引ids
	ids := make(map[string]int)
	for _, e := range equations {
		a := e[0]
		b := e[1]
		if _, has := ids[a]; !has {
			ids[a] = len(ids)
		}
		if _, has := ids[b]; !has {
			ids[b] = len(ids)
		}
	}
	n := len(ids)
	fa := make([]int, n)
	//记录权重值
	w := make([]float64, n)
	for i := range fa {
		fa[i] = i
		w[i] = 1
	}
	var find func(x int) int
	find = func(x int) int {
		if fa[x] != x {
			f := find(fa[x])
			w[x] *= w[fa[x]]
			fa[x] = f
		}
		return fa[x]
	}
	var merge func(from, to int, val float64)
	merge = func(from, to int, val float64) {
		fFrom := find(from)
		fTo := find(to)
		w[fFrom] = val * w[to] / w[from]
		fa[fFrom] = fTo
	}
	for i, e := range equations {
		merge(ids[e[0]], ids[e[1]], values[i])
	}
	ans := make([]float64, len(queries))
	for i, q := range queries {
		start, hasS := ids[q[0]]
		end, hasE := ids[q[1]]
		if hasS && hasE && find(start) == find(end) {
			ans[i] = w[start] / w[end]
		} else {
			ans[i] = -1
		}
	}
	return ans
}

//最长递增路径
/*
题目：输入一个整数矩阵，请求最长递增路径的长度。矩阵中的路径沿着上、下、左、右4个方向前行。例如，图15.14中矩阵的最长递增路径的长度为4，其中一条最长的递增路径为3→4→5→8，如阴影部分所示
*/

// 超出时间限制
func longestIncreasingPath(matrix [][]int) int {
	ans := 0
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			ans = max(ans, dfsLIP(matrix, i, j, 1))
		}
	}
	return ans

}

func dfsLIP(matrix [][]int, i, j, length int) int {
	if i < 0 || i >= len(matrix) || j < 0 || j >= len(matrix[0]) {
		return length
	}
	top, under, left, right := length, length, length, length
	if i > 0 && matrix[i-1][j] > matrix[i][j] {
		top = dfsLIP(matrix, i-1, j, length+1)
	}
	if (i+1) < len(matrix) && matrix[i+1][j] > matrix[i][j] {
		under = dfsLIP(matrix, i+1, j, length+1)
	}
	if j > 0 && matrix[i][j-1] > matrix[i][j] {
		left = dfsLIP(matrix, i, j-1, length+1)
	}
	if j+1 < len(matrix[0]) && matrix[i][j+1] > matrix[i][j] {
		right = dfsLIP(matrix, i, j+1, length+1)
	}
	v1 := max(top, under)
	v2 := max(left, right)
	return max(v1, v2)

}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 代码ac
func longestIncreasingPathV2(matrix [][]int) int {
	ans := 0
	lengths := make([][]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		lengths[i] = make([]int, len(matrix[0]))
	}
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			ans = max(ans, dfsLIPV2(matrix, lengths, i, j))
		}
	}
	return ans

}

func dfsLIPV2(matrix, lengths [][]int, i, j int) int {
	if lengths[i][j] != 0 {
		return lengths[i][j]
	}
	dirs := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	lth := 1
	for _, dir := range dirs {
		r := i + dir[0]
		c := j + dir[1]
		if r >= 0 && r < len(matrix) && c >= 0 && c < len(matrix[0]) && matrix[r][c] > matrix[i][j] {
			path := dfsLIPV2(matrix, lengths, r, c)
			lth = max(path+1, lth)
		}
	}
	lengths[i][j] = lth
	return lth
}

//拓补排序
/*
拓扑排序是指对一个有向无环图的节点进行排序之后得到的序列。如果存在一条从节点A指向节点B的边，那么在拓扑排序的序列中节点A出现在节点B的前面。一个有向无环图可以有一个或多个拓扑排序序列，但无向图或有环的有向图都不存在拓扑排序。在讨论有向无环图拓扑排序算法之前先介绍两个概念：入度和出度。
节点v的入度指的是以节点v为终点的边的数目，而节点v的出度是指以节点v为起点的边的数目。例如，在图15.16（a）的有向图中，节点2的入度是1，出度是2。一种常用的拓扑排序算法是每次从有向无环图中取出一个入度为0的节点添加到拓扑排序序列之中，然后删除该节点及所有以它为起点的边。重复这个步骤，直到图为空或图中不存在入度为0的节点。
如果最终图为空，那么图是有向无环图，此时就找到了该图的一个拓扑排序序列。如果最终图不为空并且已经不存在入度为0的节点，那么图中一定有环。
*/

//课程顺序
/*
n门课程的编号为0～n-1。输入一个数组prerequisites，它的每个元素prerequisites[i]表示两门课程的先修顺序。如果prerequisites[i]=[ai，bi]，那么必须先修完bi才能修ai。
请根据总课程数n和表示先修顺序的prerequisites得出一个可行的修课序列。如果有多个可行的修课序列，则输出任意一个可行的序列；如果没有可行的修课序列，则输出空序列
*/

func findOrder(numCourses int, prerequisites [][]int) []int {
	in := make(map[int]int, numCourses)
	for i := 0; i < numCourses; i++ {
		in[i] = 0
	}
	outNodes := make(map[int][]int)
	for _, prerequisite := range prerequisites {
		f := prerequisite[1]
		c := prerequisite[0]
		in[c]++
		outNodes[f] = append(outNodes[f], c)
	}
	var queue []int
	for node, cnt := range in {
		if cnt == 0 {
			queue = append(queue, node)
		}
	}

	var ret []int

	for len(queue) > 0 {
		node := queue[0]
		ret = append(ret, node)
		for _, next := range outNodes[node] {
			in[next]--
			if in[next] == 0 {
				queue = append(queue, next)
			}
		}
		queue = queue[1:]
	}
	if len(ret) == numCourses {
		return ret
	}
	return []int{}
}

//外星文字典
/*
一种外星语言的字母都是英文字母，但字母的顺序未知。给定该语言排序的单词列表，请推测可能的字母顺序。如果有多个可能的顺序，则返回任意一个。
如果没有满足条件的字母顺序，则返回空字符串。例如，如果输入排序的单词列表为["ac"，"ab"，"bc"，"zc"，"zb"]，那么一个可能的字母顺序是"acbz"。
*/

func alienOrder(words []string) string {
	graph := make(map[uint8][]uint8)
	charFlag := make(map[uint8]map[uint8]bool)
	in := make(map[uint8]int)
	for _, word := range words {
		for i := 0; i < len(word); i++ {
			if _, ok := in[word[i]]; !ok {
				in[word[i]] = 0
			}
		}
	}
	for i := 0; i < len(words)-1; i++ {
		pre := words[i]
		next := words[i+1]
		if len(pre) > len(next) && strings.HasPrefix(pre, next) {
			return ""
		}
		for j := 0; j < len(pre) && j < len(next); j++ {
			ch1 := pre[j]
			ch2 := next[j]
			if ch1 != ch2 && !charFlag[ch1][ch2] {
				graph[ch1] = append(graph[ch1], ch2)
				in[ch2]++
				break
			}
		}
	}
	var queue []uint8
	for node, cnt := range in {
		if cnt == 0 {
			queue = append(queue, node)
		}
	}
	var sb strings.Builder
	for len(queue) > 0 {
		node := queue[0]
		sb.WriteByte(node)
		for _, next := range graph[node] {
			in[next]--
			if in[next] == 0 {
				queue = append(queue, next)
			}
		}
		queue = queue[1:]
	}
	if len(sb.String()) == len(in) {
		return sb.String()
	}
	return ""
}

//重建序列
/*
长度为n的数组org是数字1～n的一个排列，seqs是若干序列，请判断数组org是否为可以由seqs重建的唯一序列。重建的序列是指seqs所有序列的最短公共超序列，即seqs中的任意序列都是该序列的子序列。
*/

func sequenceReconstruction(nums []int, sequences [][]int) bool {
	in := make(map[int]int)
	for _, num := range nums {
		in[num] = 0
	}
	node2outNodes := make(map[int][]int)
	flag := make(map[int]map[int]bool)
	for _, seq := range sequences {
		for i := 0; i < len(seq)-1; i++ {
			cur := seq[i]
			next := seq[i+1]
			in[next]++
			if !flag[cur][next] {
				node2outNodes[cur] = append(node2outNodes[cur], next)
			}
		}
	}
	var queue []int

	for node, cnt := range in {
		if cnt == 0 {
			queue = append(queue, node)
		}
	}
	for _, num := range nums {
		if len(queue) == 1 && queue[0] == num {
			for _, outNode := range node2outNodes[num] {
				in[outNode]--
				if in[outNode] == 0 {
					queue = append(queue, outNode)
				}
			}
			queue = queue[1:]
		} else {
			return false
		}

	}
	return true

}

//并查集
/*
并查集是一种树形的数据结构，用来表示不相交集合的数据。并查集中的每个子集是一棵树，每个元素是某棵树中的一个节点。树中的每个节点有一个指向父节点的指针，树的根节点的指针指向它自己。
例如，图15.21（a）所示是一个由两棵树组成的并查集。并查集支持两种操作，即合并和查找。
合并操作将两个子集合并成一个集合，只需要将一个子集对应的树的根节点的指针指向另一个子集对应的树的根节点。将图15.21（a）中的并查集的两个子集合并之后的并查集如图15.21（b）所示
*/

/*
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。



示例 1：
输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
示例 2：

输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
输出：3
提示：
1 <= n <= 200
n == isConnected.length
n == isConnected[i].length
isConnected[i][j] 为 1 或 0
isConnected[i][i] == 1
isConnected[i][j] == isConnected[j][i]

*/

func findCircleNum(isConnected [][]int) int {
	lth := len(isConnected)
	father := make([]int, lth)
	for i := 0; i < lth; i++ {
		father[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	group := lth
	merge := func(x, y int) {
		a, b := find(x), find(y)
		if a == b {
			return
		}
		father[b] = a
		group--
	}

	for i := 0; i < len(isConnected); i++ {
		for j := 0; j < len(isConnected[0]); j++ {
			if isConnected[i][j] == 1 {
				merge(i, j)
			}
		}
	}

	return group
}

//相似的字符串
/*
如果交换字符串X中的两个字符就能得到字符串Y，那么两个字符串X和Y相似。例如，字符串"tars"和"rats"相似（交换下标为0和2的两个字符）、字符串"rats"和"arts"相似（交换下标为0和1的字符），
但字符串"star"和"tars"不相似。输入一个字符串数组，根据字符串的相似性分组，请问能把输入数组分成几组？
如果一个字符串至少和一组字符串中的一个相似，那么它就可以放到该组中。假设输入数组中的所有字符串的长度相同并且两两互为变位词。
例如，输入数组为["tars"，"rats"，"arts"，"star"]，可以分成两组，一组为{"tars"，"rats"，"arts"}，另一组为{"star"}
*/

func numSimilarGroups(strs []string) int {
	lth := len(strs)
	father := make([]int, lth)
	for i := 0; i < lth; i++ {
		father[i] = i
	}
	group := lth
	var find func(x int) int
	find = func(x int) int {
		if x != father[x] {
			father[x] = find(father[x])
		}
		return father[x]
	}
	merge := func(x, y int) {
		a, b := find(x), find(y)
		if a == b {
			return
		}
		father[a] = b
		group--
	}
	for i := 0; i < len(strs)-1; i++ {
		for j := i + 1; j < len(strs); j++ {
			if isSimilar(strs[i], strs[j]) {
				merge(i, j)
			}
		}
	}
	return group
}

func isSimilar(str1, str2 string) bool {
	diffCnt := 0
	for i := 0; i < len(str1); i++ {
		if str1[i] != str2[i] {
			diffCnt++
		}
	}
	return diffCnt <= 2
}

//多余的边
/*
树可以看成无环的无向图。在一个包含n个节点（节点标号为从1到n）的树中添加一条边连接任意两个节点，这棵树就会变成一个有环的图。给定一个在树中添加了一条边的图，请找出这条多余的边（用这条边连接的两个节点表示）。
输入的图用一个二维数组edges表示，数组中的每个元素是一条边的两个节点[u，v]（u＜v）。如果有多个答案，请输出在数组edges中最后出现的边
*/

func findRedundantConnection(edges [][]int) []int {
	var ret []int
	lth := len(edges)
	for i := 0; i < lth; i++ {
		ufds := ConstructFDS(lth + 1)
		for j := 0; j < lth; j++ {
			if j == i {
				continue
			}
			ufds.merge(edges[j][0], edges[j][1])
		}
		if ufds.Group == 1 {
			ret = edges[i]
		}
	}
	return ret

}

func findRedundantConnectionV2(edges [][]int) []int {
	var ret []int
	lth := len(edges)
	ufds := ConstructFDS(lth + 1)
	for _, edge := range edges {
		if !ufds.union(edge[0], edge[1]) {
			return edge
		}
	}
	return ret
}

type UFDS struct {
	Father []int
	Group  int
}

func ConstructFDS(lth int) UFDS {
	udfs := UFDS{
		Father: make([]int, lth),
		Group:  lth - 1,
	}
	for i := 0; i < lth; i++ {
		udfs.Father[i] = i
	}
	return udfs
}

func (u *UFDS) find(x int) int {
	if u.Father[x] != x {
		u.Father[x] = u.find(u.Father[x])
	}
	return u.Father[x]
}

func (u *UFDS) merge(x, y int) {
	a, b := u.find(x), u.find(y)
	if a == b {
		return
	}
	u.Father[a] = b
	u.Group--
}

func (u *UFDS) union(x, y int) bool {
	a, b := u.find(x), u.find(y)
	if a != b {
		u.Father[a] = b
		return true
	}
	return false
}

//最长连续序列
/*
输入一个无序的整数数组，请计算最长的连续数值序列的长度。例如，输入数组[10，5，9，2，4，3]，则最长的连续数值序列是[2，3，4，5]，因此输出4
*/

func longestConsecutive(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	num2exist := make(map[int]bool)
	for _, num := range nums {
		num2exist[num] = true
	}
	maxLth := 0
	cpmFlag := make(map[int]bool)
	for i := 0; i < len(nums); i++ {
		if !cpmFlag[nums[i]] {
			cnt := 1
			small := nums[i] - 1
			big := nums[i] + 1
			for {
				if !num2exist[small] {
					break
				}
				cnt++
				cpmFlag[small] = true
				small--

			}
			for {
				if !num2exist[big] {
					break
				}
				cnt++
				cpmFlag[big] = true
				big++

			}
			cpmFlag[nums[i]] = true
			if cnt > maxLth {
				maxLth = cnt
			}
		}

	}

	return maxLth
}

func longestConsecutiveV2(nums []int) int {
	father := make(map[int]int)
	num2exist := make(map[int]bool)
	for _, num := range nums {
		father[num] = num
		num2exist[num] = true
	}
	var find func(x int) int
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	merge := func(x, y int) {
		a, b := find(x), find(y)
		if a == b {
			return
		}
		father[a] = b
	}
	for _, num := range nums {
		if num2exist[num-1] {
			merge(num-1, num)
		}
		if num2exist[num+1] {
			merge(num, num+1)
		}
	}
	for _, num := range nums {
		find(num)
	}
	maxLth := 0
	for k, v := range father {
		if (v - k + 1) > maxLth {
			maxLth = v - k + 1
		}
	}
	return maxLth
}

func longestConsecutiveV3(nums []int) int {
	father := make(map[int]int)
	count := make(map[int]int)
	for _, num := range nums {
		father[num] = num
		count[num] = 1
	}
	var find func(x int) int
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	merge := func(x, y int) {
		a, b := find(x), find(y)
		if a == b {
			return
		}
		father[a] = b
		count[b] += count[a]
	}
	for _, num := range nums {
		if _, ok := father[num+1]; ok {
			merge(num, num+1)
		}
	}
	maxLth := 0
	for _, v := range count {
		if v > maxLth {
			maxLth = v
		}
	}
	return maxLth
}
