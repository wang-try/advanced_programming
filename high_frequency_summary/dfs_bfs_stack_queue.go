package high_frequency_summary

// leetcode200. 岛屿数量
func numIslands(grid [][]byte) int {
	cnt := 0
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) {
			return
		}
		if grid[i][j] == '1' {
			grid[i][j] = '#'
			dfs(i+1, j)
			dfs(i-1, j)
			dfs(i, j+1)
			dfs(i, j-1)

		}

	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				dfs(i, j)
				cnt++
			}
		}
	}

	return cnt
}

// leetcode695. 岛屿的最大面积
func maxAreaOfIsland(grid [][]int) int {
	maxA := 0
	var dfs func(i, j int, area *int)
	dfs = func(i, j int, area *int) {
		if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) {
			return
		}
		if grid[i][j] == 1 {
			grid[i][j] = -1
			*area++
			dfs(i+1, j, area)
			dfs(i-1, j, area)
			dfs(i, j-1, area)
			dfs(i, j+1, area)
		}
	}

	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				area := 0
				dfs(i, j, &area)
				maxA = max(maxA, area)
			}
		}

	}
	return maxA
}

func maxAreaOfIslandV2(grid [][]int) int {
	maxArea := 0
	var dfs func(i, j int) int
	dfs = func(i, j int) int {
		area := 1
		dirs := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
		grid[i][j] = 0
		for _, dir := range dirs {
			r := i + dir[0]
			c := j + dir[1]
			if r >= 0 && r < len(grid) && c >= 0 && c < len(grid[0]) && grid[r][c] == 1 {
				area += dfs(r, c)
			}
		}
		return area
	}

	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				maxArea = max(dfs(i, j), maxArea)
			}
		}
	}
	return maxArea
}

// leetcode1254. 统计封闭岛屿的数目
func closedIsland(grid [][]int) int {
	var dfs func(i, j int) bool
	row := len(grid)
	column := len(grid[0])
	dfs = func(i, j int) bool {
		if i == 0 || j == 0 || i == row-1 || j == column-1 {
			if grid[i][j] == 1 {
				return true
			}
			return false
		}
		if grid[i][j] == 1 {
			return true
		}

		grid[i][j] = 1
		val1 := dfs(i+1, j)
		val2 := dfs(i, j+1)
		val3 := dfs(i-1, j)
		val4 := dfs(i, j-1)
		return val1 && val2 && val3 && val4
	}
	closed := 0
	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if grid[i][j] == 0 {
				if dfs(i, j) {
					closed++
				}
			}
		}
	}
	return closed

}

type MyQueue struct {
	StackA []int
	StackB []int
}

func ConstructorQ() MyQueue {
	return MyQueue{
		StackA: make([]int, 0, 10),
		StackB: make([]int, 0, 10),
	}
}

func (this *MyQueue) Push(x int) {
	this.StackA = append(this.StackA, x)
}

func (this *MyQueue) Pop() int {
	if len(this.StackB) == 0 {
		for len(this.StackA) > 0 {
			val := this.StackA[len(this.StackA)-1]
			this.StackB = append(this.StackB, val)
			this.StackA = this.StackA[:len(this.StackA)-1]
		}
	}
	val := this.StackB[len(this.StackB)-1]
	this.StackB = this.StackB[:len(this.StackB)-1]
	return val
}

func (this *MyQueue) Peek() int {
	if len(this.StackB) == 0 {
		for len(this.StackA) > 0 {
			val := this.StackA[len(this.StackA)-1]
			this.StackB = append(this.StackB, val)
			this.StackA = this.StackA[:len(this.StackA)-1]
		}
	}
	val := this.StackB[len(this.StackB)-1]
	return val
}

func (this *MyQueue) Empty() bool {
	return len(this.StackA)+len(this.StackB) == 0
}

type MyStack struct {
	QueueA []int
	QueueB []int
}

func ConstructorS() MyStack {
	return MyStack{
		QueueA: make([]int, 0, 10),
		QueueB: make([]int, 0, 10),
	}
}

func (this *MyStack) Push(x int) {
	this.QueueB = append(this.QueueB, x)
	for len(this.QueueA) > 0 {
		this.QueueB = append(this.QueueB, this.QueueA[0])
		this.QueueA = this.QueueA[1:]
	}
	this.QueueA, this.QueueB = this.QueueB, this.QueueA
}

func (this *MyStack) Pop() int {
	val := this.QueueA[0]
	this.QueueA = this.QueueA[1:]
	return val
}

func (this *MyStack) Top() int {
	return this.QueueA[0]
}

func (this *MyStack) Empty() bool {
	return len(this.QueueA) == 0
}

// leetcode994. 腐烂的橘子
func orangesRotting(grid [][]int) int {
	row := len(grid)
	column := len(grid[0])
	good := 0
	var badPosition [][2]int

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if grid[i][j] == 1 {
				good++
			} else if grid[i][j] == 2 {
				badPosition = append(badPosition, [2]int{i, j})
			}
		}
	}
	cnt := 0
	for len(badPosition) > 0 {
		lth := len(badPosition)
		dirs := [][]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
		for i := 0; i < lth; i++ {
			pos := badPosition[i]
			for _, dir := range dirs {
				c := pos[0] + dir[0]
				r := pos[1] + dir[1]
				if c >= 0 && c < row && r >= 0 && r < column && grid[c][r] == 1 {
					good--
					grid[c][r] = 2
					badPosition = append(badPosition, [2]int{c, r})
				}
			}
		}
		badPosition = badPosition[lth:]
		if len(badPosition) > 0 {
			cnt++
		}
	}

	if good > 0 {
		return -1
	}
	return cnt
}
