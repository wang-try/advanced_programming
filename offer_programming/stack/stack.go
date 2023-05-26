package stack

import "strconv"

//后缀表达式
/*
后缀表达式是一种算术表达式，它的操作符在操作数的后面。输入一个用字符串数组表示的后缀表达式，请输出该后缀表达式的计算结果。假设输入的一定是有效的后缀表达式。
例如，后缀表达式["2"，"1"，"3"，"*"，"+"]对应的算术表达式是“2+1*3”，因此输出它的计算结果5
*/
func EvalRPN(tokens []string) int {
	var stack []int
	for i := 0; i < len(tokens); i++ {
		if tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/" {
			if len(stack) >= 2 {
				var ret int
				lhs := stack[len(stack)-2]
				rhs := stack[len(stack)-1]
				switch tokens[i] {
				case "+":
					ret = lhs + rhs
				case "-":
					ret = lhs - rhs
				case "*":
					ret = lhs * rhs
				case "/":
					ret = lhs / rhs
				}
				stack = stack[:len(stack)-2]
				stack = append(stack, ret)
			}
		} else {
			num, _ := strconv.Atoi(tokens[i])
			stack = append(stack, num)
		}
	}
	return stack[0]
}

//小行星碰撞
/*
输入一个表示小行星的数组，数组中每个数字的绝对值表示小行星的大小，数字的正负号表示小行星运动的方向，正号表示向右飞行，负号表示向左飞行。如果两颗小行星相撞，那么体积较小的小行星将会爆炸最终消失，
体积较大的小行星不受影响。如果相撞的两颗小行星大小相同，那么它们都会爆炸消失。飞行方向相同的小行星永远不会相撞。求最终剩下的小行星。
例如，有6颗小行星[4，5，-6，4，8，-5]，如图6.2所示（箭头表示飞行的方向），它们相撞之后最终剩下3颗小行星[-6，4，8]。
*/

func AsteroidCollision(asteroids []int) []int {
	var stack []int
	for _, aster := range asteroids {
		tmp := -aster
		for len(stack) > 0 && stack[len(stack)-1] > 0 && stack[len(stack)-1] < tmp {
			stack = stack[:len(stack)-1]
		}
		if len(stack) > 0 && aster < 0 && stack[len(stack)-1] == tmp {
			stack = stack[:len(stack)-1]
		} else if aster > 0 || len(stack) == 0 || stack[len(stack)-1] < 0 {
			stack = append(stack, aster)
		}
	}
	return stack
}

//每日温度
/*
输入一个数组，它的每个数字是某天的温度。请计算每天需要等几天才会出现更高的温度。
例如，如果输入数组[35，31，33，36，34]，那么输出为[3，1，1，0，0]。由于第1天的温度是35℃，要等3天才会出现更高的温度36℃，因此对应的输出为3。
第4天的温度是36℃，后面没有更高的温度，它对应的输出是0。其他的以此类推。
*/
func DailyTemperatures(temperatures []int) []int {
	var stack []int
	lth := len(temperatures)
	ret := make([]int, lth)
	for i := 0; i < len(temperatures); i++ {
		j := len(stack) - 1
		for ; j >= 0 && temperatures[i] > temperatures[stack[j]]; j-- {
			ret[stack[j]] = i - stack[j]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ret
}

//直方图最大矩形面积
/*
直方图是由排列在同一基线上的相邻柱子组成的图形。输入一个由非负数组成的数组，数组中的数字是直方图中柱子的高。求直方图中最大矩形面积。假设直方图中柱子的宽都为1。
例如，输入数组[3，2，5，4，6，1，4，2]，其对应的直方图如图6.3所示，该直方图中最大矩形面积为12，如阴影部分所示。
*/
//此方法超时
func LargestRectangleArea(heights []int) int {
	max := 0
	for i := 0; i < len(heights); i++ {
		area := heights[i]
		lhs := i - 1
		rhs := i + 1
		for ; lhs >= 0 && heights[lhs] >= heights[i]; lhs-- {
			area += heights[i]
		}
		for ; rhs < len(heights) && heights[rhs] >= heights[i]; rhs++ {
			area += heights[i]
		}
		if area > max {
			max = area
		}
	}
	return max
}

// 此方法超时 O(n方)
func LargestRectangleAreaV2(heights []int) int {
	maxArea := 0
	for i := 0; i < len(heights); i++ {
		min := heights[i]
		for j := i; j < len(heights); j++ {
			if heights[j] < min {
				min = heights[j]
			}
			area := min * (j - i + 1)
			if area > maxArea {
				maxArea = area
			}
		}
	}
	return maxArea
}

// 此方法超时
func LargestRectangleAreaV3(heights []int) int {
	return helper(heights, 0, len(heights))
}
func helper(heights []int, start, end int) int {
	if start == end {
		return 0
	}
	if start+1 == end {
		return heights[start]
	}
	minIndex := start
	for i := start + 1; i < end; i++ {
		if heights[i] < heights[minIndex] {
			minIndex = i
		}
	}
	area := (end - start) * heights[minIndex]
	left := helper(heights, start, minIndex)
	right := helper(heights, minIndex+1, end)
	if left > area {
		area = left
	}
	if right > area {
		area = right
	}
	return area
}

func LargestRectangleAreaV4(heights []int) int {
	var stack []int
	maxArea := 0
	for i := 0; i < len(heights); i++ {
		j := len(stack) - 1
		for ; j >= 0 && heights[i] < heights[stack[j]]; j-- {
			height := heights[stack[j]]
			width := 0
			if j == 0 {
				width = i - (j - 1) - 1
			} else {
				width = i - stack[j-1] - 1
			}
			if height*width > maxArea {
				maxArea = height * width
			}
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}

	rhs := len(heights)
	for i := len(stack) - 1; i >= 0; i-- {
		lhs := i - 1
		if i > 0 {
			lhs = stack[i-1]
		}
		area := heights[stack[i]] * (rhs - lhs - 1)
		if area > maxArea {
			maxArea = area
		}

	}
	return maxArea
}

//矩阵中的最大矩形
/*
请在一个由0、1组成的矩阵中找出最大的只包含1的矩形并输出它的面积。例如，在图6.6的矩阵中，最大的只包含1的矩阵如阴影部分所示，它的面积是6。
1 0 1 0 0
0 0 1 1 1
1 1 1 1 1
1 0 0 1 0
*/
func MaximalRectangle(matrix []string) int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return 0
	}
	height := make([]int, len(matrix[0]))
	maxArea := 0
	for _, str := range matrix {
		for i := 0; i < len(str); i++ {
			if str[i] == '0' {
				height[i] = 0
			} else {
				height[i]++
			}
		}
		area := LargestRectangleAreaV4(height)
		if maxArea < area {
			maxArea = area
		}
	}
	return maxArea

}
