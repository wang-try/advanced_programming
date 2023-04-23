package array

func TwoSum(numbers []int, target int) []int {
	lhs := 0
	rhs := len(numbers) - 1

	for lhs < rhs {
		sum := numbers[lhs] + numbers[rhs]
		if target == sum {
			return []int{lhs, rhs}
		} else if target > sum {
			rhs--
		} else {
			lhs++
		}
	}
	return nil
}
