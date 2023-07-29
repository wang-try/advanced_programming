package algorithm

/**
*排序算法
**/

// 冒泡排序
func bubbleSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}

// 插入排序
func insertSort(nums []int) {
	for i := 1; i < len(nums); i++ {
		for j := i - 1; j >= 0; j-- {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}

	}
}

// 选择排序
func selectSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		minNum := nums[i]
		flag := i
		for j := i; j < len(nums); j++ {
			if nums[j] < minNum {
				minNum = nums[j]
				flag = j
			}
		}
		nums[i], nums[flag] = nums[flag], nums[i]
	}
}

// 快排
func quickSort(nums []int, start int, end int) {
	if start < end {
		pos := partition(nums, start, end)
		quickSort(nums, start, pos-1)
		quickSort(nums, pos+1, end)
	}
}

func partition(nums []int, start int, end int) int {
	pivot := nums[end]
	smallIndex := start
	for i := start; i < end; i++ {
		if nums[i] < pivot {
			nums[i], nums[smallIndex] = nums[smallIndex], nums[i]
			smallIndex++
		}
	}
	nums[smallIndex], nums[end] = nums[end], nums[smallIndex]
	return smallIndex

}

// 堆排序
func heapSort(nums []int) {
	length := len(nums)
	// 调整序列的前半部分元素，调整完之后第一个元素是序列的最大的元素
	for i := length/2 - 1; i >= 0; i-- {
		heapAdjust(nums, i, length)
	}
	for i := length - 1; i > 0; i-- {
		// 将第1个元素与当前最后一个元素交换，保证当前的最后一个位置的元素都是现在的这个序列中最大的
		nums[0], nums[i] = nums[i], nums[0]
		heapAdjust(nums, 0, i-1)
	}
}

func heapAdjust(nums []int, start int, len int) {
	var child int
	//若子节点指标在范围内才做比较
	for ; 2*start < len; start = child {
		child = 2*start + 1
		//先比较两个子节点大小，选择最大的
		if child < len-1 && nums[child+1] > nums[child] {
			child++
		}
		//如果父亲节点小于子节点，则交换
		if nums[start] < nums[child] {
			nums[start], nums[child] = nums[child], nums[start]
		} else {
			//如果父节点大于子节点代表调整完毕，直接跳出函数
			break
		}
	}
}

// 归并排序
func mergeSort(nums []int, start int, end int) {
	if start < end {
		mid := (start + end) / 2
		mergeSort(nums, start, mid)
		mergeSort(nums, mid+1, end)
		merge(nums, start, mid, end)
	}
}
func merge(nums []int, start int, mid int, end int) {
	temp := make([]int, 0, end+1)
	i := start
	j := mid + 1

	for i <= mid && j <= end {
		if nums[i] > nums[j] {
			temp = append(temp, nums[j])
			j++
		} else {
			temp = append(temp, nums[i])
			i++
		}
	}
	for i <= mid {
		temp = append(temp, nums[i])
		i++
	}
	for j <= end {
		temp = append(temp, nums[j])
		j++
	}
	for i, j = 0, start; j <= end; i, j = i+1, j+1 {
		nums[j] = temp[i]
	}
}
