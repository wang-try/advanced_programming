package str

import (
	"fmt"
	"testing"
)

func TestLeftShiftN(t *testing.T) {
	fmt.Println(LeftShiftN("abcdef", 3))
}

func TestStringContain(t *testing.T) {
	//fmt.Println(StringContain("ABCD", "AE"))
	fmt.Println(StringContainV2("ABVD", "AX"))
}

func TestAllPermutation(t *testing.T) {
	fmt.Println(AllPermutation("abc"))
}

func TestStr2int(t *testing.T) {
	fmt.Println(Str2Int("43535"))
}

func TestGetLeastNumbers(t *testing.T) {
	arr := []int{0, 1, 1, 2, 4, 4, 1, 3, 3, 2}
	fmt.Println(GetLeastNumbers(arr, 6))
}

func TestQuickSort(t *testing.T) {
	arr := []int{0, 1, 1, 2, 4, 4, 1, 3, 3, 2}
	QuickSort(arr, 0, 9)
	fmt.Println(arr)
}
