package integer

import (
	"fmt"
	"testing"
)

func TestAddBinary(t *testing.T) {
	fmt.Println(AddBinary("100", "1111111"))
}

func TestCountBits(t *testing.T) {
	fmt.Println(CountBits(8))
}

func TestCountBitsV2(t *testing.T) {
	fmt.Println(CountBitsV4(8))
}

func TestSingleNumber(t *testing.T) {
	fmt.Println(SingleNumber([]int{23, 9, 23, 100, 100, 4, 23, 4, 4, 100}))
}

func TestMaxProduct(t *testing.T) {
	fmt.Println(MaxProductV2([]string{"abcd", "ab", "ac", "khgh"}))
}
