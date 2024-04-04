package high_frequency_summary

import (
	"fmt"
	"testing"
)

func TestDecodeString(t *testing.T) {
	fmt.Println(decodeString("3[a]2[bc]"))
}

func TestMyAtoi(t *testing.T) {
	fmt.Println(myAtoi("words and 987"))
}

func TestRotateString(t *testing.T) {
	fmt.Println(rotateString("abcde", "cdeab"))
}

func TestAddStrings(t *testing.T) {
	fmt.Println(addStrings("1", "9"))
}

func TestLongestDecomposition(t *testing.T) {
	fmt.Println(longestDecomposition("elvtoelvto"))
}
