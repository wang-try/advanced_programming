package string

import (
	"fmt"
	"testing"
)

func TestCheckInclusion(t *testing.T) {
	fmt.Println(CheckInclusion("ab", "eidboaoo"))
}

func TestFindAnagrams(t *testing.T) {
	fmt.Println(FindAnagrams("cbadabacg", "abc"))
}

func TestLengthOfLongestSubstring(t *testing.T) {
	fmt.Println(LengthOfLongestSubstring("babcca"))
	fmt.Println(LengthOfLongestSubstringV2("babcca"))
}

func TestMinWindow(t *testing.T) {
	fmt.Println(MinWindow("ADDBANCAD", "ABC"))
}

func TestValidPalindrome(t *testing.T) {
	fmt.Println(ValidPalindrome("abca"))
}

func TestCountSubstrings(t *testing.T) {
	fmt.Println(CountSubstrings("fdsklf"))
}
