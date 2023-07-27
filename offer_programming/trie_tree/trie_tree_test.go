package trie_tree

import (
	"fmt"
	"testing"
)

func TestReplaceWords(t *testing.T) {
	fmt.Println(ReplaceWords([]string{"cat", "bat", "rat"}, "the cattle was rattled by the battery"))
}

// [[], [["hello","hallo","leetcode","judge", "judgg"]], ["hello"], ["hallo"], ["hell"], ["leetcodd"], ["judge"], ["juggg"]]
func TestConstructorD(t *testing.T) {
	md := ConstructorD()
	md.BuildDict([]string{"hello", "hallo", "leetcode", "judge", "judgg"})
	//fmt.Println(md.Search("hello"))
	//fmt.Println(md.Search("hhllo"))
	fmt.Println(md.Search("juggg"))
}

func TestMinimumLengthEncoding(t *testing.T) {
	fmt.Println(MinimumLengthEncoding([]string{"me", "time", "bell"}))
	fmt.Println(MinimumLengthEncodingV2([]string{"me", "time", "bell"}))
}

// ["MapSum", "insert", "sum", "insert", "sum"]
// [[], ["apple",3], ["apple"], ["app",2], ["ap"]]
// 添加到测试用例
// 输出
// [null,null,6,null,5]
// 预期结果
// [null,null,3,null,5]
func TestConstructorMs(t *testing.T) {
	ms := ConstructorMs()
	ms.Insert("apple", 3)
	fmt.Println(ms.Sum("apple"))
	//ms.Insert("app", 2)
	//fmt.Println(ms.Sum("ap"))
}

func TestFindMaximumXOR(t *testing.T) {
	fmt.Println(FindMaximumXOR([]int{1, 3, 4, 7}))
}
