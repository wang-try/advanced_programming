package trie_tree

import (
	"sort"
	"strings"
)

//实现前缀树
/*
题目：请设计实现一棵前缀树Trie，它有如下操作。
	● 函数insert，在前缀树中添加一个字符串。
	● 函数search，查找字符串。如果前缀树中包含该字符串，则返回true；否则返回false。
	● 函数startWith，查找字符串前缀。如果前缀树中包含以该前缀开头的字符串，则返回true；否则返回false。
*/
type Trie struct {
	Next    [26]*Trie
	IsWorld bool
}

/** Initialize your data structure here. */
func Constructor() Trie {
	return Trie{}
}

/** Inserts a word into the trie. */
func (this *Trie) Insert(word string) {
	cur := this
	for i := 0; i < len(word); i++ {
		if cur.Next[word[i]-'a'] == nil {
			cur.Next[word[i]-'a'] = &Trie{}
		}
		cur = cur.Next[word[i]-'a']
		if i == len(word)-1 {
			cur.IsWorld = true
		}
	}
}

/** Returns if the word is in the trie. */
func (this *Trie) Search(word string) bool {
	cur := this
	for _, ch := range word {
		if cur.Next[ch-'a'] != nil {
			cur = cur.Next[ch-'a']
		} else {
			return false
		}

	}
	return cur.IsWorld
}

/** Returns if there is any word in the trie that starts with the given prefix. */
func (this *Trie) StartsWith(prefix string) bool {
	for _, alph := range prefix {
		if this.Next[alph-'a'] != nil {
			this = this.Next[alph-'a']
		} else {
			return false
		}
	}
	return true
}

func (this *Trie) GetPrefix(word string) (bool, string) {
	var sb strings.Builder
	cur := this
	for i := 0; i < len(word); i++ {
		if cur.Next[word[i]-'a'] != nil {
			sb.WriteByte(word[i])
			if cur.Next[word[i]-'a'].IsWorld {
				return true, sb.String()
			}
			cur = cur.Next[word[i]-'a']
		} else {
			return false, ""
		}
	}
	return false, ""
}

func (this *Trie) GetWordByPrefix(prefix string) []string {
	var ret []string
	cur := this
	for _, ch := range prefix {
		if cur.Next[ch-'a'] == nil {
			return ret
		}
		cur = cur.Next[ch-'a']
	}
	dfsGetWords(cur, prefix, &ret)
	return ret
}

func dfsGetWords(t *Trie, word string, ret *[]string) {
	if t == nil {
		return
	}
	if t.IsWorld {
		*ret = append(*ret, word)
	}
	for i := uint8(0); i < 26; i++ {
		if t.Next[i] != nil {
			dfsGetWords(t.Next[i], word+string([]byte{i + 'a'}), ret)
		}
	}
}

//替换单词
/*
英语中有一个概念叫词根。在词根后面加上若干字符就能拼出更长的单词。例如，"an"是一个词根，在它后面加上"other"就能得到另一个单词"another"。
现在给定一个由词根组成的字典和一个英语句子，如果句子中的单词在字典中有它的词根，则用它的词根替换该单词；如果单词没有词根，则保留该单词。请输出替换后的句子。
例如，如果词根字典包含字符串["cat"，"bat"，"rat"]，英语句子为"the cattle was rattled by the battery"，则替换之后的句子是"the cat was rat by the bat"。

dictionary =["a", "aa", "aaa", "aaaa"]
sentence ="a aa a aaaa aaa aaa aaa aaaaaa bbb baba ababa"
输出
"a a a a a a a a bbb a a"
预期结果
"a a a a a a a a bbb baba a"

*/
func ReplaceWords(dictionary []string, sentence string) string {
	trie := Constructor()
	for _, dict := range dictionary {
		trie.Insert(dict)
	}
	words := strings.Split(sentence, " ")
	list := make([]string, 0, len(words))
	for _, word := range words {
		if isHasPrefix, prefix := trie.GetPrefix(word); isHasPrefix {
			list = append(list, prefix)
		} else {
			list = append(list, word)
		}

	}
	return strings.Join(list, " ")
}

//神奇的字典
/*
题目：请实现有如下两个操作的神奇字典。
● 函数buildDict，输入单词数组用来创建一个字典。
● 函数search，输入一个单词，判断能否修改该单词中的一个字符，使修改之后的单词是字典中的一个单词。
*/
type MagicDictionary struct {
	TrieTree *Trie
}

/** Initialize your data structure here. */
func ConstructorD() MagicDictionary {
	return MagicDictionary{TrieTree: &Trie{
		Next:    [26]*Trie{},
		IsWorld: false,
	}}
}

func (this *MagicDictionary) BuildDict(dictionary []string) {
	for _, dict := range dictionary {
		this.TrieTree.Insert(dict)
	}
}

func (this *MagicDictionary) Search(searchWord string) bool {
	return dfs(this.TrieTree, 0, 0, searchWord)
}

func dfs(node *Trie, index, editCnt int, searchWord string) bool {
	if node == nil {
		return false
	}
	if node.IsWorld && index == len(searchWord) && editCnt == 1 {
		return true
	}
	if editCnt <= 1 && index < len(searchWord) {
		found := false
		for i := uint8(0); i < 26 && !found; i++ {
			next := editCnt
			if i != searchWord[index]-'a' {
				next = editCnt + 1
			}
			found = dfs(node.Next[i], index+1, next, searchWord)
		}
		return found
	}
	return false
}

type SuffixTrie struct {
	Next [26]*SuffixTrie
}

func ConstructorS() *SuffixTrie {
	return &SuffixTrie{}
}

/** Inserts a word into the trie. */
func (this *SuffixTrie) Insert(word string) {
	cur := this
	for i := len(word) - 1; i >= 0; i-- {
		if cur.Next[word[i]-'a'] == nil {
			cur.Next[word[i]-'a'] = &SuffixTrie{}
		}
		cur = cur.Next[word[i]-'a']
	}
}

/** Returns if the word is in the trie. */

/** Returns if there is any word in the trie that starts with the given prefix. */
func (this *SuffixTrie) EndsWith(suffix string) bool {
	for i := len(suffix) - 1; i >= 0; i-- {
		if this.Next[suffix[i]-'a'] != nil {
			this = this.Next[suffix[i]-'a']
		} else {
			return false
		}
	}
	return true
}

//最短的单词编码
/*
输入一个包含n个单词的数组，可以把它们编码成一个字符串和n个下标。例如，单词数组["time"，"me"，"bell"]可以编码成一个字符串"time＃bell＃"，然后这些单词就可以通过下标[0，2，5]得到。
对于每个下标，都可以从编码得到的字符串中相应的位置开始扫描，直到遇到'＃'字符前所经过的子字符串为单词数组中的一个单词。
例如，从"time＃bell＃"下标为2的位置开始扫描，直到遇到'＃'前经过子字符串"me"是给定单词数组的第2个单词。给定一个单词数组，
请问按照上述规则把这些单词编码之后得到的最短字符串的长度是多少？如果输入的是字符串数组["time"，"me"，"bell"]，那么编码之后最短的字符串是"time＃bell＃"，长度是10
*/

func MinimumLengthEncoding(words []string) int {
	sort.Slice(words, func(i, j int) bool { return len(words[i]) > len(words[j]) })
	if len(words) > 0 {
		lth := 0
		st := ConstructorS()
		st.Insert(words[0])
		lth += len(words[0]) + 1
		for i := 1; i < len(words); i++ {
			if !st.EndsWith(words[i]) {
				st.Insert(words[i])
				lth += len(words[i]) + 1
			}
		}
		return lth
	}
	return 0
}
func MinimumLengthEncodingV2(words []string) int {
	st := ConstructorS()
	for _, word := range words {
		st.Insert(word)
	}
	total := 0
	dfsLth(st, 1, &total)
	return total
}

func dfsLth(st *SuffixTrie, lth int, total *int) {
	isLeaf := true
	for _, child := range st.Next {
		if child != nil {
			isLeaf = false
			dfsLth(child, lth+1, total)
		}
	}
	if isLeaf {
		*total += lth
	}

}

//单词之和
/*
题目：请设计实现一个类型MapSum，它有如下两个操作。
	● 函数insert，输入一个字符串和一个整数，在数据集合中添加一个字符串及其对应的值。如果数据集合中已经包含该字符串，则将该字符串对应的值替换成新值。
	● 函数sum，输入一个字符串，返回数据集合中所有以该字符串为前缀的字符串对应的值之和。
*/

type MapSum struct {
	T   *Trie
	k2v map[string]int
}

/** Initialize your data structure here. */
func ConstructorMs() MapSum {
	return MapSum{
		T:   &Trie{},
		k2v: make(map[string]int),
	}
}

func (this *MapSum) Insert(key string, val int) {
	this.k2v[key] = val
	this.T.Insert(key)
}

func (this *MapSum) Sum(prefix string) int {
	words := this.T.GetWordByPrefix(prefix)
	sum := 0
	for _, word := range words {
		sum += this.k2v[word]
	}
	return sum
}

//最大的异或
/*
输入一个整数数组（每个数字都大于或等于0），请计算其中任意两个数字的异或的最大值。例如，在数组[1，3，4，7]中，3和4的异或结果最大，异或结果为7。
*/

type BinaryTireTree struct {
	Next [2]*BinaryTireTree
}

func FindMaximumXOR(nums []int) int {
	head := &BinaryTireTree{}
	for _, num := range nums {
		this := head
		for i := 31; i >= 0; i-- {
			bit := (num >> i) & 1
			if this.Next[bit] == nil {
				this.Next[bit] = &BinaryTireTree{}
			}
			this = this.Next[bit]
		}
	}
	max := 0
	for _, num := range nums {
		node := head
		xor := 0
		for i := 31; i >= 0; i-- {
			bit := (num >> i) & 1
			if node.Next[1-bit] != nil {
				xor = (xor << 1) + 1
				node = node.Next[1-bit]
			} else {
				xor = xor << 1
				node = node.Next[bit]
			}
		}
		if xor > max {
			max = xor
		}
	}
	return max
}
