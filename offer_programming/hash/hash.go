package hash

import (
	"math/rand"
	"strconv"
	"strings"
)

// 面试题30：插入、删除和随机访问都是O（1）的容器
type RandomizedSet struct {
	buff  []int
	cache map[int]int
}

/** Initialize your data structure here. */
func Constructor() RandomizedSet {
	return RandomizedSet{
		buff:  make([]int, 0),
		cache: make(map[int]int),
	}
}

/** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
func (this *RandomizedSet) Insert(val int) bool {
	if _, ok := this.cache[val]; ok {
		return false
	}
	this.buff = append(this.buff, val)
	this.cache[val] = len(this.buff) - 1
	return true
}

/** Removes a value from the set. Returns true if the set contained the specified element. */
func (this *RandomizedSet) Remove(val int) bool {
	if idx, ok := this.cache[val]; ok {
		this.cache[this.buff[len(this.buff)-1]] = idx
		this.buff[idx], this.buff[len(this.buff)-1] = this.buff[len(this.buff)-1], this.buff[idx]
		this.buff = this.buff[:len(this.buff)-1]
		delete(this.cache, val)
		return true
	}
	return false

}

/** Get a random element from the set. */
func (this *RandomizedSet) GetRandom() int {
	if len(this.buff) > 0 {
		return this.buff[rand.Intn(len(this.buff))]
	}
	return 0
}

// 请设计实现一个最近最少使用（Least Recently Used，LRU）缓存，要求如下两个操作的时间复杂度都是O（1）。
type LRUCache struct {
	key2index map[int]int
	Nodes     []*LRUNode
	Tail      *LRUNode
	Head      *LRUNode
	Lth       int
	Cap       int
}

type LRUNode struct {
	Key  int
	Val  int
	Pre  *LRUNode
	Next *LRUNode
}

func LRUConstructor(capacity int) LRUCache {
	return LRUCache{
		key2index: make(map[int]int),
		Nodes:     make([]*LRUNode, capacity),
		Tail:      nil,
		Head:      nil,
		Lth:       0,
		Cap:       capacity,
	}

}

func (this *LRUCache) Get(key int) int {
	if index, ok := this.key2index[key]; ok {
		this.Put(key, this.Nodes[index].Val)
		return this.Nodes[index].Val
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	if this.Cap == 0 {
		return
	}

	if index, ok := this.key2index[key]; !ok {
		//新增
		node := &LRUNode{
			Key:  key,
			Val:  value,
			Pre:  nil,
			Next: nil,
		}
		if this.Lth < this.Cap {
			if this.Lth == 0 {
				this.Head = node
				this.Tail = node
			} else {
				node.Next = this.Head
				this.Head.Pre = node
				this.Head = node
			}
			this.Lth++
			this.Nodes[this.Lth-1] = node
			this.key2index[key] = this.Lth - 1

			return
		}
		//驱逐尾节点
		node.Next = this.Head
		this.Head.Pre = node
		this.Head = node

		this.Nodes[this.key2index[this.Tail.Key]] = node
		this.key2index[key] = this.key2index[this.Tail.Key]
		delete(this.key2index, this.Tail.Key)
		this.Tail = this.Tail.Pre
		this.Tail.Next = nil

	} else {
		//存在
		node := this.Nodes[index]
		if this.Lth == 1 {
			node.Val = value
			return
		}
		if node == this.Head {
			node.Val = value
			return
		}

		next := node.Next
		pre := node.Pre
		node.Val = value

		pre.Next = next
		node.Next = this.Head
		this.Head.Pre = node
		this.Head = node

		if node == this.Tail {
			this.Tail = pre
			return
		}
		next.Pre = pre
	}
}

//有效的变位词
/*
给定两个字符串s和t，请判断它们是不是一组变位词。在一组变位词中，它们中的字符及每个字符出现的次数都相同，但字符的顺序不能相同。
例如，"anagram"和"nagaram"就是一组变位词。
*/

func IsAnagram(str1, str2 string) bool {
	if len(str1) != len(str2) {
		return false
	}
	ch2cnt := make(map[uint8]int)
	for i := 0; i < len(str1); i++ {
		ch2cnt[str1[i]]++
	}
	for i := 0; i < len(str2); i++ {
		if ch2cnt[str2[i]] > 0 {
			ch2cnt[str2[i]]--
		} else {
			return false
		}
	}
	return true
}

//变位词组
/*
给定一组单词，请将它们按照变位词分组。例如，输入一组单词["eat"，"tea"，"tan"，"ate"，"nat"，"bat"]，这组单词可以分成3组，
分别是["eat"，"tea"，"ate"]、["tan"，"nat"]和["bat"]。假设单词中只包含英文小写字母。
*/
func GroupAnagram(strs []string) [][]string {
	dict := make(map[[26]int][]string)
	for _, v := range strs {
		ana := [26]int{}
		for _, c := range v {
			ana[c-'a']++
		}
		dict[ana] = append(dict[ana], v)
	}

	res := make([][]string, 0)
	for _, v := range dict {
		res = append(res, v)
	}
	return res

}

func GroupAnagramV2(strs []string) [][]string {
	hash := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101}
	var res [][]string
	dict := make(map[uint64][]string)
	for _, str := range strs {
		wordHash := uint64(1)
		for _, c := range str {
			wordHash *= uint64(hash[c-'a'])
		}
		dict[wordHash] = append(dict[wordHash], str)
	}
	for _, v := range dict {
		res = append(res, v)
	}
	return res
}

//外星语言是否排序
/*
有一门外星语言，它的字母表刚好包含所有的英文小写字母，只是字母表的顺序不同。给定一组单词和字母表顺序，请判断这些单词是否按照字母表的顺序排序。
例如，输入一组单词["offer"，"is"，"coming"]，以及字母表顺序"zyxwvutsrqponmlkjihgfedcba"，由于字母'o'在字母表中位于'i'的前面，因此单词"offer"排在"is"的前面；
同样，由于字母'i'在字母表中位于'c'的前面，因此单词"is"排在"coming"的前面。因此，这一组单词是按照字母表顺序排序的，应该输出true。
*/

func IsAlienSorted(words []string, order string) bool {
	ch2order := make(map[uint8]int)
	for i := 0; i < len(order); i++ {
		ch2order[order[i]] = i
	}
	for i := 0; i < len(words)-1; i++ {
		cur := words[i]
		next := words[i+1]
		cIndex := 0
		nIndex := 0
		for cIndex < len(cur) && nIndex < len(next) {
			if ch2order[cur[cIndex]] > ch2order[next[nIndex]] {
				return false
			} else if ch2order[cur[cIndex]] < ch2order[next[nIndex]] {
				break
			} else {
				cIndex++
				nIndex++
			}
		}
		if cIndex != len(cur) && nIndex == len(next) {
			return false
		}

	}
	return true
}

//最小时间差
/*
给定一组范围在00：00至23：59的时间，求任意两个时间之间的最小时间差。例如，输入时间数组["23:50"，"23：59"，"00：00"]，"23：59"和"00：00"之间只有1分钟的间隔，是最小的时间差。
*/

func FindMinDifference(timePoints []string) int {
	var hash [1440]bool
	for _, timePoint := range timePoints {
		hm := strings.Split(timePoint, ":")
		h, _ := strconv.Atoi(hm[0])
		m, _ := strconv.Atoi(hm[1])
		allMinute := h*60 + m
		if hash[allMinute] {
			return 0
		}
		hash[allMinute] = true
	}
	prev := -1
	first := -1
	last := -1
	minDiff := len(hash) - 1
	for i := 0; i < len(hash); i++ {
		if hash[i] {
			if prev > 0 {
				if i-prev < minDiff {
					minDiff = i - prev
				}
			}
			if first < 0 {
				first = i
			}
			prev = i
			last = i
		}
	}
	//要把第1个时间加上1440分钟表示第2天的同一时间，求出它与最后一个时间的时间差
	if minDiff > (len(hash) - last + first) {
		minDiff = len(hash) - last + first
	}
	return minDiff
}
