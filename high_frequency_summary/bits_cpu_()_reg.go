package high_frequency_summary

import (
	"container/list"
	"strconv"
	"strings"
)

// leetcode231. 2 的幂
func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

// leetcode136. 只出现一次的数字
func singleNumber(nums []int) int {
	ret := nums[0]
	for i := 1; i < len(nums); i++ {
		ret ^= nums[i]
	}
	return ret
}

// leetcodeLCR 133. 位 1 的个数
func hammingWeight(num uint32) int {
	cnt := 0
	for i := 0; i < 32; i++ {
		if num&1 == 1 {
			cnt++
		}
		num >>= 1
	}
	return cnt
}

// leetcode20. 有效的括号
func isValid(s string) bool {
	var stack []uint8
	for i := 0; i < len(s); i++ {
		if s[i] == '{' || s[i] == '[' || s[i] == '(' {
			stack = append(stack, s[i])
		} else {
			if len(stack) == 0 {
				return false
			}
			cmp := stack[len(stack)-1]
			if (cmp == '(' && s[i] == ')') || (cmp == '{' && s[i] == '}') || (cmp == '[' && s[i] == ']') {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		}

	}
	return len(stack) == 0
}

// leetcode32. 最长有效括号
func longestValidParentheses(s string) int {
	maxAns := 0
	dp := make([]int, len(s))
	for i := 1; i < len(s); i++ {
		if s[i] == ')' {
			if s[i-1] == '(' {
				if i >= 2 {
					dp[i] = dp[i-2] + 2
				} else {
					dp[i] = 2
				}
			} else if i-dp[i-1] > 0 && s[i-dp[i-1]-1] == '(' {
				if i-dp[i-1] >= 2 {
					dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
				} else {
					dp[i] = dp[i-1] + 2
				}
			}
			maxAns = max(maxAns, dp[i])
		}
	}
	return maxAns
}

func longestValidParenthesesV2(s string) int {
	maxAns := 0
	stack := []int{}
	stack = append(stack, -1)
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			stack = append(stack, i)
		} else {
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				stack = append(stack, i)
			} else {
				maxAns = max(maxAns, i-stack[len(stack)-1])
			}
		}
	}
	return maxAns
}

// leetcode22. 括号生成
func generateParenthesis(n int) []string {
	var dfs func(left, right int, str string)
	var ans []string
	dfs = func(left, right int, str string) {
		if left == n && right == n {
			ans = append(ans, str)
			return
		}
		if left < n {
			dfs(left+1, right, str+"(")
		}
		if left > right {
			dfs(left, right+1, str+"(")
		}
	}
	dfs(0, 0, "")
	return ans
}

// leetcode 190. 颠倒二进制位
func reverseBits(num uint32) uint32 {
	var ans uint32
	for i := 0; i < 32; i++ {
		n := num & 1
		num >>= 1
		ans <<= 1
		ans += n
	}
	return ans

}

// leetcode93. 复原 IP 地址
func restoreIpAddresses(s string) []string {
	var ans []string
	var dfs func(index int, combinations []string)
	dfs = func(index int, combinations []string) {
		if len(combinations) > 4 {
			return
		}
		if index == len(s) && len(combinations) == 4 {
			ans = append(ans, strings.Join(combinations, "."))
			return
		}
		for i := index; i < len(s) && i <= index+2; i++ {
			numStr := s[index : i+1]
			num, _ := strconv.Atoi(numStr)
			if num >= 0 && num <= 255 {
				combinations = append(combinations, numStr)
				dfs(i+1, combinations)
				if numStr == "0" {
					break
				}
				combinations = combinations[:len(combinations)-1]
			}
		}
	}
	dfs(0, []string{})
	return ans
}

// leetcode468. 验证IP地址
func validIPAddress(queryIP string) string {
	if validIPV4(queryIP) {
		return "IPv4"
	}
	if ValidIPV6(queryIP) {
		return "IPv6"
	}

	return "Neither"

}

func validIPV4(queryIP string) bool {
	parts := strings.Split(queryIP, ".")
	if len(parts) != 4 {
		return false
	}
	for _, part := range parts {
		if len(part) > 1 && part[0] == '0' {
			return false
		}
		num, err := strconv.Atoi(part)
		if err != nil {
			return false
		}
		if num < 0 || num > 255 {
			return false
		}

	}
	return true
}

func ValidIPV6(queryIp string) bool {
	parts := strings.Split(queryIp, ":")
	if len(parts) != 8 {
		return false
	}
	for _, part := range parts {
		if len(part) < 1 || len(part) > 4 {
			return false
		}

		if _, err := strconv.ParseUint(part, 16, 64); err != nil {
			return false
		}
	}

	return true
}

// leetcode数字转换为十六进制数
func toHex(num int) string {
	if num == 0 {
		return "0"
	}
	sb := &strings.Builder{}
	for i := 7; i >= 0; i-- {
		val := num >> (4 * i) & 0xf
		if val > 0 || sb.Len() > 0 {
			var digit byte
			if val < 10 {
				digit = '0' + byte(val)
			} else {
				digit = 'a' + byte(val-10)
			}
			sb.WriteByte(digit)
		}
	}
	return sb.String()
}

// leetcode165. 比较版本号
func compareVersion(version1 string, version2 string) int {
	numbers1 := strings.Split(version1, ".")
	numbers2 := strings.Split(version2, ".")

	lth1 := len(numbers1)
	lth2 := len(numbers2)
	index1, index2 := 0, 0
	for index1 < lth1 || index2 < lth2 {
		number1, number2 := 0, 0
		if index1 < lth1 {
			number1, _ = strconv.Atoi(numbers1[index1])
			index1++
		}

		if index2 < lth2 {
			number2, _ = strconv.Atoi(numbers2[index2])
			index2++
		}
		if number1 > number2 {
			return 1
		}

		if number2 > number1 {
			return -1
		}

	}
	return 0

}

// leetcode67. 二进制求和
func addBinary(a string, b string) string {
	lth1 := len(a) - 1
	lth2 := len(b) - 1
	carry := 0
	ans := ""
	for lth1 >= 0 || lth2 >= 0 {
		sum := 0
		if lth1 >= 0 {
			sum += int(a[lth1] - '0')
			lth1--
		}

		if lth2 >= 0 {
			sum += int(b[lth2] - '0')
			lth2--
		}
		sum += carry
		carry = sum / 2
		num := sum % 2
		ans = strconv.Itoa(num) + ans
	}

	if carry > 0 {
		ans = "1" + ans
	}
	return ans
}

type entry struct {
	key, value, freq int // freq 表示这本书被看的次数
}

type LFUCache struct {
	capacity   int
	minFreq    int
	keyToNode  map[int]*list.Element
	freqToList map[int]*list.List
}

func ConstructorLFU(capacity int) LFUCache {
	return LFUCache{
		capacity:   capacity,
		keyToNode:  map[int]*list.Element{},
		freqToList: map[int]*list.List{},
	}
}

func (c *LFUCache) pushFront(e *entry) {
	if _, ok := c.freqToList[e.freq]; !ok {
		c.freqToList[e.freq] = list.New() // 双向链表
	}
	c.keyToNode[e.key] = c.freqToList[e.freq].PushFront(e)
}

func (c *LFUCache) getEntry(key int) *entry {
	node := c.keyToNode[key]
	if node == nil { // 没有这本书
		return nil
	}
	e := node.Value.(*entry)
	lst := c.freqToList[e.freq]
	lst.Remove(node)    // 把这本书抽出来
	if lst.Len() == 0 { // 抽出来后，这摞书是空的
		delete(c.freqToList, e.freq) // 移除空链表
		if c.minFreq == e.freq {     // 这摞书是最左边的
			c.minFreq++
		}
	}
	e.freq++       // 看书次数 +1
	c.pushFront(e) // 放在右边这摞书的最上面
	return e
}

func (c *LFUCache) Get(key int) int {
	if e := c.getEntry(key); e != nil { // 有这本书
		return e.value
	}
	return -1 // 没有这本书
}

func (c *LFUCache) Put(key, value int) {
	if e := c.getEntry(key); e != nil { // 有这本书
		e.value = value // 更新 value
		return
	}
	if len(c.keyToNode) == c.capacity { // 书太多了
		lst := c.freqToList[c.minFreq]                           // 最左边那摞书
		delete(c.keyToNode, lst.Remove(lst.Back()).(*entry).key) // 移除这摞书的最下面的书
		if lst.Len() == 0 {                                      // 这摞书是空的
			delete(c.freqToList, c.minFreq) // 移除空链表
		}
	}
	c.pushFront(&entry{key, value, 1}) // 新书放在「看过 1 次」的最上面
	c.minFreq = 1
}

type LFUCacheV2 struct {
	cache    map[int]*node
	freqMap  map[int]*listN
	minFreq  int
	capacity int
}

func ConstructorV2(capacity int) LFUCacheV2 {
	return LFUCacheV2{
		cache:    make(map[int]*node),
		freqMap:  make(map[int]*listN),
		capacity: capacity,
	}
}

func (this *LFUCacheV2) Get(key int) int {
	if this.capacity == 0 {
		return -1
	}

	n, ok := this.cache[key]
	if !ok {
		return -1
	}

	this.incrFreq(n)
	return n.val
}

func (this *LFUCacheV2) Put(key int, value int) {
	if this.capacity == 0 {
		return
	}

	n, ok := this.cache[key]
	if ok {
		n.val = value
		this.incrFreq(n)
		return
	}

	if len(this.cache) == this.capacity {
		l := this.freqMap[this.minFreq]
		delete(this.cache, l.removeBack().key)
	}
	n = &node{key: key, val: value, freq: 1}
	this.addNode(n)
	this.cache[key] = n
	this.minFreq = 1
}

func (this *LFUCacheV2) incrFreq(n *node) {
	l := this.freqMap[n.freq]
	l.remove(n)
	if l.empty() {
		delete(this.freqMap, n.freq)
		if n.freq == this.minFreq {
			this.minFreq++
		}
	}
	n.freq++
	this.addNode(n)
}

func (this *LFUCacheV2) addNode(n *node) {
	l, ok := this.freqMap[n.freq]
	if !ok {
		l = newList()
		this.freqMap[n.freq] = l
	}
	l.pushFront(n)
}

type node struct {
	key  int
	val  int
	freq int
	prev *node
	next *node
}

type listN struct {
	head *node
	tail *node
}

func newList() *listN {
	head := new(node)
	tail := new(node)
	head.next = tail
	tail.prev = head
	return &listN{
		head: head,
		tail: tail,
	}
}

func (l *listN) pushFront(n *node) {
	n.prev = l.head
	n.next = l.head.next
	l.head.next.prev = n
	l.head.next = n
}

func (l *listN) remove(n *node) {
	n.prev.next = n.next
	n.next.prev = n.prev
	n.next = nil
	n.prev = nil
}

func (l *listN) removeBack() *node {
	n := l.tail.prev
	l.remove(n)
	return n
}

func (l *listN) empty() bool {
	return l.head.next == l.tail
}

func checkDynasty(places []int) bool {
	d2cnt := make(map[int]int)
	minD := 100
	for _, place := range places {
		d2cnt[place]++
		if place != 0 {
			minD = min(minD, place)
		}
	}
	lth := len(places)
	num := minD
	for i := 1; i <= lth; i++ {
		if _, ok := d2cnt[num]; !ok {
			if cnt, ok := d2cnt[0]; ok && cnt > 0 {
				d2cnt[0]--
			} else {
				return false
			}
		}
		num++
	}
	return true
}
