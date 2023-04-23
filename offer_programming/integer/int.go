package integer

import (
	"math"
	"strconv"
	"strings"
)

// 整数除法
func divide(dividend int, divisor int) int {
	if dividend == math.MinInt32 && divisor == -1 {
		return math.MaxInt32
	}
	dendSigned := 1
	if dividend < 0 {
		dividend *= -1
		dendSigned = 0
	}
	sorSigned := 1
	if divisor < 0 {
		divisor *= -1
		sorSigned = 0
	}
	signed := 1
	if sorSigned^dendSigned == 1 {
		signed = -1
	}
	ret := 0
	for dividend >= divisor {
		tmp, m := divisor, 1
		for dividend >= tmp<<1 {
			tmp, m = tmp<<1, m<<1
		}
		dividend -= tmp
		ret += m
	}
	return signed * ret
}

// AddBinary 二进制加法
func AddBinary(a, b string) string {
	var sb strings.Builder
	i := len(a) - 1
	j := len(b) - 1
	carry := 0
	var list []int
	for i >= 0 || j >= 0 {
		digitA := 0
		if i >= 0 {
			digitA = int(a[i] - '0')
			i--
		}
		digitB := 0
		if j >= 0 {
			digitB = int(b[j] - '0')
			j--
		}
		sum := digitA + digitB + carry
		if sum >= 2 {
			carry = 1
			sum -= 2
		}
		list = append(list, sum)
	}
	if carry == 1 {
		list = append(list, carry)
	}
	for i := len(list) - 1; i >= 0; i-- {
		sb.WriteString(strconv.Itoa(list[i]))
	}
	return sb.String()

}

// 前n个数字二级制形式中1的个数
func CountBits(num int) []int {
	if num == 0 {
		return []int{0}
	}
	var res []int
	res = append(res, 0)
	for i := 1; i <= num; i++ {
		cnt := 0
		numTmp := i
		for numTmp > 0 {
			if numTmp&1 == 1 {
				cnt++
			}
			numTmp >>= 1
		}
		res = append(res, cnt)
	}
	return res
}

func CountBitsV2(num int) []int {
	ret := make([]int, num+1)
	for i := 0; i <= num; i++ {
		j := i
		for j != 0 {
			ret[i]++
			j = j & (j - 1)
		}
	}
	return ret
}

func CountBitsV3(num int) []int {
	ret := make([]int, num+1)
	//动态规划？
	for i := 1; i <= num; i++ {
		ret[i] = ret[i&(i-1)] + 1
	}
	return ret
}

func CountBitsV4(num int) []int {
	res := make([]int, num+1)
	for i := 1; i <= num; i++ {
		res[i] = res[i>>1] + (i & 1)
	}
	return res
}

// 只出现一次的数字,其他都出现了3次
func SingleNumber(nums []int) int {
	bitSums := make([]int, 32)
	for _, num := range nums {
		for i := 0; i < 32; i++ {
			bitSums[i] += (num >> (31 - i)) & 1
		}
	}
	ret := 0
	for i := 0; i < 32; i++ {
		ret = (ret << 1) + bitSums[i]%3
	}
	return ret
}

// 单词长度的最大乘积
func MaxProduct(words []string) int {
	ch2exist := make([][26]bool, len(words))
	for pos, word := range words {
		for i := 0; i < len(word); i++ {
			ch2exist[pos][word[i]-'a'] = true
		}
	}
	maxP := 0
	for i := 0; i < len(words); i++ {
		for j := i + 1; j < len(words); j++ {
			k := 0
			for ; k < 26; k++ {
				if ch2exist[i][k] == ch2exist[j][k] && ch2exist[i][k] == true {
					break
				}
			}
			if k == 26 {
				if len(words[i])*len(words[j]) > maxP {
					maxP = len(words[i]) * len(words[j])
				}

			}
		}
	}
	return maxP
}

func MaxProductV2(words []string) int {
	bitMaps := make([]int, len(words))
	for i, word := range words {
		for j := 0; j < len(word); j++ {
			bitMaps[i] |= 1 << (word[j] - 'a')
		}

	}
	maxP := 0
	for i := 0; i < len(words); i++ {
		for j := i + 1; j < len(words); j++ {
			if bitMaps[i]&bitMaps[j] == 0 {
				if len(words[i])*len(words[j]) > maxP {
					maxP = len(words[i]) * len(words[j])
				}
			}
		}
	}
	return maxP
}
