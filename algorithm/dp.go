package algorithm

// 70. Climbing Stairs
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// 338. Counting Bits
func countBits(n int) []int {
	ans := make([]int, n+1)
	for i := 1; i <= n; i++ {
		ans[i] = ans[i/2] + i&1
	}
	return ans
}

// 392. Is Subsequence
func isSubsequence(s string, t string) bool {
	var idx, cnt int
	if len(s) == 0 {
		return true
	}

	for i := 0; i < len(t); i++ {
		if t[i] == s[idx] {
			cnt++
			idx++
		}
		if cnt == len(s) {
			return true
		}
	}
	return false
}

// 509. Fibonacci Number
func fib(n int) int {
	if n == 0 {
		return 0
	}
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

/*
若某个人拿到了1，则表示输了，所以当拿到2的时候，一定会赢，因为取个因数1，然后把剩下的1丢给对手就赢了。
对于大于2的N，最后都会先减小到2，所以其实这个游戏就是个争2的游戏。
对于一个奇数N来说，小于N的因子x一定也是个奇数，则留给对手的 N-x 一定是个偶数。
而对于偶数N来说，我们可以取1，然后变成一个奇数丢给对手，所以拿到偶数的人，将奇数丢给对手后，
下一轮自己还会拿到偶数，这样当N不断减小后，最终一定会拿到2，所以会赢
*/

// 1025. Divisor Game
func divisorGame(n int) bool {
	return n%2 == 0
}

func divisorGameV2(n int) bool {
	dp := make([]bool, n+1)
	dp[0], dp[1] = false, false

	for i := 2; i <= n; i++ {
		for j := 1; j < i; j++ {
			if i%j == 0 && !dp[i-j] {
				dp[i] = true
				break
			}
		}
	}

	return dp[n]
}
