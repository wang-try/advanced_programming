package hash

import (
	"fmt"
	"testing"
)

func TestLRUConstructor(t *testing.T) {
	//[[3],[1,1],[2,2],[3,3],[4,4],[4],[3],[2],[1],[5,5],[1],[2],[3],[4],[5]]
	//[null,null,null,null,null,4,3,2,-1,null,-1,2,3,-1,5]
	//[null,null,null,null,null,4,3,2,-1,null,-1,2,-1,4,5]
	lru := LRUConstructor(3)
	lru.Put(1, 1)
	lru.Put(2, 2)
	lru.Put(3, 3)
	lru.Put(4, 4)
	lru.Get(4)
	lru.Get(3)
	lru.Get(2)
	lru.Get(1)
	lru.Put(5, 5)
	lru.Get(1)
	lru.Get(2)
	lru.Get(3)
	lru.Get(4)
	lru.Get(5)
}

func TestFindMinDifference(t *testing.T) {
	fmt.Println(FindMinDifference([]string{"05:31", "22:08", "00:35"}))
}
