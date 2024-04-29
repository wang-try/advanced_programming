package main

import (
	"fmt"
	"github.com/fatih/color"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// 全局锁，同一时刻只能有一个哲学家吃饭
type Chopstick struct {
	sync.Mutex
}
type Philosopher struct {
	id                            int
	name                          string
	leftChopstick, rightChopstick *Chopstick
	status                        string
}

var lock sync.Mutex

// TODO 有了全局锁，筷子锁没有用？
func (p *Philosopher) dinn() {
	for {
		//冥想
		mark(p, "冥想")
		randomPause(10)
		mark(p, "饿了")
		//左手筷子
		lock.Lock()
		p.leftChopstick.Lock()
		mark(p, "拿起左手筷子")
		p.rightChopstick.Lock()
		//右手筷子
		mark(p, "拿起右手筷子")
		//吃饭
		mark(p, "吃饭")
		//放下左手筷子
		randomPause(10)
		p.leftChopstick.Unlock()
		p.rightChopstick.Unlock()
		//放下右手筷子
		lock.Unlock()
	}
}

func randomPause(max int) {
	time.Sleep(time.Millisecond * time.Duration(rand.Intn(max)))
}

// 显示此哲学家的状态
func mark(p *Philosopher, action string) {
	fmt.Printf("%s开始%s\n", p.name, action)
	p.status = fmt.Sprintf("%s开始%s\n", p.name, action)
}

func main() {
	// 哲学家数量
	count := 5
	// 创建5根筷子
	chopsticks := make([]*Chopstick, count)
	for i := 0; i < count; i++ {
		chopsticks[i] = new(Chopstick)
	}
	names := []string{color.RedString("孔子"), color.MagentaString("庄子"), color.CyanString("墨子"), color.GreenString("孙子"), color.WhiteString("老子")}
	philosophers := make([]*Philosopher, count)
	for i := 0; i < count; i++ {
		philosophers[i] = &Philosopher{
			id:             0,
			name:           names[i],
			leftChopstick:  chopsticks[i],
			rightChopstick: chopsticks[(i+1)%count],
		}
		go philosophers[i].dinn()
	}

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	<-sigs
	fmt.Println("退出中... 每个哲学家的状态:")
	for _, p := range philosophers {
		fmt.Print(p.status)
	}

}
