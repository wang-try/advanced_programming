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

type Chopstick struct {
	sync.Mutex
}

type Philosopher struct {
	id                            int
	name                          string
	leftChopstick, rightChopstick *Chopstick
	status                        string
}

func (p *Philosopher) dinn() {
	for {
		mark(p, "冥想")
		randomPause(10)
		mark(p, "饿了")
		if p.id%2 == 1 { // 奇数
			p.leftChopstick.Lock() // 先尝试拿起左手边的筷子
			mark(p, "拿起左手筷子")
			p.rightChopstick.Lock() // 再尝试拿起右手边的筷子
			mark(p, "用膳")
			randomPause(10)
			p.rightChopstick.Unlock() // 先尝试放下右手边的筷子
			p.leftChopstick.Unlock()  // 再尝试放下左手边的筷子
		} else {
			p.rightChopstick.Lock() // 先尝试拿起右手边的筷子
			mark(p, "拿起右手筷子")
			p.leftChopstick.Lock() // 再尝试拿起左手边的筷子
			mark(p, "用膳")
			randomPause(10)
			p.leftChopstick.Unlock()  // 先尝试放下左手边的筷子
			p.rightChopstick.Unlock() // 再尝试放下右手边的筷子
		}
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
			id:             i,
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
