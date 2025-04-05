package main

import (
	"time"
)

//有问题

type MsgType uint8

const (
	Prepare MsgType = iota // 准备阶段请求
	Promise                // 准备阶段响应
	Propose                // 提交阶段请求
	Accept                 // 提交阶段响应
)

type Message struct {
	Type   MsgType
	From   int    // 发送方ID
	To     int    // 接收方ID
	Number int    // 提案号
	Value  string // 提案值
}

type Proposer struct {
	id        int          // 节点唯一标识
	round     int          // 当前轮次
	number    int          // 提案号(由轮次和节点ID组合生成)
	value     string       // 提案值
	acceptors map[int]bool // 记录Acceptor响应状态
	net       Network      // 网络通信接口
}

// 提案号生成规则采用(round << 16) | id的方式确保全局唯一性
func (p *Proposer) Prepare() []Message {
	p.round++
	p.number = (p.round << 16) | p.id
	var msgs []Message
	for acceptor := range p.acceptors {
		msgs = append(msgs, Message{
			Type:   Prepare,
			From:   p.id,
			To:     acceptor,
			Number: p.number,
		})
	}
	return msgs
}

func (p *Proposer) Accept() []Message {
	var msgs []Message
	for acceptor := range p.acceptors {
		msgs = append(msgs, Message{
			Type:   Propose,
			From:   p.id,
			To:     acceptor,
			Number: p.number,
			Value:  p.value,
		})
	}
	return msgs
}

type Acceptor struct {
	id             int     // 节点ID
	promiseNumber  int     // 已承诺的提案号
	acceptedNumber int     // 已接受提案号
	acceptedValue  string  // 已接受提案值
	learners       []int   // Learner节点列表
	net            Network // 网络通信接口
}

func (a *Acceptor) HandlePrepare(msg Message) Message {
	if msg.Number > a.promiseNumber {
		a.promiseNumber = msg.Number
		return Message{
			Type:   Promise,
			From:   a.id,
			To:     msg.From,
			Number: a.acceptedNumber,
			Value:  a.acceptedValue,
		}
	}
	return Message{Type: Promise, Number: -1} // 拒绝
}

func (a *Acceptor) HandlePropose(msg Message) Message {
	if msg.Number >= a.promiseNumber {
		a.promiseNumber = msg.Number
		a.acceptedNumber = msg.Number
		a.acceptedValue = msg.Value
		return Message{Type: Accept}
	}
	return Message{Type: Accept, Number: -1} // 拒绝
}

type Network struct {
	queue map[int]chan Message // 节点通信队列
}

// 发送消息到指定节点
func (n *Network) Send(msg Message) {
	n.queue[msg.To] <- msg
}

// 带超时的消息接收
func (n *Network) RecvFrom(id int, timeout time.Duration) (Message, bool) {
	select {
	case msg := <-n.queue[id]:
		return msg, true
	case <-time.After(timeout):
		return Message{}, false
	}
}

//func TestSingleProposer(t *testing.T) {
//	network := newNetwork(1, 2, 3)
//	proposer := NewProposer(1, []int{2, 3})
//	go proposer.Run("valueA")
//
//	time.Sleep(1 * time.Second)
//	if learner.Chosen() != "valueA" {
//		t.Error("Consensus failed")
//	}
//}
