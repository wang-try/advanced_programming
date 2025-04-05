package main

import (
	"fmt"
	"math/rand"
	"time"
)

// 提议结构
type Proposal struct {
	Number int
	Value  interface{}
}

// 接受者结构
type Acceptor struct {
	AcceptedProposal Proposal
	HighestN         int
}

func (a *Acceptor) ReceivePrepare(n int) (int, bool) {
	if n > a.HighestN {
		a.HighestN = n
		return a.AcceptedProposal.Number, true
	}
	return 0, false
}

func (a *Acceptor) ReceiveAccept(p Proposal) bool {
	if p.Number >= a.HighestN {
		a.AcceptedProposal = p
		a.HighestN = p.Number
		return true
	}
	return false
}

// 提议者结构
type Proposer struct {
	Proposals []Proposal
	N         int
}

func (p *Proposer) Prepare(acceptors []*Acceptor) (int, interface{}, bool) {
	p.N++
	var highestN int
	var highestV interface{}
	for _, a := range acceptors {
		n, ok := a.ReceivePrepare(p.N)
		if ok {
			if n > highestN {
				highestN = n
				highestV = a.AcceptedProposal.Value
			}
		}
	}
	return highestN, highestV, highestN > 0
}

func (p *Proposer) Accept(acceptors []*Acceptor, v interface{}) bool {
	proposal := Proposal{
		Number: p.N,
		Value:  v,
	}
	acceptedCount := 0
	for _, a := range acceptors {
		if a.ReceiveAccept(proposal) {
			acceptedCount++
		}
	}
	return acceptedCount > len(acceptors)/2
}

// 学习者结构
type Learner struct {
	AcceptedProposal Proposal
}

func (l *Learner) Learn(proposal Proposal) {
	l.AcceptedProposal = proposal
}

func main() {
	// 初始化接受者
	acceptor1 := &Acceptor{}
	acceptor2 := &Acceptor{}
	acceptor3 := &Acceptor{}
	acceptors := []*Acceptor{acceptor1, acceptor2, acceptor3}

	// 初始化提议者
	proposer := &Proposer{}

	// 初始化学习者
	learner := &Learner{}

	// 提议者发起提议
	for i := 0; i < 5; i++ {
		rand.Seed(time.Now().UnixNano())
		value := rand.Intn(100)
		_, v, ok := proposer.Prepare(acceptors)
		var proposalValue interface{}
		if ok {
			proposalValue = v
		} else {
			proposalValue = value
		}
		if proposer.Accept(acceptors, proposalValue) {
			fmt.Printf("Proposal %d accepted with value: %v\n", proposer.N, proposalValue)
			learner.Learn(Proposal{Number: proposer.N, Value: proposalValue})
		} else {
			fmt.Printf("Proposal %d not accepted\n", proposer.N)
		}
	}

	fmt.Printf("Learned value: %v\n", learner.AcceptedProposal.Value)
}
