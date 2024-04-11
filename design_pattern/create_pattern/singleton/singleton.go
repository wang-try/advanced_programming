package main

import (
	"fmt"
	"sync"
)

/*

优点：
(1) 单例模式提供了对唯一实例的受控访问。
(2) 节约系统资源。由于在系统内存中只存在一个对象。

缺点：
(1) 扩展略难。单例模式中没有抽象层。
(2) 单例类的职责过重。

 适用场景
(1) 系统只需要一个实例对象，如系统要求提供一个唯一的序列号生成器或资源管理器，或者需要考虑资源消耗太大而只允许创建一个对象。
(2) 客户调用类的单个实例只允许使用一个公共访问点，除了该公共访问点，不能通过其他途径访问该实例。




三个要点：
		一是某个类只能有一个实例；
		二是它必须自行创建这个实例；
		三是它必须自行向整个系统提供这个实例。
*/

/*
	保证一个类永远只能有一个对象
*/

// 1、保证这个类非公有化，外界不能通过这个类直接创建一个对象
//
//	那么这个类就应该变得非公有访问 类名称首字母要小写
//
// 懒汉模式
type singelton struct{}

// 2、但是还要有一个指针可以指向这个唯一对象，但是这个指针永远不能改变方向
//
//	Golang中没有常指针概念，所以只能通过将这个指针私有化不让外部模块访问
var instance *singelton = new(singelton)

// 3、如果全部为私有化，那么外部模块将永远无法访问到这个类和对象，
//
//	所以需要对外提供一个方法来获取这个唯一实例对象
//	注意：这个方法是否可以定义为singelton的一个成员方法呢？
//	    答案是不能，因为如果为成员方法就必须要先访问对象、再访问函数
//	     但是类和对象目前都已经私有化，外界无法访问，所以这个方法一定是一个全局普通函数
func GetInstance() *singelton {
	return instance
}

func (s *singelton) SomeThing() {
	fmt.Println("单例对象的某方法")
}

func main() {
	s := GetInstance()
	s.SomeThing()
}

// 恶汉模式
type singeltonE struct{}

var instanceE *singeltonE

func GetInstanceE() *singeltonE {
	//只有首次GetInstance()方法被调用，才会生成这个单例的实例
	if instanceE == nil {
		instanceE = new(singeltonE)
		return instanceE
	}

	//接下来的GetInstance直接返回已经申请的实例即可
	return instanceE
}

func (s *singelton) SomeThingE() {
	fmt.Println("单例对象的某方法")
}

//线程安全

var once sync.Once

type singeltonV3 struct{}

var instanceV3 *singeltonV3

func GetInstanceV3() *singeltonV3 {

	once.Do(func() {
		instanceV3 = new(singeltonV3)
	})

	return instanceV3
}

func (s *singeltonV3) SomeThingV3() {
	fmt.Println("单例对象的某方法")
}