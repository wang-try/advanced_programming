package main

/*
Proxy模式又叫做代理模式，是构造型的设计模式之一，它可以为其他对象提供一种代理（Proxy）以控制对这个对象的访问。
	所谓代理，是指具有与代理元（被代理的对象）具有相同的接口的类，客户端必须通过代理与被代理的目标类交互，而代理一般在交互的过程中（交互前后），进行某些特别的处理。
用一个日常可见的案例来理解“代理”的概念，如下图：

这里假设有一个“自己”的角色，正在玩一款网络游戏。称这个网络游戏就是代理模式的“Subject”，表示要做一件事的目标或者对象事件主题。
（1）“自己”有一个给游戏角色升级的需求或者任务，当然“自己”可以独自完成游戏任务的升级。
（2）或者“自己”也可以邀请以为更加擅长游戏的“游戏代练”来完成升级这件事，这个代练就是“Proxy”代理。
（3）“游戏代练”不仅能够完成升级的任务需求，还可以额外做一些附加的能力。比如打到一些好的游戏装备、加入公会等等周边收益。
所以代理的出现实则是为了能够覆盖“自己”的原本的需求，且可以额外做其他功能，这种额外创建的类是不影响已有的“自己”和“网络游戏”的的关系。
是额外添加，在设计模式原则上，是符合“开闭原则”思想。那么当需要给“自己”增加额外功能的时候，又不想改变自己，那么就选择邀请一位”代理”来完成吧。


代理模式的优缺点
优点：
(1) 能够协调调用者和被调用者，在一定程度上降低了系统的耦合度。
(2) 客户端可以针对抽象主题角色进行编程，增加和更换代理类无须修改源代码，符合开闭原则，系统具有较好的灵活性和可扩展性。
缺点：
(1) 代理实现较为复杂。
适用场景
	为其他对象提供一种代理以控制对这个对象的访问。

*/

import "fmt"

type Goods struct {
	Kind string //商品种类
	Fact bool   //商品真伪
}

// =========== 抽象层 ===========
// 抽象的购物主题Subject
type Shopping interface {
	Buy(goods *Goods) //某任务
}

// =========== 实现层 ===========
// 具体的购物主题， 实现了shopping， 去韩国购物
type KoreaShopping struct{}

func (ks *KoreaShopping) Buy(goods *Goods) {
	fmt.Println("去韩国进行了购物, 买了 ", goods.Kind)
}

// 具体的购物主题， 实现了shopping， 去美国购物
type AmericanShopping struct{}

func (as *AmericanShopping) Buy(goods *Goods) {
	fmt.Println("去美国进行了购物, 买了 ", goods.Kind)
}

// 具体的购物主题， 实现了shopping， 去非洲购物
type AfrikaShopping struct{}

func (as *AfrikaShopping) Buy(goods *Goods) {
	fmt.Println("去非洲进行了购物, 买了 ", goods.Kind)
}

// 海外的代理
type OverseasProxy struct {
	shopping Shopping //代理某个主题，这里是抽象类型
}

func (op *OverseasProxy) Buy(goods *Goods) {
	// 1. 先验货
	if op.distinguish(goods) == true {
		//2. 进行购买
		op.shopping.Buy(goods) //调用原被代理的具体主题任务
		//3 海关安检
		op.check(goods)
	}
}

// 创建一个代理,并且配置关联被代理的主题
func NewProxy(shopping Shopping) Shopping {
	return &OverseasProxy{shopping}
}

// 验货流程
func (op *OverseasProxy) distinguish(goods *Goods) bool {
	fmt.Println("对[", goods.Kind, "]进行了辨别真伪.")
	if goods.Fact == false {
		fmt.Println("发现假货", goods.Kind, ", 不应该购买。")
	}
	return goods.Fact
}

// 安检流程
func (op *OverseasProxy) check(goods *Goods) {
	fmt.Println("对[", goods.Kind, "] 进行了海关检查， 成功的带回祖国")
}

func use() {
	g1 := Goods{
		Kind: "韩国面膜",
		Fact: true,
	}

	g2 := Goods{
		Kind: "CET4证书",
		Fact: false,
	}

	//如果不使用代理来完成从韩国购买任务
	var shopping Shopping
	shopping = new(KoreaShopping) //具体的购买主题

	//1-先验货
	if g1.Fact == true {
		fmt.Println("对[", g1.Kind, "]进行了辨别真伪.")
		//2-去韩国购买
		shopping.Buy(&g1)
		//3-海关安检
		fmt.Println("对[", g1.Kind, "] 进行了海关检查， 成功的带回祖国")
	}

	fmt.Println("---------------以下是 使用 代理模式-------")
	var overseasProxy Shopping
	overseasProxy = NewProxy(shopping)
	overseasProxy.Buy(&g1)
	overseasProxy.Buy(&g2)
}
