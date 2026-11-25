#!/usr/bin/env python3
"""Draw interface semantics as grids and ownership timelines."""

from draw_figures import Diagram, NAVY, TEAL, BLUE, AMBER, INK, MUTED, LINE, PALE, AQUA, GOLD


def broadcast_axes():
    d = Diagram(430, "二维 tile 的两种广播方向")
    d.label(20, 30, "X[M,N]：行尺度与通道尺度的轴含义不同", NAVY)
    d.text(20, 80, "行尺度 r[M] → [M,1]", 23, NAVY, 600)
    d.text(480, 80, "通道尺度 c[N] → [1,N]", 23, NAVY, 600)
    for panel in [0, 1]:
        bx = 100 + panel*460
        by = 160
        for i in range(4):
            for j in range(6):
                fill = ["#e0eff0","#d3e6e7","#c7dfe1","#bad6d9"][i] if panel==0 else ["#e1eaf3","#d6e2ef","#cbdbea","#bfd3e5","#b4cce0","#a9c4db"][j]
                d.rect(bx+j*45, by+i*34, 45, 34, fill, "#ffffff", r=0)
        if panel==0:
            for i in range(4):
                d.rect(20,by+i*34,35,34,["#e0eff0","#d3e6e7","#c7dfe1","#bad6d9"][i],"#ffffff",r=0)
                d.path([(55,by+i*34+17),(91,by+i*34+17)],color=TEAL)
            d.text(bx+135,333,"沿 N 轴复用同一个行值",20,TEAL,600,"middle")
        else:
            for j in range(6):
                d.rect(bx+j*45,105,45,28,["#e1eaf3","#d6e2ef","#cbdbea","#bfd3e5","#b4cce0","#a9c4db"][j],"#ffffff",r=0)
                d.path([(bx+j*45+22,133),(bx+j*45+22,153)],color=BLUE)
            d.text(bx+135,333,"沿 M 轴复用同一个通道值",20,BLUE,600,"middle")
    d.text(235, 382, "Y = X * r[:, None]", 23, NAVY, 600, "middle")
    d.text(695, 382, "Y = X * c[None, :]", 23, NAVY, 600, "middle")
    d.note("插入的单例轴决定广播方向；图中颜色表示同一份尺度被重复使用。")
    d.save("broadcast-axes.svg")


def block_window():
    d = Diagram(565, "块指针窗口与逻辑边界")
    d.label(20, 30, "shape=(64,128) · offsets=(16,96) · block_shape=(32,64)", NAVY)
    x,y=55,120
    for i in range(8):
        for j in range(8):
            d.rect(x+j*80,y+i*40,80,40,"#f3f5f7","#d5dde4",r=0)
    d.rect(535,200,160,160,AQUA,TEAL,r=0)
    d.rect(695,200,160,160,GOLD,AMBER,r=0)
    d.path([(535,188),(855,188)],color=TEAL)
    d.text(695,170,"窗口宽度 64",20,TEAL,600,"middle")
    for xx, label in [(55,"0"),(535,"96"),(695,"128"),(855,"160")]:
        d.text(xx,100,label,19,MUTED,anchor="middle")
    for yy, label in [(120,"0"),(200,"16"),(360,"48"),(440,"64")]:
        d.text(43,yy+6,label,18,MUTED,anchor="end")
    d.text(615,266,"有效 32 列",20,TEAL,600,"middle")
    d.text(615,302,"正常读 / 写",19,TEAL,anchor="middle")
    d.text(775,266,"越界 32 列",20,AMBER,600,"middle")
    d.text(775,302,"读补零 / 禁止写",18,AMBER,anchor="middle")
    d.text(55,485,"advance(ptr, (16,0)) 使 offsets 变为 (32,96)，不搬运数据。",21,NAVY)
    d.note("边界来自 shape；stride 按元素计数。补齐规则由对应 load/store 参数指定。")
    d.save("block-pointer-window.svg")


def runtime_lifecycle():
    d = Diagram(570, "异步提交与资源生命周期")
    d.label(20,30,"一个 producer 与跨 stream consumer 共用同一份 buffer",NAVY)
    start,end=160,865
    for yy,label in [(115,"Host"),(250,"Stream A"),(385,"Stream B")]:
        d.text(20,yy+8,label,21,NAVY,600)
        d.path([(start,yy),(end,yy)],arrow=True)
    d.box(168,79,130,60,"分配",fill=PALE,size=21)
    d.box(342,79,150,60,"提交 A",fill=PALE,size=21)
    d.box(711,79,150,60,"可回收",fill=GOLD,color=AMBER,size=21)
    d.rect(350,217,212,66,AQUA,TEAL,r=2)
    d.text(456,257,"producer",23,TEAL,600,"middle")
    d.path([(418,139),(418,207)],dashed=True)
    d.text(578,307,"完成事件",19,TEAL,anchor="start")
    d.path([(562,283),(562,385)],color=TEAL)
    d.rect(580,352,175,66,"#e5edf6",BLUE,r=2)
    d.text(668,392,"consumer",23,BLUE,600,"middle")
    d.text(480,406,"wait",20,MUTED)
    d.path([(755,352),(785,327),(785,139)],color=AMBER)
    d.path([(180,470),(755,470)],arrow=False,color=TEAL)
    d.path([(180,456),(180,484)],arrow=False,color=TEAL)
    d.path([(755,456),(755,484)],arrow=False,color=TEAL)
    d.text(468,507,"Tensor / LocalMem owner 在所有访问结束前保持存活",21,TEAL,600,"middle")
    d.note("事件与等待表达跨 stream 依赖；Host 提交返回不表示 producer 或 consumer 已完成。")
    d.save("runtime-lifecycle.svg")


def bf16_rounding():
    d = Diagram(440, "BF16 最近偶数舍入的两个相邻中点")
    d.label(20, 30, "最近偶数舍入 · 相同距离下选择保留位为偶数的一端", NAVY)
    d.path([(55, 195), (850, 195)], arrow=True)
    for xx, value, bits, parity in [(80, "1", "0x3f80", "偶"),
                                    (450, "1 + 2⁻⁷", "0x3f81", "奇"),
                                    (820, "1 + 2⁻⁶", "0x3f82", "偶")]:
        d.rect(xx-5, 190, 10, 10, NAVY, NAVY, r=5)
        d.text(xx, 235, value, 23, NAVY, 600, "middle")
        d.text(xx, 270, bits + " · " + parity, 19, MUTED, anchor="middle")
    for xx in [265, 635]:
        d.rect(xx-5, 190, 10, 10, AMBER, AMBER, r=5)
        d.text(xx, 235, "正中间", 19, AMBER, anchor="middle")
    d.path([(265,185),(265,123),(80,123),(80,180)],color=AMBER)
    d.path([(635,185),(635,123),(820,123),(820,180)],color=AMBER)
    d.text(172, 98, "舍入到左端", 21, AMBER, 600, "middle")
    d.text(728, 98, "舍入到右端", 21, AMBER, 600, "middle")
    d.text(40, 330, "1 + 2⁻⁸ → 1", 24, NAVY, 600)
    d.text(40, 375, "(1 + 2⁻⁷) + 2⁻⁸ → 1 + 2⁻⁶", 24, NAVY, 600)
    d.note("位模式标注对应 BF16；图示覆盖 1 附近的正常数，两个中点不应都向下截断。")
    d.save("bf16-rounding.svg")


def main():
    for fn in [broadcast_axes, block_window, runtime_lifecycle, bf16_rounding]:
        fn()


if __name__ == "__main__":
    main()
