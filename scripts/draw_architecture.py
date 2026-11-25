#!/usr/bin/env python3
"""Draw data-layout and near-memory diagrams from the guide's worked examples."""

from draw_figures import Diagram, NAVY, TEAL, BLUE, AMBER, INK, MUTED, LINE, PALE, AQUA, GOLD


def system_overview():
    d = Diagram(805, "双核 Zeus 系统与每核 L2 Buffer、VP、DT 和 LDG")
    d.label(20, 30, "系统视图 · 两个计算核各自组织工作与近存数据", NAVY)
    d.box(20, 62, 860, 70, "Host · 应用、框架与运行时",
          "准备数据与参数 · 提交 kernel · 管理对象与同步", PALE, size=25)

    def bidirectional(x, y1, y2):
        d.parts.append(
            f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
            f'stroke="{TEAL}" stroke-width="2" '
            'marker-start="url(#arrow)" marker-end="url(#arrow)"/>'
        )

    bidirectional(450, 132, 177)
    d.box(20, 177, 860, 62, "GDG 存储分组与设备互连",
          fill=PALE, size=24)
    for x, name in [(20, "Core 0"), (470, "Core 1")]:
        bidirectional(x+205, 239, 283)
        d.rect(x, 283, 410, 469, "#ffffff", LINE, r=3)
        d.text(x+20, 320, name, 24, NAVY, 600)
        d.text(x+390, 320, "存储按核归属", 20, MUTED, anchor="end")
        d.box(x+20, 342, 208, 74, "LDG · 3D DRAM",
              "权重 / KV Cache", PALE, size=21)
        d.box(x+294, 342, 96, 74, "PE", "矩阵计算", AQUA, size=25)
        d.path([(x+228, 387), (x+294, 387)], color=TEAL)
        d.text(x+261, 373, "WL1", 18, TEAL, 600, "middle")
        d.box(x+20, 459, 370, 60, "L2 Buffer · 512 KiB",
              "本核 tile 与中间结果", PALE, size=23)
        d.path([(x+316, 459), (x+316, 416)], color=TEAL)
        d.path([(x+366, 416), (x+366, 459)], color=TEAL)
        for offset, unit, label in [
            (20, "VP", "向量与归约"),
            (214, "DT", "片上数据变换"),
        ]:
            gx = x+offset
            bidirectional(gx+88, 519, 561)
            d.rect(gx, 561, 176, 129, "#ffffff", LINE, r=3)
            d.text(gx+88, 584, label, 20, NAVY, 600, "middle")
            for i in range(4):
                bx = gx+11+(i % 2)*83
                by = 596+(i // 2)*46
                d.box(bx, by, 71, 36, f"{unit}{i}",
                      fill=AQUA, size=20)
        d.text(x+205, 728, "CT · LD/ST · LW", 22, NAVY, 600, "middle")
    d.note("逻辑连接示意；各核 L2 Buffer 与 LDG 按核归属，容量不按框图面积绘制。")
    d.save("system-overview.svg")


def near_memory():
    d = Diagram(825, "四 VP、四 DT、L2 Buffer 与 PE 的核内数据路径")
    d.label(20, 30, "核内视图 · 控制、向量、变换与矩阵通路", NAVY)
    d.text(20, 86, "PE 的大矩阵操作数常驻 LDG", 25, NAVY, 600)
    d.text(20, 126, "激活与中间结果使用本核 L2 Buffer", 22, MUTED)
    d.text(20, 165, "两条数据路径在 PE 汇合", 22, TEAL, 600)
    # The outlines identify stacked memory, not a package layer count.
    for dx, dy, fill in [(12, -12, "#dbe5ef"), (6, -6, "#e7eef5"), (0, 0, PALE)]:
        d.rect(560+dx, 84+dy, 300, 98, fill, LINE, r=2)
    d.text(710, 124, "LDG · 3D DRAM", 26, NAVY, 600, "middle")
    d.text(710, 161, "权重 / KV Cache", 22, INK, anchor="middle")
    d.rect(20, 212, 860, 531, "#ffffff", LINE, r=3)
    d.box(50, 238, 580, 53, "CT · 循环、地址与指令发射", fill=PALE, size=23)
    for start, name, description in [(50, "VP", "向量与归约"), (370, "DT", "片上数据变换")]:
        d.text(start, 328, description, 21, NAVY, 600)
        for i in range(4):
            d.box(start+i*67, 345, 59, 51, f"{name}{i}", fill=AQUA, size=20)
            port_x = start+i*67+29.5
            d.parts.append(
                f'<line x1="{port_x}" y1="396" x2="{port_x}" y2="450" '
                f'stroke="{TEAL}" stroke-width="2" marker-start="url(#arrow)" '
                'marker-end="url(#arrow)"/>'
            )
    d.box(50, 450, 580, 95, "L2 Buffer · 512 KiB",
          "输入 tile · 部分和 · 向量结果 · 布局暂存", PALE, size=26)
    d.box(720, 278, 140, 80, "WL1 × 8", "读取与分发", PALE, size=23)
    d.box(720, 450, 140, 95, "PE", "矩阵乘累加", AQUA, size=29)
    d.path([(790, 182), (790, 278)], color=TEAL)
    d.path([(790, 358), (790, 450)], color=TEAL)
    d.text(760, 407, "矩阵读取", 20, TEAL, 600, "end")
    d.path([(630, 476), (720, 476)], color=TEAL)
    d.text(674, 461, "AL1", 18, TEAL, 600, "middle")
    d.path([(720, 523), (630, 523)], color=TEAL)
    d.text(674, 552, "OL1", 18, TEAL, 600, "middle")
    d.box(50, 630, 140, 65, "LD", "原样读入", PALE)
    d.box(230, 630, 140, 65, "ST", "原样写出", PALE)
    d.box(430, 630, 200, 65, "LW → WA", "按布局写 LDG", GOLD, AMBER)
    d.path([(120, 630), (120, 545)])
    d.path([(300, 545), (300, 630)])
    d.path([(530, 545), (530, 630)], color=AMBER)
    d.text(50, 725, "LD/ST：经互连访问 GDG 或 LDG 原始字节", 19, MUTED)
    d.path([(630, 663), (892, 663), (892, 320), (860, 320)], color=AMBER)
    d.path([(850, 278), (850, 182)], color=AMBER)
    d.text(746, 643, "WL1 写接口", 19, AMBER, anchor="middle")
    d.text(20, 785, "VP 与 DT 各有四条物理通道；编译器按依赖、工作量与存储冲突安排并行。", 20, MUTED)
    d.note("箭头表示数据方向。WL1 提供矩阵读取与外部访问接口；示意图不按带宽比例绘制。")
    d.save("ldg-near-memory.svg")


def dense_blocked():
    d = Diagram(490, "同一矩阵在 Dense 与 Blocked 中的物理块顺序")
    d.label(20, 30, "W[256,256] · BF16 · Blocked 使用 BN=256、BK=128", NAVY)
    colors = [AQUA, "#dce9f3", GOLD, "#e8e5f1"]
    d.text(20, 80, "逻辑矩阵 [N,K]", 23, NAVY, 600)
    for row in range(2):
        for col in range(2):
            i = row*2+col
            d.rect(20+col*140, 108+row*110, 140, 110, colors[i], "#ffffff", r=0)
            d.text(90+col*140, 151+row*110, "ABCD"[i], 29, NAVY, 600, "middle")
            d.text(90+col*140, 188+row*110, "128 × 128", 19, MUTED, anchor="middle")
    for y, title, order in [(108, "Dense：全矩阵统一 pack", [0,1,2,3]),
                            (265, "Blocked：左半块、右半块各自 pack", [0,2,1,3])]:
        d.text(350, y-22, title, 22, NAVY, 600)
        for j,i in enumerate(order):
            d.rect(350+j*132, y, 132, 68, colors[i], "#ffffff", r=0)
            d.text(416+j*132, y+44, "ABCD"[i], 27, NAVY, 600, "middle")
            d.text(350+j*132, y+99, f"{j*32} KiB", 18, MUTED)
    d.rect(20, 388, 860, 63, PALE, LINE, r=2)
    d.text(38, 414, "W[0,128]：Dense +32 KiB，Blocked +64 KiB", 21, NAVY, 600)
    d.text(38, 442, "W[128,0]：Dense +64 KiB，Blocked +32 KiB", 21, NAVY, 600)
    d.note("每个字母代表 32 KiB 的已 pack 子区；外层块顺序变化，块内仍采用相同 normal 规则。")
    d.save("dense-blocked.svg")


def ldg_layout():
    d = Diagram(610, "BF16 权重逻辑坐标到 LDG 七维布局")
    d.label(20, 30, "B[N=128, K=128] · BF16 · 32 KiB", NAVY)
    x, y, cw, ch = 55, 102, 132, 35
    for n in range(8):
        for k in range(2):
            d.rect(x+k*cw, y+n*ch, cw, ch, AQUA if k == 0 else "#e5ebf5", r=0)
            d.text(x+k*cw+66, y+n*ch+24, f"块 ({n}, {k})", 17, anchor="middle")
    d.text(x+66, 86, "K: 0–63", 18, MUTED, anchor="middle")
    d.text(x+198, 86, "K: 64–127", 18, MUTED, anchor="middle")
    d.text(55, 415, "每个小块：16 × 64 × 2 B = 2 KiB", 19)
    d.path([(340, 240), (480, 240)])
    d.text(410, 210, "pack", 21, NAVY, 600, "middle")
    d.text(410, 276, "按物理次序", 18, MUTED, anchor="middle")
    for k in range(2):
        for n in range(8):
            xx=505+n*44
            yy=110+k*150
            d.rect(xx, yy, 44, 80, AQUA if k == 0 else "#e5ebf5", r=0)
            d.text(xx+22, yy+44, str(n), 20, anchor="middle")
        d.text(505, yy-14, f"K 组 {k} · 8 个 N 小块依次排列", 19, NAVY, 600)
        d.text(505, yy+110, f"字节区间 [{k*16384}, {(k+1)*16384})", 19, MUTED)
    d.rect(20, 460, 860, 116, "#f6f7f9", r=2)
    d.text(38, 490, "物理 shape  [1, 1, 2, 8, 1, 16, 64]", 22, NAVY, 600)
    d.text(38, 525, "B[0,64]：逻辑偏移 128 B → 物理偏移 16,384 B", 20)
    d.text(38, 557, "B[16,0]：逻辑偏移 4,096 B → 物理偏移 2,048 B", 20)
    d.note("此例固定 normal 布局、blocksize=128、k1=8、c1=1、k0=16、c0=64。")
    d.save("ldg-layout.svg")


def kv_update():
    d = Diagram(560, "专用 KV 格式的单 token 写入")
    d.label(20, 30, "PAGE_SIZE=1024 · D=128 · token_offset=1000 · BF16", NAVY)
    d.box(20, 65, 290, 78, "新 token [1,128]", "256 B · 位于 L2 Buffer", AQUA)
    d.box(445, 65, 435, 78, "LDG payload 页", "以宿主解析的页地址为基址", PALE)
    d.path([(310, 104), (445, 104)])
    d.text(377, 88, "LW", 19, MUTED, anchor="middle")
    for y, title in [(195, "K 页 · 256 KiB"), (354, "V 页 · 256 KiB")]:
        d.text(20, y, title, 23, NAVY, 600)
        d.rect(20, y+20, 860, 65, "#f2f4f6", r=1)
    for xx in [720, 820]:
        d.rect(xx, 218, 10, 59, TEAL, TEAL, r=0)
    d.text(20, 312, "K：两段 × 128 B，起点 +242688、+259072", 20)
    for j in range(8):
        d.rect(642+j*29, 377, 6, 59, BLUE, BLUE, r=0)
    d.text(20, 471, "V：八段 × 32 B，起点 +247040 + j × 2048", 20)
    d.text(20, 507, "只修改当前 token 对应的 256 B；其他 token 保持原值。", 21, NAVY)
    d.note("位置示意未按字节比例绘制。PE 读取使用的专用索引表与 payload 是不同对象。")
    d.save("kv-cache-update.svg")


def attention_slots():
    d = Diagram(645, "普通 weight 布局中的 K 行写入与 V 列写入")
    d.label(20, 30, "slot=389 · D=256 · R=1024 · 物理 block=128×64 BF16", NAVY)
    d.text(20, 75, "K[R,D]：一行拆成 4 段", 22, NAVY, 600)
    for j in range(4):
        xx=25+j*218
        d.rect(xx, 105, 194, 145, "#edf2f6", r=1)
        d.rect(xx, 134, 194, 12, AQUA, TEAL, r=0)
        d.text(xx+97, 194, f"block {12+j}", 22, NAVY, 600, "middle")
        d.text(xx+97, 224, "128 行 × 64 列", 17, MUTED, anchor="middle")
    d.text(25, 286, "row_block=3，row_in_block=5；每段写 64 个 BF16 元素。", 20)
    d.text(20, 344, "V[D,R]：一列拆成 2 段", 22, NAVY, 600)
    for j, block in enumerate([6,22]):
        xx=25+j*436
        d.rect(xx, 372, 400, 145, "#edf2f6", r=1)
        d.rect(xx+31, 372, 12, 145, AQUA, TEAL, r=0)
        d.text(xx+226, 434, f"block {block}", 23, NAVY, 600, "middle")
        d.text(xx+226, 470, "128 行 × 64 列", 19, MUTED, anchor="middle")
    d.text(25, 555, "col_block=6，col_in_block=5；每段写 128 个 BF16 元素。", 20)
    d.note("普通 weight 路径：K 页内沿用 normal；V 不同于专用 vcache。分页寻址另有契约。")
    d.save("attention-slots.svg")


def paged_attention():
    d = Diagram(590, "页表从逻辑 token 到物理槽位的映射")
    d.label(20, 30, "block_table=[3,1] · page_size=256 · seq_len=390", NAVY)
    d.text(20, 80, "逻辑序列", 22, NAVY, 600)
    d.rect(20, 104, 430, 60, AQUA, r=1)
    d.rect(450, 104, 225, 60, GOLD, r=1)
    d.rect(675, 104, 205, 60, "#f1f1f1", r=1)
    d.text(235, 141, "逻辑页 0：token 0–255", 21, anchor="middle")
    d.text(562, 141, "256–389", 21, anchor="middle")
    d.text(777, 141, "不参与计算", 19, MUTED, anchor="middle")
    d.text(20, 210, "物理存储", 22, NAVY, 600)
    for i in range(4):
        xx=20+i*220
        d.rect(xx, 243, 200, 84, AQUA if i==3 else GOLD if i==1 else PALE, r=1)
        d.text(xx+100, 276, f"物理页 {i}", 21, NAVY, 600, "middle")
        d.text(xx+100, 309, f"{i*256}–{i*256+255}", 19, anchor="middle")
    d.path([(235,164),(235,186),(780,186),(780,243)], color=TEAL)
    d.path([(562,164),(562,224),(340,224),(340,243)], color=AMBER)
    d.rect(20, 375, 860, 155, "#f6f7f9", r=2)
    d.text(40, 408, "physical_slot = block_table[t // 256] × 256 + t % 256", 21, NAVY, 600)
    for xx, label in [(40, "t=0 → 768"), (300, "t=255 → 1023"),
                      (600, "t=256 → 256")]:
        d.text(xx, 448, label, 21)
    d.text(40, 486, "t=389 → 389；第二个逻辑页只有 134 个有效 token。", 21)
    d.note("逻辑页次序决定 Attention 的序列语义；物理页号不等于 token 的先后次序。")
    d.save("paged-attention.svg")


def online_softmax():
    d = Diagram(625, "在线 Softmax 的状态缩放与更新")
    d.label(20, 30, "逐块扫描 K/V：只保留每个 query 的 m、l、o", NAVY)
    d.box(20, 75, 258, 82, "旧状态", "m_old · l_old · o_old", PALE)
    d.box(346, 75, 534, 82, "当前块分数 s = QKᵀ × scale", "先应用长度与因果 mask", AQUA)
    d.path([(613,157),(613,198)])
    d.box(346, 198, 534, 70, "m_new = max(m_old, max(s))", fill=AQUA, size=23)
    d.path([(278,116),(311,116),(311,232),(346,232)])
    d.path([(613,268),(613,317)])
    d.box(346, 317, 534, 70, "p = exp(s − m_new)", fill=AQUA, size=23)
    d.box(20, 317, 258, 70, "α = exp(m_old − m_new)", fill=GOLD, size=19)
    d.path([(149,157),(149,317)])
    d.path([(410,268),(410,292),(230,292),(230,317)])
    d.path([(149,387),(149,437)])
    d.path([(613,387),(613,437)])
    d.rect(20, 437, 860, 108, "#f2f5f8", r=2)
    d.text(450, 477, "l_new = α × l_old + sum(p)", 24, NAVY, 600, "middle")
    d.text(450, 515, "o_new = α × o_old + pV", 24, NAVY, 600, "middle")
    d.text(450, 590, "扫描结束：O = o / l", 24, NAVY, 600, "middle")
    d.note("m 的尺度变化必须同时用于旧 l 和旧 o；只更新分母会改变结果。")
    d.save("online-softmax.svg")


def matmul_reuse():
    d = Diagram(540, "跨 N tile 复用激活的逻辑存储预算")
    d.label(20, 30, "BM=BN=BK=128 · BF16 输入 · FP32 累加", NAVY)
    d.box(20, 166, 215, 115, "A tile", "128 × 128 · 32 KiB", AQUA)
    for yy, suffix in [(75,"0"), (270,"1")]:
        d.box(310, yy, 200, 85, f"W tile {suffix}", "128 × 128 · 32 KiB", PALE)
        d.rect(562, yy+22, 52, 52, AQUA, TEAL, r=26)
        d.text(588, yy+56, "×", 27, TEAL, 600, "middle")
        d.box(680, yy, 200, 100, f"acc {suffix}", "128 × 128 · 64 KiB", GOLD, AMBER)
        d.path([(510,yy+48),(562,yy+48)])
        d.path([(614,yy+48),(680,yy+48)])
    d.path([(235,201),(270,201),(270,58),(588,58),(588,97)])
    d.path([(235,249),(270,249),(270,382),(588,382),(588,344)])
    d.text(20, 321, "同一份 A", 22, TEAL, 600)
    d.text(20, 355, "服务两个 N 块", 20, MUTED)
    d.rect(20, 404, 860, 89, "#f5f6f8", r=2)
    d.text(40, 439, "片上 A + 两份 acc：32 + 128 = 160 KiB", 22, NAVY, 600)
    d.text(40, 474, "A 双缓冲后为 192 KiB；LDG 侧两份 W 共 64 KiB", 21)
    d.note("仅列主要活跃量；W 从 LDG 读取，布局临时量与 L2 Buffer 实际分配另行核对。")
    d.save("matmul-reuse.svg")


def int4_packing():
    d = Diagram(485, "INT4 输出通道配对与半字节解释")
    d.label(20, 30, "逻辑 W[256,K]：同一列 k，前后两组 128 行配对", NAVY)
    d.box(20, 80, 380, 96, "W[145,7] = 5", "高四位：0101（5）", PALE)
    d.box(500, 80, 380, 96, "W[17,7] = −3", "低四位：1101（D）", AQUA)
    d.path([(210,176),(210,215),(310,215),(310,254)],color=BLUE)
    d.path([(690,176),(690,215),(590,215),(590,254)],color=TEAL)
    for i, value in enumerate("01011101"):
        d.rect(170+i*70, 254, 70, 72, PALE if i<4 else AQUA, "#ffffff", r=0)
        d.text(205+i*70, 301, value, 29, NAVY, 600, "middle")
    d.text(310, 363, "高四位：通道 145", 22, BLUE, 600, "middle")
    d.text(590, 363, "低四位：通道 17", 22, TEAL, 600, "middle")
    d.text(450, 418, "合成字节 0x5D；随后仍需按 7D 规则重排", 24, NAVY, 600, "middle")
    d.note("不是相邻 K 元素配对。读取低四位 D 时，必须保留它表示有符号 −3 的语义。")
    d.save("int4-packing.svg")


def uscale_allocation():
    d = Diagram(525, "INT4 和 UScale 的有效字节与分配空间")
    d.label(20, 30, "N=256 · K=128 · G=128 · 一个 64 KiB 对齐槽位", NAVY)
    for x,w,fill,title,sub in [(20,215,AQUA,"INT4 编码","16 KiB"),
                               (235,215,"#e5edf6","UScale 页","16 KiB"),
                               (450,430,"#f0f0f0","槽位补齐","32 KiB")]:
        d.box(x, 100, w, 96, title, sub, fill)
    for x,t in [(20,"0"),(235,"16 KiB"),(450,"32 KiB"),(880,"64 KiB")]:
        d.text(x,82,t,18,MUTED,anchor="start" if x==20 else "end" if x==880 else "middle")
    d.text(20, 254, "展开 UScale 页：每路 2 KiB，只有前 64 B 是本例的有效系数", 22, NAVY, 600)
    for lane in range(8):
        x=20+lane*108
        d.rect(x,290,99,112,"#edf1f5",LINE,r=1)
        d.rect(x,290,4,112,BLUE,BLUE,r=0)
        d.text(x+50,337,f"lane {lane}",19,NAVY,600,"middle")
        d.text(x+50,374,"32 × BF16",16,MUTED,anchor="middle")
    d.text(20, 449, "有效 scale：8 × 64 B = 512 B；分配页：8 × 2 KiB = 16 KiB", 22, NAVY)
    d.note("蓝色细条表示有效系数。分配补齐不直接等于执行时的物理读取量。")
    d.save("uscale-allocation.svg")


def vp_pipeline():
    d = Diagram(676, "单个 VP 的两组 32 lane：宽模式与级联模式")
    d.label(20, 30, "单 VP 放大 · 相同硬件的两种数据路径", NAVY)
    for x, title, subtitle in [
        (20, "宽模式 · 两组并行", "不同元素使用相同运算配置"),
        (460, "级联模式 · 前后接续", "同一批元素经过两段运算"),
    ]:
        d.rect(x, 66, 420, 530, "#ffffff", LINE, r=4)
        d.text(x+210, 105, title, 25, NAVY, 600, "middle")
        d.text(x+210, 135, subtitle, 20, MUTED, anchor="middle")

    d.box(48, 157, 364, 70, "x[0] … x[63]",
          "64 个 BF16 元素", PALE, size=24)
    d.box(488, 157, 364, 70, "x[0] … x[31]",
          "32 个 BF16 元素", PALE, size=24)

    # Wide mode sends separate element ranges through equal configurations.
    d.path([(230, 227), (230, 251)], arrow=False)
    for cx, first, last, name, fill, color in [
        (130, 0, 31, "A", AQUA, TEAL),
        (330, 32, 63, "B", PALE, BLUE),
    ]:
        d.path([(230, 251), (cx, 251), (cx, 296)], color=color)
        d.rect(cx-82, 296, 164, 122, fill, LINE)
        d.text(cx, 328, f"x[{first}] … x[{last}]", 20,
               color, anchor="middle")
        d.text(cx, 363, f"组 {name} · 32 lane", 21,
               NAVY, 600, "middle")
        d.text(cx, 398, "f(x)", 24, INK, anchor="middle")
        d.path([(cx, 418), (cx, 464), (230, 464)],
               color=color, arrow=False)
    d.path([(230, 464), (230, 504)])
    d.box(48, 504, 364, 65, "y[0] … y[63]",
          "y = f(x) · 64 个元素", AQUA, size=24)

    # Cascaded mode preserves element identity across the two groups.
    d.path([(670, 227), (670, 257)], color=TEAL)
    d.box(562, 257, 216, 86, "组 A · 32 lane",
          "u = f(x)", AQUA, TEAL, size=23)
    d.path([(670, 343), (670, 385)], color=BLUE)
    d.text(692, 371, "u[0] … u[31]", 19, BLUE)
    d.box(562, 385, 216, 86, "组 B · 32 lane",
          "y = g(u)", PALE, BLUE, size=23)
    d.path([(670, 471), (670, 504)], color=BLUE)
    d.box(488, 504, 364, 65, "y[0] … y[31]",
          "y = g(f(x)) · 32 个元素", AQUA, size=24)

    d.text(20, 632, "每核有 4 个独立 VP；本图只展开其中 1 个。",
           22, NAVY, 600)
    d.note("以可用的 BF16 单输入路径为例；箭头表示数据依赖，不表示一拍完成。")
    d.save("vp-pipelines.svg")


def main():
    for fn in [system_overview, near_memory, vp_pipeline, dense_blocked, ldg_layout, kv_update, attention_slots,
               paged_attention, online_softmax, matmul_reuse,
               int4_packing, uscale_allocation]:
        fn()


if __name__ == "__main__":
    main()
