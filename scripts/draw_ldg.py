#!/usr/bin/env python3
"""Draw the guide's worked LDG layouts without editing shared diagrams."""

from draw_figures import (
    Diagram, NAVY, TEAL, AMBER, INK, MUTED, LINE, PALE, AQUA, GOLD,
)


COLORS = [AQUA, "#dfeaf5", GOLD, "#ece5f4"]


def miniblocks():
    """Keep the mathematical and storage views visible at the same time."""
    d = Diagram(590, "B[128,256] 与 W[256,128] 的四个 BF16 miniblock")
    d.label(20, 30, "BF16 · B[K,N] 与 W[N,K] 保存同一组元素", NAVY)
    for x, title, order, height in [
        (20, "数学视图 B[128,256]", [0, 2, 1, 3], 85),
        (530, "存储视图 W[256,128]", [0, 1, 2, 3], 110),
    ]:
        d.text(x, 80, title, 24, NAVY, 600)
        for row in range(2):
            for col in range(2):
                block = order[row * 2 + col]
                bx, by = x + col * 165, 128 + row * height
                d.rect(bx, by, 165, height, COLORS[block], "#ffffff", 0)
                d.text(bx + 82, by + height / 2 + 7,
                       f"miniblock {block}", 21, NAVY, 600, "middle")
        axis = ["N: 0–127", "N: 128–255"] if x == 20 else [
            "K: 0–63", "K: 64–127"]
        for col, text in enumerate(axis):
            d.text(x + 82 + col * 165, 112, text, 20,
                   MUTED, anchor="middle")
        row_axis = "每行组 64 个 K" if x == 20 else "每行组 128 个 N"
        d.text(x, 128 + 2 * height + 32, row_axis, 20, MUTED)
    d.path([(365, 224), (495, 224)], color=TEAL)
    d.text(430, 199, "W = Bᵀ", 23, TEAL, 600, "middle")
    d.text(430, 261, "坐标视图", 20, MUTED, anchor="middle")
    d.text(20, 421, "物理字节流：先遍历同一 N 分组的 K 块，再切换 N 分组", 23,
           NAVY, 600)
    for block in range(4):
        x = 20 + block * 215
        d.rect(x, 449, 215, 68, COLORS[block], "#ffffff", 0)
        d.text(x + 107, 491, f"miniblock {block}", 23, NAVY, 600, "middle")
        d.text(x, 547, str(block * 16384), 20, MUTED)
    d.text(880, 547, "65536 B", 20, MUTED, anchor="end")
    d.note("一个 miniblock：B 的 64×128 元素 = W 的 128×64 元素 = 16 KiB。")
    d.save("ldg-v005-miniblocks.svg")


def seven_dim():
    """Expose all axis roles and one complete byte-offset decomposition."""
    d = Diagram(650, "七维 reshape 与 permute，以及 B[70,145] 的字节位置")
    d.label(20, 30, "W[256,128] · normal BF16 · blocksize = 128", NAVY)
    names = ["n_b", "n_g", "n_l", "n_i", "k_g", "k_l", "k_i"]
    sizes = [2, 1, 8, 16, 2, 1, 64]
    colors = [PALE, PALE, AQUA, AQUA, GOLD, GOLD, GOLD]
    for y, label, order in [
        (98, "reshape：按 N、K 的原逻辑次序拆开坐标", list(range(7))),
        (244, "permute (0,1,4,2,5,3,6)：形成连续 miniblock", [0, 1, 4, 2, 5, 3, 6]),
    ]:
        d.text(20, y - 20, label, 23, NAVY, 600)
        for i, axis in enumerate(order):
            x = 20 + i * 123
            d.rect(x, y, 121, 83, colors[axis], "#ffffff", 0)
            d.text(x + 60, y + 31, names[axis], 23, NAVY, 600, "middle")
            d.text(x + 60, y + 66, sizes[axis], 25, INK, anchor="middle")
    d.text(20, 372, "miniblock 3：八个 2 KiB 片段；选中片段 1", 23, NAVY, 600)
    for lane in range(8):
        x = 20 + lane * 107.5
        d.rect(x, 392, 107.5, 62, GOLD if lane == 1 else PALE,
               "#ffffff", 0)
        d.text(x + 54, 431, f"片段 {lane}", 22,
               AMBER if lane == 1 else INK, 600, "middle")
    d.text(20, 491, "B[70,145] = W[145,70]", 25, NAVY, 600)
    d.text(20, 529, "块内行索引 17 → 片段 1、片内行 1；列索引 6", 23, MUTED)
    d.rect(20, 550, 860, 42, AQUA, LINE, 2)
    d.text(450, 579, "3 × 16384 + 1 × 2048 + 1 × 128 + 6 × 2 = 51340 B",
           23, TEAL, 600, "middle")
    d.note("n_l 表示 N 片段轴，k_l 在本例中大小为 1；最右侧轴在物理字节流中连续。")
    d.save("ldg-v005-seven-dim.svg")


def blocked():
    """Show the same four regions in mathematical and physical order."""
    d = Diagram(580, "B[256,256] 的 Dense 与 BN256 BK128 Blocked 比较")
    d.label(20, 30, "数学 B[K=256,N=256] · BF16 · 总量 128 KiB", NAVY)
    d.text(20, 83, "B 的逻辑分区", 24, NAVY, 600)
    for row, names in enumerate(["AC", "BD"]):
        for col, name in enumerate(names):
            idx = "ABCD".index(name)
            x, y = 20 + col * 145, 121 + row * 105
            d.rect(x, y, 145, 105, COLORS[idx], "#ffffff", 0)
            d.text(x + 72, y + 47, name, 29, NAVY, 600, "middle")
            d.text(x + 72, y + 82, "128 × 128", 20, MUTED, anchor="middle")
    d.text(92, 112, "N 0–127", 20, MUTED, anchor="middle")
    d.text(237, 112, "N 128–255", 20, MUTED, anchor="middle")
    d.text(20, 364, "上半：K 0–127", 21, MUTED)
    d.text(20, 398, "下半：K 128–255", 21, MUTED)
    for y, title, order in [
        (121, "Dense：完整 W[256,256] 一次 pack", "ABCD"),
        (293, "Blocked：两份 W[256,128] 各自 pack", "ACBD"),
    ]:
        d.text(350, y - 24, title, 22, NAVY, 600)
        for i, name in enumerate(order):
            x = 350 + i * 132
            d.rect(x, y, 132, 65, COLORS["ABCD".index(name)],
                   "#ffffff", 0)
            d.text(x + 66, y + 43, name, 28, NAVY, 600, "middle")
            d.text(x, y + 96, f"{i * 32} KiB", 20, MUTED)
        if order == "ACBD":
            d.path([(350, 410), (350, 422), (611, 422), (611, 410)],
                   arrow=False)
            d.path([(614, 410), (614, 422), (878, 422), (878, 410)],
                   arrow=False)
            d.text(480, 451, "独立块 0 · 64 KiB", 20, MUTED, anchor="middle")
            d.text(746, 451, "独立块 1 · 64 KiB", 20, MUTED, anchor="middle")
    d.rect(20, 476, 860, 62, PALE, LINE, 2)
    d.text(38, 502, "W[0,128]：Dense +32768 B；Blocked +65536 B", 22, NAVY, 600)
    d.text(38, 530, "每个字母代表 32 KiB 子区，含两个 16 KiB miniblock。", 21, MUTED)
    d.note("Blocked 对应本例 Tr=2、Tc=2、aligned_size=0；仅更改计算 tile 不会改变存储。")
    d.save("ldg-v005-blocked.svg")


def uscale():
    """Show scale-page capacity separately from the useful coefficient bytes."""
    d = Diagram(600, "N256 K128 G32 的 BF16 UScale 页与坐标 S[145,2]")
    d.label(20, 30, "S[N=256,q=4] · G=32 · 一个 16 KiB UScale 页", NAVY)
    d.text(20, 79, "每 lane：8 个有效 slot，共 256 B；余下 1792 B 补齐", 23, NAVY, 600)
    for lane in range(8):
        x = 20 + lane * 107.5
        d.text(x + 53, 113, f"lane {lane}", 21, MUTED, anchor="middle")
        d.rect(x, 129, 107.5, 192, PALE, "#ffffff", 0)
        d.rect(x, 129, 107.5, 24, GOLD if lane == 1 else AQUA,
               "#ffffff", 0)
        d.text(x + 53, 347, "2 KiB", 21, MUTED, anchor="middle")
        if lane == 1:
            d.rect(x + 2, 131, 103.5, 188, "none", AMBER, 0)
    d.text(20, 392, "放大 lane 1 的 256 B 有效区域（每格 32 B）", 23, NAVY, 600)
    for slot in range(8):
        x = 20 + slot * 107.5
        d.rect(x, 412, 107.5, 65, GOLD if slot == 6 else AQUA,
               "#ffffff", 0)
        d.text(x + 53, 439, f"slot {slot}", 22, NAVY, 600, "middle")
        d.text(x + 53, 465, f"g={slot % 4}", 20, MUTED, anchor="middle")
    d.text(20, 507, "slot 0–3：N 16–31", 21, MUTED)
    d.text(450, 507, "slot 4–7：N 144–159", 21, MUTED)
    d.text(450, 544, "S[145,2] → lane 1、slot 6、行 1 → 2048 + 192 + 2 = 2242 B",
           22, TEAL, 600, "middle")
    d.note("上图按容量比例标出有效区；下图单独放大。权重 miniblock 与 scale 页的排列不同。")
    d.save("ldg-v005-uscale.svg")


def index_pages():
    """Resolve a concrete logical token through a permuted payload-page map."""
    d = Diagram(630, "专用 K 页索引：token 1025 到 payload 字节地址 32896")
    d.label(20, 30, "BF16 K · 每页 1024 token × 128 dim = 256 KiB", NAVY)
    d.text(20, 81, "逻辑页 → 表项编码", 24, NAVY, 600)
    d.box(20, 111, 320, 72, "页 0 → 18", "token 0–1023", PALE, size=24)
    d.box(20, 208, 320, 72, "页 1 → 2", "token 1024–2047", AQUA, size=24)
    d.text(20, 321, "每个表项复制到 8 路", 21, MUTED)
    d.text(20, 355, "编码单位：16 KiB", 21, MUTED)
    d.text(480, 81, "本核 LDG 相对字节地址", 24, NAVY, 600)
    for y, height, title, sub, fill in [
        (111, 65, "索引表 · 32 KiB", "[0,32768)", PALE),
        (197, 92, "逻辑页 1 payload", "[32768,294912)", AQUA),
        (310, 92, "逻辑页 0 payload", "[294912,557056)", GOLD),
    ]:
        d.box(480, y, 400, height, title, sub, fill, size=23)
    d.path([(340, 244), (480, 244)], color=TEAL)
    d.text(410, 223, "2 × 16384", 20, TEAL, 600, "middle")
    d.text(20, 454, "读取 token 1025、dim 0 的地址链", 24, NAVY, 600)
    steps = [("1025 // 1024", "逻辑页 1"), ("index[1] = 2", "payload 编码"),
             ("2 × 16384", "页首 32768 B"), ("页内 token 1", "偏移 128 B")]
    for i, (title, sub) in enumerate(steps):
        x = 20 + i * 220
        d.box(x, 476, 200, 66, title, sub, AQUA if i == 1 else PALE,
              size=21)
        if i < 3:
            d.path([(x + 200, 509), (x + 220, 509)])
    d.text(450, 577, "最终相对地址：32768 + 128 = 32896 B", 24, TEAL, 600, "middle")
    d.note("这里的表项定位 K payload；USaddr 定位 scale 第一页，两者含义不同。")
    d.save("ldg-v005-index-pages.svg")


def layout_overview():
    """Separate resource usage from layouts and quantization metadata."""
    d = Diagram(640, "LDG 用途：Weight 布局与分页 KV")
    d.label(20, 31, "先区分用途，再选择布局与访问接口", NAVY)
    d.box(310, 55, 280, 64, "LDG 数据", fill=PALE, size=26)
    for center in (230, 670):
        d.path([(450, 119), (450, 145), (center, 145),
                (center, 175)])

    d.rect(20, 175, 420, 425, "#FFFFFF", LINE, 12)
    d.rect(460, 175, 420, 425, "#FFFFFF", LINE, 12)
    d.text(40, 212, "Weight · 矩阵权重", 25, NAVY, 600)
    d.text(40, 245, "存储视图 W[N,K]", 21, MUTED)
    for y, title, subtitle, fill in [
        (273, "Dense", "对完整矩阵连续 pack", AQUA),
        (348, "Blocked · 紧排", "独立块依次紧密存放", PALE),
        (423, "Blocked · 带 padding", "固定槽尾保留未用空间", GOLD),
    ]:
        d.rect(40, y, 380, 64, fill, LINE, 7)
        d.text(56, y + 27, title, 22, NAVY, 600)
        d.text(56, y + 53, subtitle, 19, MUTED)
    d.text(40, 535, "分组量化可配套 UScale", 22, TEAL, 600)
    d.text(40, 565, "UScale 是量化参数，不是另一种布局", 19, MUTED)

    d.text(480, 212, "分页 KV · 键 / 值缓存", 25, NAVY, 600)
    d.text(480, 245, "页间按表项定位，物理页可离散", 21, MUTED)
    for x, title in [(490, "逻辑页 0"), (720, "逻辑页 1")]:
        d.box(x, 279, 140, 52, title, fill=PALE, size=21)
    d.text(675, 363, "物理页（示意）", 19, MUTED, anchor="middle")
    for x, title in [(490, "页 0"), (720, "页 1")]:
        d.box(x, 383, 140, 52, title, fill=AQUA, size=22)
        d.path([(x + 70, 331), (x + 70, 383)])
    d.text(675, 415, "···", 29, MUTED, 600, "middle")
    d.text(480, 472, "页内使用对应的 K / V 数据布局", 21, NAVY, 600)
    d.box(490, 491, 168, 52, "K 布局", fill=AQUA, size=22)
    d.box(692, 491, 168, 52, "V 布局", fill=GOLD, size=22)
    d.text(480, 579, "专用 KV 接口不支持 UScale", 21, TEAL, 600)
    d.note("分页描述页间地址关系；每个页内仍有明确的数据布局。")
    d.save("ldg-layout-overview.svg")


def miniblock_wl1():
    """Show fixed byte granularity and shared access, not scheduling."""
    d = Diagram(560, "16 KiB miniblock 的八个 2 KiB 片段与 WL1 归属")
    d.label(20, 32, "固定字节粒度：一个 miniblock = 16 KiB", NAVY)
    d.text(20, 68, "字节量固定；元素形状取决于数据类型", 22, MUTED)
    d.rect(20, 96, 860, 64, AQUA, LINE, 8)
    d.text(450, 137,
           "BF16 示例：W[128,64] = 8192 个元素 × 2 B = 16 KiB",
           24, NAVY, 600, "middle")
    d.text(450, 202, "一个 miniblock · 8 个等长片段", 23,
           NAVY, 600, "middle")
    d.path([(20, 224), (20, 214), (880, 214), (880, 224)],
           color=TEAL, arrow=False)

    d.rect(20, 350, 860, 92, PALE, LINE, 8)
    width = 860 / 8
    for lane in range(8):
        x = 20 + lane * width
        d.rect(x, 232, width, 76, AQUA, "#FFFFFF", 0)
        d.text(x + width / 2, 263, f"片段 {lane}", 22,
               NAVY, 600, "middle")
        d.text(x + width / 2, 293, "2 KiB", 21,
               MUTED, anchor="middle")
        d.box(x + 7, 371, width - 14, 50, f"WL1 {lane}",
              fill="#FFFFFF", size=21)
        d.path([(x + width / 2, 308), (x + width / 2, 371)])
    d.path([(20, 457), (20, 467), (880, 467), (880, 457)],
           color=TEAL, arrow=False)
    d.text(450, 506, "8 路 WL1 共同完成对一个 miniblock 的访问", 25,
           TEAL, 600, "middle")
    d.note("箭头表示片段归属，不表示 8 个独立可调度操作，也不承诺单拍完成。")
    d.save("ldg-miniblock-wl1.svg")


def blocked_padding():
    """Compare full packing, independent blocks, and 96 KiB slots."""
    d = Diagram(744, "W[256,256] 的 Dense、紧排 Blocked 和带 padding Blocked")
    d.label(20, 32, "BF16 · W[256,256] · 有效数据 128 KiB", NAVY)
    for col, text in enumerate(("K 0–127", "K 128–255")):
        d.text(168 + col * 128, 82, text, 20,
               MUTED, anchor="middle")
    d.text(86, 90, "N ↓", 20, MUTED, anchor="end")
    for row, names in enumerate(("AB", "CD")):
        d.text(86, 137 + row * 70,
               "0–127" if row == 0 else "128–255",
               20, MUTED, anchor="end")
        for col, name in enumerate(names):
            x, y = 104 + col * 128, 96 + row * 70
            d.rect(x, y, 128, 70, COLORS["ABCD".index(name)],
                   "#FFFFFF", 0)
            d.text(x + 64, y + 31, name, 27, NAVY, 600, "middle")
            d.text(x + 64, y + 58, "32 KiB", 19,
                   MUTED, anchor="middle")
    d.path([(373, 166), (414, 166)])
    for index, names in enumerate(("AC", "BD")):
        x = 430 + index * 236
        d.rect(x, 96, 214, 140, "#FFFFFF", LINE, 6)
        d.text(x + 107, 126, f"独立块 {index}", 22,
               NAVY, 600, "middle")
        d.text(x + 107, 155, "W[256,128]", 20,
               MUTED, anchor="middle")
        for col, name in enumerate(names):
            d.rect(x + col * 107, 170, 107, 66,
                   COLORS["ABCD".index(name)], "#FFFFFF", 0)
            d.text(x + 53.5 + col * 107, 212, name, 28,
                   NAVY, 600, "middle")
        d.text(x + 107, 264, "64 KiB", 21, MUTED, anchor="middle")

    origin, scale = 208, 3.5

    def payload(y, offset, names):
        for index, name in enumerate(names):
            x = origin + (offset + index * 32) * scale
            d.rect(x, y, 32 * scale, 62,
                   COLORS["ABCD".index(name)], "#FFFFFF", 0)
            d.text(x + 16 * scale, y + 41, name, 28,
                   NAVY, 600, "middle")

    def ticks(y, values):
        for value in values:
            x = origin + value * scale
            d.path([(x, y), (x, y + 7)], arrow=False)
            anchor = "end" if value == 192 else "middle"
            d.text(x, y + 30, str(value), 20,
                   MUTED, anchor=anchor)

    for y, title, subtitle in [
        (310, "Dense", "整矩阵 pack"),
        (430, "Blocked", "紧排"),
        (550, "Blocked", "带 padding"),
    ]:
        d.text(20, y + 30, title, 25, NAVY, 600)
        d.text(20, y + 60, subtitle, 21, MUTED)
    payload(310, 0, "ABCD")
    ticks(372, (0, 64, 128))
    d.text(684, 349, "总计 128 KiB", 23, NAVY, 600)
    d.text(684, 379, "刻度单位：KiB", 19, MUTED)

    d.text(20, 520, "aligned_size = 0", 18, MUTED)
    payload(430, 0, "ACBD")
    for offset in (0, 64):
        d.rect(origin + offset * scale, 430, 64 * scale, 62,
               "none", TEAL, 0)
    ticks(492, (0, 64, 128))
    d.text(684, 469, "总计 128 KiB", 23, NAVY, 600)

    d.text(20, 640, "aligned_size = 96", 18, MUTED)
    payload(550, 0, "AC")
    payload(550, 96, "BD")
    for offset in (64, 160):
        x = origin + offset * scale
        d.rect(x, 550, 32 * scale, 62, "#E9EDF0", "#FFFFFF", 0)
        d.text(x + 16 * scale, 575, "间隔", 20,
               MUTED, 600, "middle")
        d.text(x + 16 * scale, 601, "32 KiB", 19,
               MUTED, anchor="middle")
    for offset in (0, 96):
        x = origin + offset * scale
        d.rect(x, 550, 96 * scale, 62, "none", TEAL, 0)
        d.path([(x, 654), (x, 664), (x + 96 * scale, 664),
                (x + 96 * scale, 654)], color=TEAL, arrow=False)
        d.text(x + 48 * scale, 697, "96 KiB = 64 + 32", 22,
               TEAL, 600, "middle")
    ticks(612, (0, 64, 96, 160, 192))
    d.note("带 padding：两槽共 192 KiB，载荷仍为 128 KiB；灰色区域不含有效权重。")
    d.save("ldg-blocked-padding.svg")


def main():
    for draw in (miniblocks, seven_dim, blocked, uscale, index_pages,
                 layout_overview, miniblock_wl1, blocked_padding):
        draw()


if __name__ == "__main__":
    main()
