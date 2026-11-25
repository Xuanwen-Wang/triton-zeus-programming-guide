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


def main():
    for draw in (miniblocks, seven_dim, blocked, uscale, index_pages):
        draw()


if __name__ == "__main__":
    main()
