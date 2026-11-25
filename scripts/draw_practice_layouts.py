#!/usr/bin/env python3
"""Draw a concrete UScale coordinate-to-byte mapping."""

from draw_figures import Diagram, AQUA, GOLD, LINE, MUTED, NAVY, TEAL


def uscale_coordinate():
    d = Diagram(490, "UScale 系数 S[145,0] 到物理字节偏移")
    d.label(20, 30, "N=256 · K=128 · G=128 · 每通道一个 BF16 系数", NAVY)
    d.box(20, 62, 206, 82, "S[145, 0]", "逻辑通道 145，K 组 0", AQUA)
    d.path([(232, 103), (274, 103)])
    d.box(282, 62, 310, 82, "145 = 128×1 + 16×1 + 1",
          "band 1 · lane 1 · row 1", AQUA, size=20)
    d.path([(599, 103), (641, 103)])
    d.box(650, 62, 230, 82, "段号 = 1×1 + 0 = 1",
          "段内索引 row=1", GOLD, size=20)

    d.text(20, 198, "物理页 0：八路各占 2 KiB，合计 16 KiB", 22, NAVY, 600)
    width = 860 / 8
    for lane in range(8):
        x = 20 + lane * width
        d.rect(x, 220, width, 66, GOLD if lane == 1 else AQUA, LINE, r=0)
        d.text(x + width / 2, 248, f"lane {lane}", 19, NAVY,
               600, "middle")
        d.text(x + width / 2, 273, "2 KiB", 17, MUTED,
               anchor="middle")
    d.path([(181, 292), (181, 328), (291, 328), (291, 351)], TEAL)

    d.text(20, 335, "展开 lane 1", 20, NAVY, 600)
    d.rect(20, 359, 205, 60, AQUA, LINE, r=0)
    d.rect(225, 359, 205, 60, GOLD, LINE, r=0)
    d.rect(430, 359, 450, 60, "#F2F4F6", LINE, r=0)
    d.text(122, 384, "段 0：n=16…31", 18, NAVY, 600, "middle")
    d.text(122, 407, "lane 内 [0,32) B", 17, MUTED, anchor="middle")
    d.text(327, 384, "段 1：n=144…159", 18, NAVY, 600, "middle")
    d.text(327, 407, "lane 内 [32,64) B", 17, MUTED, anchor="middle")
    d.text(655, 393, "补齐至 2 KiB（示意不按比例）", 18, MUTED,
           anchor="middle")
    d.text(20, 460, "S[145,0]：0×16384 + 1×2048 + 1×32 + 1×2 = 2082 B",
           22, TEAL, 600)
    d.save("uscale-coordinate.svg")


if __name__ == "__main__":
    uscale_coordinate()
