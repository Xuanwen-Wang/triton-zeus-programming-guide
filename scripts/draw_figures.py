#!/usr/bin/env python3
"""Draw original, editable SVG diagrams for the programming guide."""

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
NAVY = "#183D52"
TEAL = "#147D89"
BLUE = "#387EA8"
AMBER = "#B87924"
INK = "#203D4D"
MUTED = "#627987"
LINE = "#B6CBD5"
PALE = "#F0F5F8"
AQUA = "#E7F3F3"
GOLD = "#FFF2DC"


class Diagram:
    def __init__(self, height, title):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="900" '
            f'height="{height}" viewBox="0 0 900 {height}">',
            f'<title>{escape(title)}</title>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" '
            'refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" '
            f'fill="{MUTED}"/></marker></defs>',
            '<g font-family="Noto Sans CJK SC, sans-serif">',
        ]

    def rect(self, x, y, w, h, fill=PALE, stroke=LINE, r=8):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )

    def text(self, x, y, text, size=21, fill=INK, weight=400,
             anchor="start", line=30):
        for i, s in enumerate(str(text).split("\n")):
            self.parts.append(
                f'<text x="{x}" y="{y+i*line}" font-size="{size}" '
                f'fill="{fill}" font-weight="{weight}" '
                f'text-anchor="{anchor}">{escape(s)}</text>'
            )

    def box(self, x, y, w, h, title, subtitle=None, fill=PALE,
            color=NAVY, size=22):
        self.rect(x, y, w, h, fill)
        cy = y + h / 2 + (7 if not subtitle else -5)
        self.text(x+w/2, cy, title, size, color, 600, "middle")
        if subtitle:
            self.text(x+w/2, cy+30, subtitle, 18, MUTED, anchor="middle")

    def path(self, points, color=MUTED, dashed=False, arrow=True):
        pts = " ".join(f"{x},{y}" for x, y in points)
        extra = ' stroke-dasharray="7 6"' if dashed else ""
        extra += ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<polyline points="{pts}" fill="none" '
                          f'stroke="{color}" stroke-width="2"{extra}/>')

    def label(self, x, y, text, color=TEAL):
        self.text(x, y, text, 20, color, 600)

    def note(self, text):
        self.text(18, self.height-12, text, 18, MUTED)

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/name).write_text("\n".join(self.parts+["</g></svg>\n"]),
                              encoding="utf-8")


def architecture():
    d = Diagram(535, "Zeus V3 编程可见的职责分工")
    d.box(18, 18, 240, 105, "Host 宿主", "Python · ZECC · 运行时")
    d.box(340, 18, 540, 105, "Device 设备", "加载程序 · 分配参数 · 执行任务")
    d.path([(258, 70), (340, 70)])
    d.text(299, 51, "提交", 18, MUTED, anchor="middle")
    for x, name in [(18, "Core 0"), (460, "Core 1")]:
        d.rect(x, 163, 420, 300, fill="#FFFFFF")
        d.label(x+20, 196, name, NAVY)
        d.box(x+20, 215, 110, 70, "CT", "控制流")
        d.box(x+148, 215, 110, 70, "VP", "向量/归约", AQUA)
        d.box(x+278, 215, 120, 70, "PE", "矩阵运算", AQUA)
        d.box(x+20, 321, 252, 76, "L2 Buffer", "暂存 tile 与中间结果")
        d.box(x+293, 321, 105, 76, "DT", "转换/搬运")
        d.path([(x+203, 321), (x+203, 285)])
        d.path([(x+293, 358), (x+272, 358)])
        d.text(x+20, 437, "LD / ST 连接外部数据存储", 19, MUTED)
        d.path([(610, 123), (610, 143), (x+210, 143), (x+210, 163)])
    d.note("抽象职责图。权重有专用路径；图中不表示容量、数量比例或峰值性能。")
    d.save("architecture.svg")


def tile_mapping():
    d = Diagram(390, "2500 个元素在两个核上的块分配")
    d.label(18, 30, "numel = 2500     BLOCK = 1024     CORE_NUM = 2", NAVY)
    for i, (title, sub, fill) in enumerate([
        ("tile 0", "[0, 1024)", AQUA),
        ("tile 1", "[1024, 2048)", AQUA),
        ("tile 2", "[2048, 2500)", GOLD),
    ]):
        d.box(18+i*294, 65, 270, 90, title, sub, fill)
    d.box(18, 245, 565, 78, "Core 0", "2 个 tile · 2048 个有效元素", AQUA)
    d.box(606, 245, 270, 78, "Core 1", "1 个 tile · 452 个有效元素", GOLD)
    d.path([(153, 155), (153, 205), (300, 205), (300, 245)])
    d.path([(447, 155), (447, 205), (300, 205)], arrow=False)
    d.path([(741, 155), (741, 245)])
    d.text(721, 201, "572 个补齐位置", 18, MUTED, anchor="end")
    d.note("最后一个 tile 读越界部分补零，只写回有效的 452 个元素。")
    d.save("tile-mapping.svg")


def memory_path():
    d = Diagram(445, "向量加法与矩阵乘的数据路径")
    d.label(18, 29, "向量加法")
    names = [("GDG", "x、z"), ("L2 Buffer", "输入 tile"),
             ("VP", "FP32 运算"), ("L2 Buffer", "输出 tile"), ("GDG", "y")]
    for i, (t, s) in enumerate(names):
        d.box(18+i*176, 60, 158, 80, t, s, AQUA if i == 2 else PALE)
        if i < 4:
            d.path([(176+i*176, 100), (194+i*176, 100)])
    d.text(185, 51, "LD", 18, MUTED, anchor="middle")
    d.text(713, 51, "ST", 18, MUTED, anchor="middle")
    d.label(18, 202, "矩阵乘")
    d.box(18, 230, 228, 80, "L2 Buffer", "activation tile")
    d.box(18, 330, 228, 80, "LDG", "已格式化的权重 tile")
    d.box(359, 260, 174, 92, "PE", "乘加与累加", AQUA)
    d.box(647, 260, 228, 92, "L2 Buffer", "累加结果 / 输出")
    d.path([(246, 270), (309, 270), (309, 287), (359, 287)])
    d.path([(246, 370), (309, 370), (309, 326), (359, 326)])
    d.path([(533, 307), (647, 307)])
    d.text(587, 287, "结果", 18, MUTED, anchor="middle")
    d.note("概念数据路径。DT 可参与布局转换；它不代表所有外部内存搬运。")
    d.save("memory-path.svg")


def dependency():
    d = Diagram(430, "发布、执行完成与缓冲区复用的依赖")
    d.label(18, 32, "同一个 tile 的数据依赖", NAVY)
    d.box(18, 63, 250, 79, "LOAD 完成", "输入数据就绪")
    d.box(325, 63, 250, 79, "VP 完成", "结果数据就绪", AQUA)
    d.box(632, 63, 250, 79, "STORE 完成", "结果可回收")
    d.path([(268, 102), (325, 102)])
    d.path([(575, 102), (632, 102)])
    d.text(297, 163, "RAW", 18, MUTED, anchor="middle")
    d.text(603, 163, "RAW", 18, MUTED, anchor="middle")
    d.label(18, 220, "复用同一块 L2 Buffer 之前", NAVY)
    d.box(18, 248, 365, 85, "旧 tile 的最后一个 reader 完成", "释放缓冲区所有权", GOLD, AMBER, 21)
    d.box(497, 248, 385, 85, "下一 tile 才能覆盖缓冲区", "避免读到被覆盖的数据", AQUA, TEAL, 21)
    d.path([(383, 290), (497, 290)])
    d.text(440, 270, "WAR", 18, MUTED, anchor="middle")
    d.note("提交命令 ≠ 执行完成。箭头表示必须满足的先后关系，不表示时间长度。")
    d.save("dependency-timeline.svg")


def stride():
    d = Diagram(368, "逻辑列数与物理行步长")
    d.label(18, 30, "shape = (3, 4)      strides = (6, 1)      单位为元素", NAVY)
    for row in range(3):
        d.text(18, 99+74*row, f"row {row}", 21, MUTED)
        for col in range(6):
            x, y = 125+col*123, 61+row*74
            d.rect(x, y, 109, 57, AQUA if col < 4 else GOLD)
            d.text(x+55, y+36, str(row*6+col) if col < 4 else "间隔", 20,
                   INK if col < 4 else AMBER, anchor="middle")
    d.text(125, 310, "address(row, col) = base + row × 6 + col", 23, NAVY, 500)
    d.note("逻辑有效列只占前 4 列。shape 决定边界，stride 决定地址间隔。")
    d.save("tensor-stride.svg")


def interface_map():
    d = Diagram(375, "存储类型和计算类型的边界")
    d.box(18, 18, 864, 64, "访存契约", "shape · stride · 指针所属存储 · 有效边界")
    labels = [("BF16", "输入存储"), ("FP32", "显式转换"),
              ("计算", "算术 / 归约"), ("BF16", "结果舍入"), ("写回", "有效范围")]
    for i, (t, s) in enumerate(labels):
        d.box(18+i*176, 149, 158, 85, t, s, AQUA if i in [1, 2] else PALE)
        if i < 4:
            d.path([(176+i*176, 191), (194+i*176, 191)])
    d.text(185, 132, "load", 18, MUTED, anchor="middle")
    d.text(713, 132, "store", 18, MUTED, anchor="middle")
    d.box(18, 284, 420, 59, "编译期：constexpr 决定特化", fill=PALE, size=21)
    d.box(460, 284, 422, 59, "运行期：地址、尺寸与标量参数", fill=PALE, size=21)
    d.save("interface-map.svg")


def compiler():
    d = Diagram(490, "Triton Zeus 编译流水线")
    stages = [
        ("Triton Python", "kernel、signature、constants"),
        ("TTIR", "张量形状、类型与指针表达式"),
        ("Linalg / MemRef", "结构化计算、循环与缓冲区"),
        ("ZeusHigh", "执行单元、描述符与存储分配"),
        ("ZeusLow", "设备命令、控制流与同步"),
        (".zbin", "供运行时加载的设备程序"),
    ]
    for i, (t, s) in enumerate(stages):
        y = 15+i*72
        d.box(18, y, 310, 55, t, fill=AQUA if i in [3, 4] else PALE)
        d.text(385, y+35, s, 22)
        if i < 5:
            d.path([(173, y+55), (173, y+72)])
    d.note(".S 与 .debug 用于诊断；.o 可封装设备二进制供宿主链接。")
    d.save("compiler-pipeline.svg")


def jit_aot():
    d = Diagram(455, "JIT、AOT 与 hybrid 使用同一特化身份")
    d.box(205, 15, 490, 68, "确定 kernel 特化身份", "源码 · 常量 · 签名 · 目标 · 选项")
    for x, name, s in [(18, "JIT", "查缓存；缺失则编译"),
                        (315, "AOT", "查 bundle；缺失则报错"),
                        (612, "hybrid", "先 bundle；缺失走 JIT")]:
        d.box(x, 153, 270, 90, name, s, AQUA if name == "JIT" else PALE)
        d.path([(450, 83), (450, 117), (x+135, 117), (x+135, 153)])
        d.path([(x+135, 243), (x+135, 278), (450, 278)], arrow=False)
    d.path([(450, 278), (450, 309)])
    d.box(130, 309, 640, 83, "运行时打包参数并提交当前 stream", "设备或明确配置的仿真后端", AQUA)
    d.note("运行时参数值变化通常复用特化；constexpr 与编译属性变化可能产生新特化。")
    d.save("jit-aot.svg")


def debug_flow():
    d = Diagram(405, "按症状缩小问题范围")
    rows = [("编译失败", "第一个失败的 IR 阶段", "类型、形状、布局\n接口约束"),
            ("执行异常", "参数与设备访问", "ABI、地址、分配\n所有权与同步"),
            ("输出异常", "数值与有效范围", "输入、舍入、中间值\n有效范围与尾部")]
    for i, (a, b, c) in enumerate(rows):
        y = 20+i*116
        d.box(18, y, 170, 82, a, fill=GOLD, color=AMBER)
        d.box(242, y, 280, 82, b, fill=AQUA, size=21)
        d.path([(188, y+41), (242, y+41)])
        d.text(552, y+32, c, 20, line=31)
    d.note("先固定输入、参数和版本，再减少 kernel 与失败形状；每次只改变一个因素。")
    d.save("debug-flow.svg")


def validation():
    d = Diagram(450, "不同验证环境回答不同问题")
    rows = [("静态检查 / 编译", "语法、IR 合法性与代码生成", PALE),
            ("功能仿真", "给定输入的输出与参考是否一致", AQUA),
            ("CycleSim", "模型中的时序、空闲与流水间隔", PALE),
            ("VCS / RTL", "RTL 路径的功能与时序", AQUA),
            ("真实设备", "实际执行、数值与设备耗时", PALE)]
    for i, (t, s, fill) in enumerate(rows):
        y = 12+i*78
        d.box(18, y, 240, 61, t, fill=fill, size=21)
        d.text(300, y+38, s, 22)
    d.note("每层都绑定自己的输入与版本。通过一层，不能自动推断其他层通过。")
    d.save("validation-layers.svg")


def performance():
    d = Diagram(498, "串行处理与理想双缓冲的抽象时间轴")
    d.label(18, 28, "串行：一个 tile 完整结束后处理下一个", NAVY)
    for i in range(3):
        start = 171+i*228
        for dx, w, t, c in [(0, 76, f"L{i}", BLUE), (76, 114, f"C{i}", TEAL),
                             (190, 38, f"S{i}", AMBER)]:
            d.rect(start+dx, 58, w-3, 47, c, c, 4)
            d.text(start+dx+(w-3)/2, 89, t, 20, "#FFFFFF", 500, "middle")
    d.text(18, 89, "依次执行", 21)
    d.label(18, 171, "流水：在依赖允许时重叠，覆盖旧缓冲前等待 reader", NAVY)
    for y, lane in [(202, "LOAD"), (268, "COMPUTE"), (334, "STORE")]:
        d.text(18, y+31, lane, 20)
        d.path([(158, y+45), (867, y+45)], color=LINE, arrow=False)
    for i, start in [(0, 0), (1, 76), (2, 228)]:
        d.rect(171+start, 202, 73, 45, BLUE, BLUE, 4)
        d.text(207+start, 232, f"L{i}", 20, "#FFFFFF", 500, "middle")
    for i, start in [(0, 76), (1, 190), (2, 304)]:
        d.rect(171+start, 268, 111, 45, TEAL, TEAL, 4)
        d.text(226+start, 298, f"C{i}", 20, "#FFFFFF", 500, "middle")
    for i, start in [(0, 190), (1, 304), (2, 418)]:
        d.rect(171+start, 334, 35, 45, AMBER, AMBER, 4)
        d.text(188+start, 364, f"S{i}", 18, "#FFFFFF", 500, "middle")
    d.path([(399, 330), (399, 252)], dashed=True)
    d.text(462, 420, "L = load    C = compute    S = store", 20, MUTED, anchor="middle")
    d.text(462, 452, "缓冲槽：tile 0、2 使用 ping；tile 1 使用 pong", 20, MUTED, anchor="middle")
    d.note("示意长度不代表实测周期；可否重叠取决于资源、缓冲区和编译器生成的依赖。")
    d.save("performance-timeline.svg")


def rmsnorm():
    d = Diagram(476, "RMSNorm 的有效列和补齐列")
    d.label(18, 28, "数学示意：M = 3，H = 6，tile = 4 × 8", NAVY)
    for r in range(4):
        for c in range(8):
            x, y = 20+c*49, 64+r*56
            valid = r < 3 and c < 6
            d.rect(x, y, 44, 44, AQUA if valid else GOLD, r=4)
            d.text(x+22, y+29, "x" if valid else "0", 20,
                   TEAL if valid else AMBER, anchor="middle")
    d.text(20, 324, "3 行 × 6 列有效数据", 20, TEAL)
    d.text(20, 354, "多出的行不写回；补齐列参与 sum 时为 0", 18, MUTED)
    d.path([(425, 157), (486, 157)])
    d.box(500, 59, 380, 70, "FP32 平方并沿列求和", fill=AQUA)
    d.box(500, 155, 380, 70, "除以真实 H = 6，再加 ε", fill=GOLD, color=AMBER)
    d.box(500, 251, 380, 70, "rsqrt → 逐元素缩放", fill=AQUA)
    d.path([(690, 129), (690, 155)])
    d.path([(690, 225), (690, 251)])
    d.text(500, 369, "y = x × rsqrt(sum(x²) / H + ε) × w", 20, NAVY)
    d.note("列 tile 的宽度 8 用于组织计算；归一化分母始终是逻辑隐藏维度 H。")
    d.save("rmsnorm.svg")


def matmul():
    d = Diagram(500, "矩阵乘在 K 维上的三次累加")
    d.label(18, 28, "M = N = 256，K = 384；BM = BN = BK = 128", NAVY)
    d.text(18, 75, "A 行块：128 × 384", 21, NAVY, 500)
    d.text(475, 75, "W 逻辑行块：128 × 384（N × K）", 21, NAVY, 500)
    for i in range(3):
        d.box(18+i*129, 100, 119, 87, f"A{i}", "128 × 128", AQUA, size=24)
        d.box(475+i*129, 100, 119, 87, f"W{i}", "128 × 128", PALE, size=24)
    d.text(18, 224, "K 块 0", 19, MUTED)
    d.text(147, 224, "K 块 1", 19, MUTED)
    d.text(276, 224, "K 块 2", 19, MUTED)
    d.box(18, 280, 260, 83, "A0 · W0ᵀ", "FP32 accumulator", AQUA)
    d.box(320, 280, 260, 83, "+ A1 · W1ᵀ", "保持同一个累加器", AQUA)
    d.box(622, 280, 260, 83, "+ A2 · W2ᵀ", "完成后转换并写回", AQUA)
    d.path([(278, 322), (320, 322)])
    d.path([(580, 322), (622, 322)])
    d.text(450, 421, "C[128,128] = Σₖ tl.dot(A[k], tl.trans(W[k]))", 23,
           NAVY, 500, "middle")
    d.note("图中展示一个 C tile；全矩阵还有其他 M/N tile，彼此拥有独立输出范围。")
    d.save("matmul-tiles.svg")


def fusion():
    d = Diagram(492, "融合保留残差输出及其 BF16 舍入")
    d.label(18, 31, "独立算子链：两次启动", NAVY)
    d.box(18, 65, 211, 79, "Add", "FP32 加法", AQUA)
    d.box(318, 65, 265, 79, "R = BF16(x + r)", "写回 residual", GOLD, AMBER)
    d.box(670, 65, 211, 79, "RMSNorm", "重新读取 R", AQUA)
    d.path([(229, 105), (318, 105)])
    d.path([(583, 105), (670, 105)])
    d.text(780, 182, "→ 输出 y", 20, MUTED, anchor="middle")
    d.label(18, 242, "融合算子：一次启动，保留同样的数值边界", NAVY)
    d.rect(18, 271, 863, 151, fill="#FFFFFF")
    d.box(36, 294, 200, 79, "Add", "FP32 加法", AQUA)
    d.box(299, 294, 260, 79, "R = BF16(x + r)", "转换回 FP32 参与归一化", GOLD, AMBER, 21)
    d.box(624, 294, 237, 79, "RMSNorm", "生成 y", AQUA)
    d.path([(236, 334), (299, 334)])
    d.path([(559, 334), (624, 334)])
    d.text(429, 404, "输出 residual", 18, MUTED, anchor="middle")
    d.text(742, 404, "输出 y", 18, MUTED, anchor="middle")
    d.note("融合减少中间结果重读与启动；如果外部仍使用 residual，就必须保留该输出。")
    d.save("fusion.svg")


def migration():
    d = Diagram(405, "GPU Triton 向 Zeus 的迁移顺序")
    cells = [(18, 20, "01  固定算子契约", "形状 · dtype · 布局 · 误差"),
             (318, 20, "02  跑通单核", "Zeus 运行入口与 ABI"),
             (618, 20, "03  覆盖尾块", "非整块 · 动态值 · 边界"),
             (618, 213, "04  核对双核分工", "输出归属 · 空核 · 同步"),
             (318, 213, "05  同条件测量", "基线 · 时间范围 · trace"),
             (18, 213, "06  框架回归", "真实调用链 · 输出质量")]
    for x, y, t, s in cells:
        d.box(x, y, 264, 103, t, s, AQUA if t[:2] in ["02", "04"] else PALE, size=21)
    d.path([(282, 72), (318, 72)])
    d.path([(582, 72), (618, 72)])
    d.path([(750, 123), (750, 213)])
    d.path([(618, 265), (582, 265)])
    d.path([(318, 265), (282, 265)])
    d.note("每一步先满足当前条件，再进入下一步；GPU 的 launch 参数需要重新解释。")
    d.save("migration.svg")


def reading_route():
    d = Diagram(329, "从第一个算子走向系统开发")
    for i, (t, s) in enumerate([
        ("01–02", "认识软件栈\n跑通向量加法"),
        ("03–05", "硬件与编程模型\n理解类型和接口"),
        ("06–08", "部署、调试与测量\n建立开发闭环"),
        ("09–10", "完成典型算子\n迁移现有工作负载"),
    ]):
        x = 18+i*220
        d.rect(x, 25, 204, 220, AQUA if i in [1, 2] else PALE)
        d.text(x+102, 78, t, 33, TEAL, 600, "middle")
        d.text(x+102, 143, s, 20, INK, anchor="middle", line=39)
        if i < 3:
            d.path([(x+204, 135), (x+220, 135)])
    d.note("附录按任务提供接口、选项、术语与示例索引。")
    d.save("reading-route.svg")


def error_metrics():
    d = Diagram(348, "比较前先统一参考和数值边界")
    d.box(18, 20, 240, 80, "同一份输入", "相同 dtype 与数据内容")
    d.box(360, 20, 230, 80, "Zeus 输出", fill=AQUA)
    d.box(360, 160, 230, 80, "参考输出", "相同的中间舍入边界")
    d.path([(258, 60), (360, 60)])
    d.path([(258, 60), (305, 60), (305, 200), (360, 200)])
    d.box(664, 75, 218, 115, "比较", "绝对误差 + 相对误差", GOLD, AMBER)
    d.path([(590, 60), (625, 60), (625, 112), (664, 112)])
    d.path([(590, 200), (625, 200), (625, 153), (664, 153)])
    d.text(450, 291, "|actual − reference| ≤ atol + rtol × |reference|", 24,
           NAVY, 500, "middle")
    d.note("先定义 NaN / Inf 与零附近的处理，再判断是否通过；不要用平均误差掩盖局部失败。")
    d.save("error-metrics.svg")


def ownership():
    d = Diagram(337, "分工先确定输出所有权")
    d.box(18, 25, 290, 80, "Core 0", "拥有 tile 0–3 的输出", AQUA)
    d.box(592, 25, 290, 80, "Core 1", "拥有 tile 4–7 的输出", PALE)
    for i in range(8):
        d.rect(18+i*109, 187, 104, 65, AQUA if i < 4 else PALE, r=5)
        d.text(70+i*109, 227, f"tile {i}", 19, TEAL if i < 4 else NAVY,
               anchor="middle")
    d.path([(163, 105), (163, 147), (234, 147), (234, 187)])
    d.path([(737, 105), (737, 147), (670, 147), (670, 187)])
    d.note("示意为等量工作。真实分配还需考虑尾块、空核，以及外部存储事务的所有权。")
    d.save("output-ownership.svg")


def executor_models():
    d = Diagram(405, "Multi-Executor 与 Parallel-Executor 的区别")
    d.label(18, 27, "Multi-Executor：一个逻辑操作的内部拆分", NAVY)
    d.box(18, 56, 240, 100, "操作 A", "一份逻辑输入与输出")
    d.box(380, 56, 230, 100, "A 的分片 0", "执行 lane 0", AQUA)
    d.box(652, 56, 230, 100, "A 的分片 1", "执行 lane 1", AQUA)
    d.path([(258, 106), (380, 106)])
    d.path([(318, 106), (318, 178), (767, 178), (767, 156)])
    d.label(18, 226, "Parallel-Executor：多个完整且独立的逻辑操作", NAVY)
    d.box(18, 253, 240, 99, "独立操作 A、B", "分别拥有各自的输出")
    d.box(380, 253, 230, 99, "完整操作 A", "执行 lane 0", AQUA)
    d.box(652, 253, 230, 99, "完整操作 B", "执行 lane 1", AQUA)
    d.path([(258, 303), (380, 303)])
    d.path([(318, 303), (318, 370), (767, 370), (767, 352)])
    d.note("是否能够并行由操作、布局、资源与依赖共同决定；图中只示意两个 lane。")
    d.save("executor-models.svg")


def reproduction_record():
    d = Diagram(352, "一次验证需要绑定的材料")
    d.box(232, 15, 436, 63, "同一次运行的唯一记录", "用文件校验值绑定内容")
    rows = [(18, "计算意图", "kernel · signature · 常量", "版本 · 命令 · 执行环境"),
            (315, "实际输入", "二进制 · 输入张量", "动态值 · stride · 随机种子"),
            (612, "执行结果", "参考输出 · 实际输出", "误差 · 日志 · trace")]
    for x, title, first, second in rows:
        d.rect(x, 153, 270, 133, AQUA if x == 315 else PALE)
        d.text(x+135, 186, title, 23, TEAL, 600, "middle")
        d.text(x+135, 225, first, 19, INK, anchor="middle")
        d.text(x+135, 257, second, 19, INK, anchor="middle")
        d.path([(450, 78), (450, 115), (x+135, 115), (x+135, 153)])
    d.note("仅重现文件名或 shape 不够；数据、程序、版本和判定规则都要对应。")
    d.save("reproduction-record.svg")


def argument_abi():
    d = Diagram(360, "参数值的地址与设备数据地址")
    d.label(18, 28, "指针参数：以向量加法的输出参数为例", NAVY)
    d.box(18, 56, 185, 81, "args[0]", "宿主指针数组")
    d.box(305, 56, 266, 81, "保存 out 地址的槽位", "内容是设备地址", GOLD, AMBER, 21)
    d.box(671, 56, 211, 81, "设备输出 buffer", "保存 BF16 元素", AQUA, size=21)
    d.path([(203, 96), (305, 96)])
    d.path([(571, 96), (671, 96)])
    d.label(18, 206, "标量参数：以运行时 numel 为例", NAVY)
    d.box(18, 233, 185, 81, "args[3]", "宿主指针数组")
    d.box(305, 233, 266, 81, "保存 numel 的槽位", "内容是 i32 数值", GOLD, AMBER, 21)
    d.path([(203, 274), (305, 274)])
    d.text(623, 263, "签名确定读取宽度", 21, MUTED)
    d.text(623, 295, "此处读取 32 位整数", 21, MUTED)
    d.note("运行时根据签名打包参数值；不能把设备 buffer 首地址直接当作参数值槽位。")
    d.save("argument-abi.svg")


def main():
    for fn in [architecture, tile_mapping, memory_path, dependency, stride,
               interface_map, compiler, jit_aot, debug_flow, validation,
               performance, rmsnorm, matmul, fusion, migration, reading_route,
               error_metrics, ownership, executor_models, reproduction_record,
               argument_abi]:
        fn()
    print(f"Rendered {len(list(OUT.glob('*.svg')))} SVG figures to {OUT}")


if __name__ == "__main__":
    main()
