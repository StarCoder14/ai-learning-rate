# -*- coding: utf-8 -*-
"""
진로주제 탐구 7단계 - 발표용 애니메이션 (GIF)

새로운 계산은 없다. 앞 단계의 학습 규칙을 한 칸씩 그려서 움직이는 그림으로 저장한다.
  애니1 : 1단계 거미줄 그림이 한 칸씩 그려지는 모습 (r = 0.4 계단 / 0.9 나선 / 1.1 발산)
  애니2 : 4b단계 f(x, y) = x² + 10y² 에서 관성 없음과 관성 있음이 동시에 출발하는 경주

실시간으로 학습률을 바꿔 보는 창은 stage7_slider.py 에 따로 있다.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------- 한글 폰트 설정 ----------------
def set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans KR"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ---------------- 저장 위치와 설정값 ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 폴더
FIG_DIR = os.path.join(ROOT, "figures")

COBWEB_RATES = [0.4, 0.9, 1.1]            # 애니1 : 계단 / 나선 / 발산
COBWEB_STEPS = 12                         # 애니1 : 몇 번 반복할지
A0 = 1.0                                  # 1단계와 같은 초기값
RACE_B = 10                               # 애니2 : y축의 뾰족한 정도
RACE_STEPS = 36                           # 애니2 : 관성 없음이 35번째 항에서 도달하므로 36칸
FPS = 3                                   # 1초에 3칸

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# ---------------- 학습 규칙 (앞 단계와 같음) ----------------
def cobweb_points(r, n=COBWEB_STEPS):
    """거미줄 계단의 꼭짓점들. 세로 이동(다음 항 계산)과 가로 이동(다음 입력으로)이 번갈아 나온다."""
    ratio = 1 - 2 * r
    x, px, py = A0, [A0], [0.0]
    for _ in range(n):
        y = ratio * x
        px += [x, y]
        py += [y, y]
        x = y
    return px, py

def momentum_path(r, m, n=RACE_STEPS):
    """4b단계와 같은 모멘텀 학습 규칙으로 (x, y) 점들을 만든다. m = 0 이면 3단계와 같다."""
    points, prev = [(1.0, 1.0)], (1.0, 1.0)
    for _ in range(n):
        x, y = points[-1]
        new = (x - r * 2 * x + m * (x - prev[0]), y - r * 2 * RACE_B * y + m * (y - prev[1]))
        prev = (x, y)
        points.append(new)
    return points

# ---------------- 공통 ----------------
def style_axes(ax):
    """격자와 테두리를 옅게 해서 데이터가 먼저 보이게 한다."""
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT)

def save_gif(fig, update, frames, name):
    """update(칸 번호)를 frames번 불러 GIF로 저장한다. 마지막 장면에서 잠깐 멈추도록 몇 칸 더 넣는다."""
    frame_list = list(range(frames)) + [frames - 1] * FPS * 2
    anim = FuncAnimation(fig, update, frames=frame_list, blit=False)
    anim.save(os.path.join(FIG_DIR, name), writer=PillowWriter(fps=FPS), dpi=110)
    plt.close(fig)

# ---------------- 애니1 : 거미줄 그림 ----------------
def setup_cobweb(ax, r, lim=1.6):
    """보라색 학습 규칙 직선과 점선 y = x를 미리 그려 두고, 계단 선만 비워 둔다."""
    ratio = 1 - 2 * r
    ax.plot([-lim, lim], [-lim, lim], color=INK_SOFT, linestyle="--", linewidth=0.8)
    ax.plot([-lim, lim], [-lim * ratio, lim * ratio], color="#4a3aa7", linewidth=1.5)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="a(n)")
    ax.set_title(f"r = {r}  (공비 {ratio:.1f})", fontsize=11)
    style_axes(ax)
    color = BLUE if abs(ratio) < 1 else RED
    line, = ax.plot([], [], color=color, linewidth=1.6)
    return line

def make_cobweb_gif():
    fig, axes = plt.subplots(1, len(COBWEB_RATES), figsize=(11, 4.7))
    lines = [setup_cobweb(ax, r) for ax, r in zip(axes, COBWEB_RATES)]
    walks = [cobweb_points(r) for r in COBWEB_RATES]
    axes[0].set_ylabel("a(n+1)")
    title = fig.suptitle("", fontsize=13)

    def update(k):
        for line, (px, py) in zip(lines, walks):
            line.set_data(px[:2 * k + 1], py[:2 * k + 1])     # k번째 항까지의 계단
        title.set_text(f"거미줄 그림: {k}번 반복 (보라색 = 학습 규칙, 점선 = y = x)")
        return lines

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_gif(fig, update, COBWEB_STEPS + 1, "애니1_거미줄그림.gif")

# ---------------- 애니2 : 관성 없음 vs 관성 있음 ----------------
def setup_race(ax, title, color, gx, gy):
    ax.contour(gx, gy, gx ** 2 + RACE_B * gy ** 2, levels=[0.05, 0.2, 0.5, 1, 2, 4],
               colors=GRID, linewidths=1.2)                  # 높이가 같은 점을 이은 선
    ax.plot(0, 0, marker="*", markersize=14, color=INK, zorder=4)
    ax.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), aspect="equal", xlabel="x")
    style_axes(ax)
    line, = ax.plot([], [], color=color, linewidth=1.5, marker="o", markersize=4,
                    markeredgecolor="white", zorder=3)
    label = ax.set_title(title, fontsize=11)
    return line, label

def race_cases():
    """(이름, 학습률, 관성, 색). 관성 없음은 3단계 최선, 관성 있음은 4b단계의 이론 최적."""
    r_best = 2 / (1 + np.sqrt(RACE_B)) ** 2
    m_best = ((np.sqrt(RACE_B) - 1) / (np.sqrt(RACE_B) + 1)) ** 2
    return [("관성 없음", 1 / (1 + RACE_B), 0, BLUE), ("관성 있음", r_best, m_best, ORANGE)]

def make_race_gif():
    cases = race_cases()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
    grid = np.linspace(-1.45, 1.45, 200)
    gx, gy = np.meshgrid(grid, grid)
    parts = [setup_race(ax, name, color, gx, gy) for ax, (name, _, _, color) in zip(axes, cases)]
    paths = [momentum_path(r, m) for _, r, m, _ in cases]
    axes[0].set_ylabel("y")
    fig.suptitle(f"f(x, y) = x² + {RACE_B}y²: 같은 곳에서 동시에 출발", fontsize=13)

    def update(k):
        for (line, label), path, (name, r, m, _) in zip(parts, paths, cases):
            line.set_data([p[0] for p in path[:k + 1]], [p[1] for p in path[:k + 1]])
            size = max(abs(path[k][0]), abs(path[k][1]))
            done = " 도착!" if size < 0.001 else ""
            label.set_text(f"{name} (r = {r:.3f}, m = {m:.2f})\n{k}번째 항, 거리 {size:.4f}{done}")
        return [p[0] for p in parts]

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_gif(fig, update, RACE_STEPS + 1, "애니2_관성경주.gif")

def main():
    make_cobweb_gif()
    make_race_gif()
    print("저장 완료: figures/애니1_거미줄그림.gif, figures/애니2_관성경주.gif")

main()
