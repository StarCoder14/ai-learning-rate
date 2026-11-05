# -*- coding: utf-8 -*-
"""
진로주제 탐구 7단계 - 발표 중 실시간 시연용 슬라이더

실행하면 창이 열린다. 아래 막대(슬라이더)로 학습률 r을 움직이면
1단계 f(x) = x² 의 거미줄 그림과 항의 변화가 바로 다시 그려진다.
  - r을 1 근처로 가져가면 정사각형(진동)이 되는 순간을 보여줄 수 있다.
  - r을 0.5로 두면 한 번에 도착한다.

실행: python src/stage7_slider.py
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import Slider

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

# ---------------- 설정값 ----------------
A0 = 1.0                 # 초기값
N_TERMS = 20             # 계산할 항의 개수
R_START = 0.4            # 처음 보여 줄 학습률
LIM = 1.6                # 거미줄 그림의 범위

BLUE, ORANGE, RED, VIOLET = "#2a78d6", "#1baf7a", "#e34948", "#4a3aa7"
INK_SOFT, GRID = "#52514e", "#e4e3df"

# ---------------- 계산 (1단계와 같음) ----------------
def make_sequence(r):
    """공비 (1 - 2r)인 등비수열의 항."""
    seq = [A0]
    for _ in range(N_TERMS):
        seq.append(seq[-1] * (1 - 2 * r))
    return seq

def cobweb(seq):
    """거미줄 계단의 꼭짓점들."""
    px, py = [seq[0]], [0.0]
    for a, b in zip(seq, seq[1:]):
        px += [a, b]
        py += [b, b]
    return px, py

def verdict(r):
    """이론으로 본 판정과 선 색."""
    ratio = abs(1 - 2 * r)
    if ratio < 1:
        return "수렴", BLUE
    return ("진동", ORANGE) if abs(ratio - 1) < 1e-9 else ("발산", RED)

# ---------------- 화면 ----------------
def build():
    """창을 만들고, 슬라이더를 움직일 때마다 다시 그리는 함수를 연결한다."""
    fig, (ax_web, ax_seq) = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.subplots_adjust(bottom=0.22, wspace=0.3)
    ax_web.plot([-LIM, LIM], [-LIM, LIM], color=INK_SOFT, linestyle="--", linewidth=0.8)
    rule, = ax_web.plot([], [], color=VIOLET, linewidth=1.5)
    web, = ax_web.plot([], [], linewidth=1.6)
    terms, = ax_seq.plot([], [], marker="o", markersize=4, linewidth=1.5)
    ax_web.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xlabel="a(n)", ylabel="a(n+1)")
    ax_seq.set(xlim=(0, N_TERMS), ylim=(-3, 3), xlabel="항의 번호 n", ylabel="a(n)")
    for ax in (ax_web, ax_seq):
        ax.grid(color=GRID)
    slider = Slider(fig.add_axes([0.2, 0.07, 0.6, 0.04]), "학습률 r", 0.0, 1.3,
                    valinit=R_START, valstep=0.01)
    parts = (fig, rule, web, terms, ax_web)
    slider.on_changed(lambda r: redraw(parts, r))
    redraw(parts, R_START)
    return fig, slider, parts

def redraw(parts, r):
    """학습률 r에 맞게 보라색 직선, 계단, 항의 그래프, 제목을 다시 그린다."""
    fig, rule, web, terms, ax_web = parts
    seq = make_sequence(r)
    label, color = verdict(r)
    rule.set_data([-LIM, LIM], [-LIM * (1 - 2 * r), LIM * (1 - 2 * r)])
    web.set_data(*cobweb(seq))
    terms.set_data(range(len(seq)), seq)
    web.set_color(color)
    terms.set_color(color)
    fig.suptitle(f"r = {r:.2f},  공비 1 - 2r = {1 - 2 * r:.2f}  →  {label}", fontsize=14)
    fig.canvas.draw_idle()

if __name__ == "__main__":
    figure, keep_slider, _ = build()      # 슬라이더를 변수에 담아 두어야 창에서 계속 작동한다
    plt.show()
