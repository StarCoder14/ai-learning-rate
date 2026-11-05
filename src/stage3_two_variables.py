# -*- coding: utf-8 -*-
"""
진로주제 탐구 3단계 - 2변수 함수의 지그재그

질문: 변수가 두 개이고 한쪽이 더 뾰족하면 학습이 어떻게 달라지는가?

이론:
  f(x, y) = a·x^2 + b·y^2 (a <= b, 즉 y축이 더 뾰족하다)
  y를 고정하면 x에 대한 이차함수이므로 x방향 기울기는 2a·x, 마찬가지로 y방향 기울기는 2b·y (판별식으로 유도)
  학습 규칙  x(n+1) = (1 - 2ar)·x(n),   y(n+1) = (1 - 2br)·y(n)
  -> 서로 영향을 주지 않는 등비수열 두 개가 동시에 돈다.

이론 예측:
  예측 1 (안전 상한): 두 수열이 모두 수렴해야 하므로  0 < r < 1/a 이고 0 < r < 1/b
                      ->  0 < r < 1/max(a, b)   : 더 뾰족한 축이 상한을 정한다.
  예측 2 (최적 학습률): 느린 쪽 공비의 절댓값이 가장 작아지는 것은 두 절댓값이 같을 때
                      1 - 2ar = -(1 - 2br)  ->  r = 1/(a + b),  그때 공비의 절댓값 = (b - a)/(b + a)
                      -> b/a 가 클수록 가장 좋은 학습률로도 느려진다. y는 공비가 음수라 지그재그로 움직인다.
  예측 3 (작은 학습률): 모든 경우에 안전한 작은 r을 쓰면, 도달 속도는 완만한 x축이 정한다.
                      -> b를 바꿔도 도달한 항의 번호가 같다.
  예측 4 (코드 검증): a = b = 1 이면 x, y 각각 1단계와 같다.

검증 방법: 공비 공식을 쓰지 않고 학습 규칙(기울기만큼 빼기)을 그대로 반복해서 판정한다.
도달 판정은 두 좌표 중 절댓값이 큰 쪽이 TOL보다 작아졌는지로 한다.
"""

import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

# ---------------- 저장 위치 ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 폴더
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")

# ---------------- 설정값 ----------------
A = 1                                      # x축의 뾰족한 정도 (통제변인)
B_VALUES = [1, 2, 4, 10, 25]               # y축의 뾰족한 정도 b (조작변인)
X0, Y0 = 1.0, 1.0                          # 시작점 (통제변인)
N_TERMS = 3000                             # 최대 반복 횟수
TOL = 0.001                                # 도달 판정 기준

RANGE_RATES = [i / 200 for i in range(1, 221)]     # 실험 1 : r = 0.005 ~ 1.100
FINE_STEP = 1000                                   # 실험 2 : r을 0.001 간격으로 훑는다
SMALL_RATE = 0.03                                  # 실험 3 : 모든 b에서 안전한 작은 학습률

STYLE_BLUE, STYLE_ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# ---------------- 학습 과정 ----------------
def learn(b, r, a=A, n=N_TERMS):
    """두 좌표를 동시에 학습시킨 (x, y) 점들을 반환한다."""
    points = [(X0, Y0)]
    for _ in range(n):
        x, y = points[-1]
        x = x - r * (2 * a * x)              # x방향 기울기만큼 뺀다
        y = y - r * (2 * b * y)              # y방향 기울기만큼 뺀다
        points.append((x, y))
        size = max(abs(x), abs(y))
        if size > 1e6 or size < 1e-9:        # 충분히 커지거나 작아지면 중단
            break
    return points

def size_of(point):
    """두 좌표 중 절댓값이 큰 쪽 = 최솟값 (0, 0)에서 떨어진 정도."""
    return max(abs(point[0]), abs(point[1]))

def judge(points):
    """마지막 점의 크기로 수렴 / 진동 / 발산을 판정한다."""
    size = size_of(points[-1])
    if size < 1e-6:
        return "수렴"
    return "발산" if size > 100 else "진동"

def steps_to_reach(points):
    """크기가 TOL보다 작아진 첫 항의 번호. 두 수열 모두 크기가 계속 줄어들므로 다시 커지지 않는다."""
    for i, p in enumerate(points):
        if size_of(p) < TOL:
            return i
    return None

# ---------------- 이론값 계산 ----------------
def theory_steps(ratio_size):
    """공비의 절댓값을 몇 번 곱해야 TOL보다 작아지는지 센다."""
    value, n = 1.0, 0
    while value >= TOL:
        value, n = value * ratio_size, n + 1
    return n

# ---------------- CSV 저장 ----------------
def save_csv(name, header, rows):
    with open(os.path.join(RES_DIR, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------------- 실험 1 : 안전 상한 ----------------
def range_row(b):
    """b 하나에 대해 실제 수렴 경계를 이론 상한 1/max(a, b)와 비교한다."""
    result = {r: judge(learn(b, r)) for r in RANGE_RATES}
    ok = [r for r in RANGE_RATES if result[r] == "수렴"]
    fail = min(r for r in RANGE_RATES if result[r] != "수렴")
    upper = 1 / max(A, b)
    match = max(ok) < upper and np.isclose(fail, upper) and result[fail] == "진동"
    return [A, b, round(upper, 4), max(ok), fail, result[fail], "일치" if match else "불일치"]

# ---------------- 실험 2 : 가장 좋은 학습률과 속도 ----------------
def best_rate_row(b):
    """이론 최적 r에서 직접 돌린 결과와, 0.001 간격으로 훑어 찾은 가장 빠른 r을 비교한다."""
    upper = 1 / max(A, b)
    rates = [i / FINE_STEP for i in range(1, round(upper * FINE_STEP))]
    steps = {r: steps_to_reach(learn(b, r)) for r in rates}
    best = min(s for s in steps.values() if s is not None)
    fastest = [r for r in rates if steps[r] == best]
    r_theory = 1 / (A + b)
    ratio = (b - A) / (b + A)
    t_steps = theory_steps(ratio) if ratio > 0 else 1
    at_theory = steps_to_reach(learn(b, r_theory))
    # 이론 r에서 이론대로 도달하고, 격자의 어떤 r도 그보다 빠르지 않으면 일치
    match = at_theory == t_steps and best >= at_theory
    return [A, b, b / A, round(r_theory, 4), round(ratio, 4), t_steps, at_theory,
            min(fastest), max(fastest), best, "일치" if match else "불일치"]

# ---------------- 실험 3 : 작은 학습률에서의 속도 ----------------
def small_rate_row(b, r=SMALL_RATE):
    """작은 학습률에서 전체 도달 항과, x축만 / y축만 따로 본 도달 항을 비교한다."""
    points = learn(b, r)
    total = steps_to_reach(points)
    x_only = next(i for i, p in enumerate(points) if abs(p[0]) < TOL)
    y_only = next(i for i, p in enumerate(points) if abs(p[1]) < TOL)
    slower = "x축(완만)" if x_only >= y_only else "y축(뾰족)"
    ratio_x, ratio_y = 1 - 2 * A * r, 1 - 2 * b * r
    t_steps = theory_steps(max(abs(ratio_x), abs(ratio_y)))   # 느린 쪽 공비로 계산
    return [A, b, r, round(ratio_x, 4), round(ratio_y, 4), t_steps,
            x_only, y_only, total, slower, "일치" if t_steps == total else "불일치"]

# ---------------- 그림 공통 ----------------
def style_axes(ax):
    """격자와 테두리를 옅게 해서 데이터가 먼저 보이게 한다."""
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT)

def save_fig(fig, name, top=1.0):
    fig.tight_layout(rect=(0, 0, 1, top))       # top < 1 이면 전체 제목 자리를 남긴다
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close(fig)

# ---------------- 그림 9 : 등고선 위의 학습 경로 ----------------
def draw_one_path(ax, b, gx, gy):
    """등고선(높이가 같은 점을 이은 선) 위에 이론 최적 학습률로 학습한 경로를 그린다."""
    r = 1 / (A + b)
    points = learn(b, r)
    ax.contour(gx, gy, A * gx ** 2 + b * gy ** 2, levels=[0.05, 0.2, 0.5, 1, 2, 4],
               colors=GRID, linewidths=1.2)
    ax.plot([p[0] for p in points], [p[1] for p in points], color=STYLE_BLUE, linewidth=1.5,
            marker="o", markersize=5, markeredgecolor="white", zorder=3)
    ax.plot(0, 0, marker="*", markersize=14, color=INK, zorder=4)
    ax.set_title(f"b = {b}  (r = 1/{A + b} = {r:.3f})\n{steps_to_reach(points)}번째 항에서 도달",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    style_axes(ax)

def draw_contour_paths(bs=(1, 4, 10)):
    fig, axes = plt.subplots(1, len(bs), figsize=(13, 5))
    grid = np.linspace(-1.3, 1.3, 300)
    gx, gy = np.meshgrid(grid, grid)
    for ax, b in zip(axes, bs):
        draw_one_path(ax, b, gx, gy)
    axes[0].set_ylabel("y")
    fig.suptitle("f(x, y) = x² + b·y²에서 가장 좋은 학습률로 학습한 경로 (★ = 최솟값)", fontsize=13)
    save_fig(fig, "그림9_2변수_지그재그.png", top=0.93)

# ---------------- 그림 10 : 뾰족함의 차이와 학습 속도 ----------------
def draw_speed(best_rows, small_rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = range(len(best_rows))                      # b/a 값을 같은 간격으로 놓는다
    ax.plot(pos, [row[8] for row in small_rows], color=STYLE_ORANGE, linewidth=2, marker="s",
            markersize=7, markeredgecolor="white", zorder=3,
            label=f"모두 같은 작은 학습률 r = {SMALL_RATE}")
    ax.plot(pos, [row[6] for row in best_rows], color=STYLE_BLUE, linewidth=2, marker="o",
            markersize=8, markeredgecolor="white", zorder=3,
            label="각자 가장 좋은 학습률 r = 1/(a+b)")
    ax.plot(pos, [row[5] for row in best_rows], linestyle="none", marker="_", markersize=22,
            markeredgewidth=2, color=INK, zorder=4, label="이론으로 계산한 도달 항")
    for i, row in zip(pos, best_rows):
        ax.annotate(f"a{row[6]}", (i, row[6]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color=INK)
    ax.set_xticks(pos, [f"{row[2]:g}" for row in best_rows])
    ax.set(xlabel="뾰족함의 비 b/a", ylabel=f"크기 < {TOL}에 도달한 항의 번호",
           title="두 축의 뾰족함 차이가 클수록 학습이 느려진다")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=3)
    save_fig(fig, "그림10_2변수_속도.png")

# ---------------- 실행 ----------------
def report_range():
    rows = [range_row(b) for b in B_VALUES]
    save_csv("3단계_안전상한.csv", ["a", "b", "이론 상한 1/max(a,b)", "실험: 수렴한 가장 큰 r",
                                  "실험: 처음 수렴 안 한 r", "그때 판정", "이론과 비교"], rows)
    print("\n[실험 1 안전 상한]")
    for row in rows:
        print(f"  b={row[1]:<3} 이론 상한 {row[2]:<6g} 실험 {row[3]:g}~{row[4]:g}({row[5]})  {row[6]}")

def report_best():
    rows = [best_rate_row(b) for b in B_VALUES]
    save_csv("3단계_최적학습률.csv", ["a", "b", "b/a", "이론 최적 r = 1/(a+b)", "이론 공비 크기 (b-a)/(b+a)",
                                    "이론 도달 항", "실험: 이론 r로 돌린 도달 항",
                                    "실험: 격자에서 가장 빠른 r (최소)", "실험: 격자에서 가장 빠른 r (최대)",
                                    "실험: 격자 최소 도달 항", "이론과 비교"], rows)
    print("\n[실험 2 가장 좋은 학습률]")
    for row in rows:
        print(f"  b={row[1]:<3} 이론 r={row[3]:<7g} 공비 {row[4]:<7g} 이론 a{row[5]:<4} "
              f"이론 r로 실험 a{row[6]:<4} 격자 최선 r={row[7]:g}~{row[8]:g} a{row[9]:<4} {row[10]}")
    return rows

def report_small():
    rows = [small_rate_row(b) for b in B_VALUES]
    save_csv("3단계_작은학습률.csv", ["a", "b", "학습률 r", "x축 공비", "y축 공비", "이론 도달 항 (느린 쪽 공비)",
                                    "x축만 도달 항", "y축만 도달 항", "전체 도달 항", "속도를 정한 축",
                                    "이론과 비교"], rows)
    print(f"\n[실험 3 작은 학습률 r={SMALL_RATE}]")
    for row in rows:
        print(f"  b={row[1]:<3} 이론 a{row[5]:<4} x축 a{row[6]:<4} y축 a{row[7]:<4} "
              f"전체 a{row[8]:<4} ← {row[9]}  {row[10]}")
    return rows

def main():
    # 코드 검증: a = b = 1 이면 1단계와 같아야 한다 (r = 0.1일 때 20번째 항 0.0115)
    x20, y20 = learn(1, 0.1, n=20)[20]
    print(f"[코드 검증] b=1, r=0.1의 20번째 항 = ({x20:.4f}, {y20:.4f}) (1단계 결과 0.0115)")
    report_range()
    best_rows = report_best()
    small_rows = report_small()
    draw_contour_paths()
    draw_speed(best_rows, small_rows)
    print("\n저장 완료: results/3단계_*.csv, figures/그림9, 그림10")

main()
