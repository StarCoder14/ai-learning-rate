# -*- coding: utf-8 -*-
"""
진로주제 탐구 4단계 - 모멘텀과 3항 점화식

질문: 관성을 주면 왜, 그리고 언제 더 빨리 학습하는가?

학습 규칙 (f(x) = x^2, 접선의 기울기 2x):
  x(n+1) = x(n) - r * 2x(n) + m * (x(n) - x(n-1))     <- 직전 이동량의 m배를 이어받음
         = (1 + m - 2r) x(n) - m x(n-1)                  <- 3항 점화식
  특성방정식  g(t) = t^2 - (1 + m - 2r) t + m = 0,  판별식 D = (1 + m - 2r)^2 - 4m

이론 예측:
  예측 1 (경로 모양): D > 0 이면 두 실근 -> 계단형 (부호가 많아야 한 번 바뀜)
                      D < 0 이면 허근   -> 출렁이며 수렴 (부호가 여러 번 바뀜)
  예측 2 (수렴 범위): 두 근이 모두 -1과 1 사이  <=>  g(1) > 0, g(-1) > 0, 두 근의 곱 m < 1
                      g(1) = 2r > 0,  g(-1) = 2 + 2m - 2r > 0   ->   0 < r < 1 + m  (단, 0 <= m < 1)
                      즉 관성을 주면 안전한 학습률의 상한이 1에서 1 + m 으로 넓어진다.
  예측 3 (속도): 근의 절댓값 중 큰 쪽이 속도를 정한다 (1단계의 |공비| 역할).
                 학습률 r이 작을 때, 중근이 되는 m = (1 - √(2r))^2 근처에서 가장 빠르다.
  예측 4 (코드 검증): m = 0 이면 1단계와 같다.

검증 방법: 특성방정식을 쓰지 않고 학습 규칙을 그대로 반복해서 판정한다.
"""

import os
import csv
import math
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
A0 = 1.0                                   # 초기값, 처음에는 움직인 적이 없으므로 x(-1) = x(0) (통제변인)
N_TERMS = 2000                             # 반복 횟수
TOL = 0.001                                # 도달 판정 기준

# 실험 1 : 경로 모양 (r < 0.5 라서 관성이 없으면 계단형인 경우들)
SHAPE_RATES = [0.05, 0.1, 0.2, 0.3, 0.4]
SHAPE_MS = [i / 20 for i in range(0, 20)]          # m = 0 ~ 0.95

# 실험 2 : 수렴 범위
RANGE_MS = [0, 0.3, 0.6, 0.9, 1.0]                 # 모멘텀 계수 m (조작변인 1)
RANGE_RATES = [i / 20 for i in range(1, 45)]       # 학습률 r = 0.05 ~ 2.20 (조작변인 2)

# 실험 3 : 속도 (학습률이 작은 경우)
SPEED_RATE = 0.05
SPEED_MS = [i / 20 for i in range(0, 20)]          # m = 0 ~ 0.95

STYLE = {"수렴": ("#2a78d6", "o"), "진동": ("#1baf7a", "D"), "발산": ("#eb6834", "X")}
SHAPE_COLOR = {"계단형": "#2a78d6", "출렁임": "#eb6834"}
INK, INK_SOFT, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"

# ---------------- 학습 과정 ----------------
def slope(x):
    """f(x) = x^2 위의 점 x에서 접선의 기울기 (판별식으로 구한 2x)."""
    return 2 * x

def learn(r, m, n=N_TERMS):
    """모멘텀 학습 규칙을 n번 반복한 x(0), x(1), ... 을 반환한다."""
    prev, xs = A0, [A0]
    for _ in range(n):
        x = xs[-1]
        move = x - prev                             # 직전에 움직인 양
        xs.append(x - r * slope(x) + m * move)      # 기울기만큼 빼고, 관성만큼 더 간다
        prev = x
        if abs(xs[-1]) > 1e6:                       # 너무 커지면 발산으로 보고 중단
            break
    return xs

# ---------------- 실험값 측정 ----------------
def sign_changes(xs):
    """부호가 바뀐 횟수. 너무 작아져 컴퓨터가 0으로 만드는 구간은 세지 않는다."""
    alive = [x for x in xs if abs(x) > 1e-200]
    return sum(1 for a, b in zip(alive, alive[1:]) if a * b < 0)

def judge(xs):
    """마지막 20개 항의 크기로 수렴 / 진동 / 발산을 판정한다."""
    size = max(abs(x) for x in xs[-20:])
    if size < 1e-6:
        return "수렴"
    return "발산" if size > 100 else "진동"

def steps_to_reach(xs):
    """|x| < TOL 이 된 뒤 다시는 벗어나지 않는 첫 항의 번호. 도달 못 하면 None."""
    if abs(xs[-1]) >= TOL:
        return None
    last_out = max((i for i, x in enumerate(xs) if abs(x) >= TOL), default=-1)
    return last_out + 1

# ---------------- 이론값 계산 ----------------
def discriminant(r, m):
    """특성방정식 t^2 - (1+m-2r)t + m = 0 의 판별식."""
    return (1 + m - 2 * r) ** 2 - 4 * m

def biggest_root_size(r, m):
    """두 근의 절댓값 중 큰 값. 허근이면 두 근의 곱이 m 이므로 절댓값은 √m."""
    c, d = 1 + m - 2 * r, discriminant(r, m)
    if d < 0:
        return math.sqrt(m)
    return (abs(c) + math.sqrt(d)) / 2

def theory_upper(m):
    """이론 상한 1 + m. m >= 1 이면 수렴하는 r이 없다."""
    return 1 + m if m < 1 else None

# ---------------- CSV 저장 ----------------
def save_csv(name, header, rows):
    with open(os.path.join(RES_DIR, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------------- 실험 1 : 판별식과 경로 모양 ----------------
def shape_rows():
    """판별식의 부호로 예측한 모양과 실제 부호가 바뀐 횟수로 본 모양을 비교한다."""
    rows = []
    for r in SHAPE_RATES:
        for m in SHAPE_MS:
            d = discriminant(r, m)
            predicted = "계단형" if d >= 0 else "출렁임"
            flips = sign_changes(learn(r, m))
            actual = "계단형" if flips <= 1 else "출렁임"
            rows.append([r, m, round(d, 4), predicted, flips, actual,
                         "일치" if predicted == actual else "불일치"])
    return rows

# ---------------- 실험 2 : 수렴 범위 ----------------
def range_rows():
    """각 m에서 실제로 수렴한 학습률의 경계를 이론 상한 1 + m 과 비교한다."""
    rows = []
    for m in RANGE_MS:
        result = {r: judge(learn(r, m)) for r in RANGE_RATES}
        ok = [r for r in RANGE_RATES if result[r] == "수렴"]
        fail = min((r for r in RANGE_RATES if result[r] != "수렴"), default=None)
        upper = theory_upper(m)
        if upper is None:
            match = not ok
        else:
            match = bool(ok) and max(ok) < upper and math.isclose(fail, upper)
        rows.append([m, upper if upper is not None else "없음", max(ok) if ok else "없음",
                     fail, result[fail], "일치" if match else "불일치", result])
    return rows

# ---------------- 실험 3 : 속도 ----------------
def speed_rows():
    """학습률이 작을 때 m에 따라 도달까지 걸린 항의 수를 잰다."""
    rows = []
    for m in SPEED_MS:
        d = discriminant(SPEED_RATE, m)
        steps = steps_to_reach(learn(SPEED_RATE, m))
        rows.append([SPEED_RATE, m, round(d, 4), "두 실근" if d >= 0 else "허근",
                     round(biggest_root_size(SPEED_RATE, m), 4), steps])
    return rows

# ---------------- 그림 공통 ----------------
def style_axes(ax):
    """격자와 테두리를 옅게 해서 데이터가 먼저 보이게 한다."""
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT)

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close(fig)

# ---------------- 그림 6 : 판별식과 경로 모양 ----------------
def draw_paths(r=SPEED_RATE, ms=(0, 0.3, 0.6, 0.9)):
    fig, axes = plt.subplots(1, len(ms), figsize=(13, 3.6), sharey=True)
    for ax, m in zip(axes, ms):
        xs = learn(r, m, n=80)
        d = discriminant(r, m)
        shape = "계단형" if d >= 0 else "출렁임"
        ax.axhline(0, color=INK_SOFT, linewidth=0.8)
        ax.plot(range(len(xs)), xs, color=SHAPE_COLOR[shape], linewidth=2)
        sign = "D > 0 · 두 실근" if d >= 0 else "D < 0 · 허근"
        ax.set_title(f"m = {m}   ({sign} → {shape})", fontsize=11)
        ax.set_xlabel("항의 번호 n")
        style_axes(ax)
    axes[0].set_ylabel("x(n)")
    fig.suptitle(f"학습률 r = {r}에서 관성 m에 따른 학습 경로", fontsize=13)
    save_fig(fig, "그림6_모멘텀_경로모양.png")

# ---------------- 그림 7 : 수렴 범위 ----------------
def draw_judgements(ax, row_i, result):
    """한 줄(m 하나)의 r별 판정을 색과 모양으로 찍는다."""
    for label, (color, marker) in STYLE.items():
        rs = [r for r in RANGE_RATES if result[r] == label]
        ax.scatter(rs, [row_i] * len(rs), s=38, color=color, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=3,
                   label=label if row_i == 0 else None)   # 범례에는 한 번씩만

def draw_range(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    for row_i, row in enumerate(rows):
        upper = row[1]
        draw_judgements(ax, row_i, row[6])
        if upper == "없음":
            ax.text(1.0, row_i + 0.2, "m = 1이면 어떤 학습률로도 수렴하지 않는다 (두 근의 곱 = 1)",
                    ha="center", va="bottom", fontsize=9, color=INK)
        else:
            ax.plot([upper] * 2, [row_i - 0.38, row_i + 0.38], color=INK, linewidth=2, zorder=4)
            ax.text(upper, row_i + 0.42, f"이론 상한 1+m = {upper:g}", ha="center",
                    va="bottom", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [f"m = {row[0]:g}" for row in rows])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("학습률 r")
    ax.set_title("관성 m을 주면 안전한 학습률의 범위가 넓어진다 (f(x) = x²)")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=3)
    save_fig(fig, "그림7_모멘텀_수렴범위.png")

# ---------------- 그림 8 : 속도 ----------------
def draw_speed(rows):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    best_m = (1 - math.sqrt(2 * SPEED_RATE)) ** 2          # 중근이 되는 m
    ax.axvspan(best_m, 1, color=SHADE, zorder=0)
    ms = [row[1] for row in rows if row[5] is not None]
    steps = [row[5] for row in rows if row[5] is not None]
    ax.plot(ms, steps, color="#2a78d6", linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", zorder=3)
    ax.axvline(best_m, color=INK, linewidth=1.5, linestyle="--", zorder=2)
    top = max(steps)
    ax.text(best_m, top, f" 중근 (D = 0)\n m = {best_m:.3f}", va="top", fontsize=10, color=INK)
    ax.text(best_m / 2, top * 0.08, "D > 0 · 두 실근", ha="center", color=INK_SOFT)
    ax.text((best_m + 1) / 2, top * 0.08, "D < 0 · 허근", ha="center", color=INK_SOFT)
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(0, top * 1.08)
    ax.set_xlabel("관성 m")
    ax.set_ylabel(f"|x| < {TOL}에 도달한 항의 번호")
    ax.set_title(f"학습률이 작을 때(r = {SPEED_RATE}) 관성에 따른 학습 속도")
    style_axes(ax)
    save_fig(fig, "그림8_모멘텀_속도.png")

# ---------------- 실행 ----------------
def report_shape():
    rows = shape_rows()
    save_csv("4단계_경로모양.csv", ["학습률 r", "관성 m", "판별식 D", "이론 모양",
                                   "실험: 부호 바뀐 횟수", "실험 모양", "비교"], rows)
    wrong = [row for row in rows if row[6] == "불일치"]
    print(f"\n[실험 1 경로 모양] {len(rows)}개 중 일치 {len(rows) - len(wrong)}개")
    for row in wrong:
        print(f"  불일치: r={row[0]} m={row[1]} D={row[2]} 이론 {row[3]} / 부호 바뀜 {row[4]}회")

def report_range():
    rows = range_rows()
    save_csv("4단계_수렴범위.csv", ["관성 m", "이론 상한 1+m", "실험: 수렴한 가장 큰 r",
                                   "실험: 처음 수렴 안 한 r", "그때 판정", "이론과 비교"],
             [row[:6] for row in rows])
    print("\n[실험 2 수렴 범위]")
    for row in rows:
        print(f"  m={row[0]:<4g} 이론 상한 {row[1]!s:<5} 실험 수렴 최대 {row[2]!s:<5} "
              f"처음 실패 {row[3]:g}({row[4]})  {row[5]}")
    return rows

def report_speed():
    rows = speed_rows()
    save_csv("4단계_속도.csv", ["학습률 r", "관성 m", "판별식 D", "근의 종류",
                               "이론: 큰 근의 절댓값", "실험: 도달한 항"], rows)
    best = min((row for row in rows if row[5] is not None), key=lambda row: row[5])
    theory_best = min(rows, key=lambda row: row[4])
    print(f"\n[실험 3 속도] r={SPEED_RATE}  관성 없음: a{rows[0][5]}  "
          f"가장 빠른 m={best[1]}: a{best[5]}  (이론상 큰 근이 가장 작은 m={theory_best[1]})")
    return rows

def main():
    # 코드 검증: m = 0 이면 1단계와 같아야 한다 (r = 0.1일 때 a20 = 0.0115)
    check = learn(0.1, 0, n=20)[20]
    print(f"[코드 검증] m=0, r=0.1의 20번째 항 = {check:.4f} (1단계 결과 0.0115)")
    report_shape()
    range_result = report_range()
    speed_result = report_speed()
    draw_paths()
    draw_range(range_result)
    draw_speed(speed_result)
    print("\n저장 완료: results/4단계_*.csv, figures/그림6~8")

main()
