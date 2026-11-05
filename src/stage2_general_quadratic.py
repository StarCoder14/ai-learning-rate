# -*- coding: utf-8 -*-
"""
진로주제 탐구 2단계 - 일반 이차함수로 확장

질문: 함수의 모양이 달라지면 안전한 학습률도 달라지는가?

이론 (docs/roadmap.md 2단계):
  f(x) = a(x - p)^2 + q 에서 x = t 의 접선의 기울기는 2a(t - p) (판별식으로 유도)
  학습 규칙  x(n+1) = x(n) - r * 2a(x(n) - p)
  b(n) = x(n) - p 로 놓으면  b(n+1) = (1 - 2ar) * b(n)  -> 공비 (1 - 2ar)인 등비수열
  수렴 조건  |1 - 2ar| < 1   ->   0 < r < 1/a
  예측 1: a가 클수록(뾰족할수록) 안전한 학습률의 상한 1/a가 작아진다.
  예측 2: r = 1/a 에서는 공비가 -1이라 진동한다.
  예측 3: r = 1/(2a) 에서는 공비가 0이라 한 번에 도달한다 (가장 빠르다).
  예측 4: p, q, 초기 거리를 바꿔도 수렴 여부는 달라지지 않는다.

검증 방법:
  공식 (1 - 2ar)을 쓰지 않고, 학습 규칙(기울기만큼 빼기)을 그대로 반복한 뒤
  꼭짓점까지의 거리가 줄었는지/그대로인지/커졌는지로 판정한다.
  그래야 이론과 실험이 서로 독립적인 근거가 된다.
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
A_VALUES = [0.5, 1, 2, 5]                  # 함수의 뾰족한 정도 a (조작변인 1)
RATES = [i / 20 for i in range(1, 45)]     # 학습률 r = 0.05 ~ 2.20, 0.05 간격 (조작변인 2)
P, Q = 0, 0                                # 꼭짓점 (p, q) (통제변인)
D0 = 1.0                                   # 시작점과 꼭짓점 사이의 거리 (통제변인)
N_TERMS = 200                              # 반복 횟수
TOL = 0.001                                # 도달 판정 기준: 꼭짓점까지 거리 < 0.001

# 통제변인 검증용 조건 (p, q, 시작 거리)
CONTROL_CASES = [(0, 0, 1.0), (3, -2, 1.0), (-5, 10, 1.0), (0, 0, 10.0)]
CONTROL_A = 2

# 판정별 색과 모양 (색만으로 구분하지 않도록 모양도 다르게)
STYLE = {"수렴": ("#2a78d6", "o"), "진동": ("#1baf7a", "D"), "발산": ("#eb6834", "X")}
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# ---------------- 학습 과정 ----------------
def slope(x, a, p):
    """f(x) = a(x-p)^2 + q 위의 점 x에서 접선의 기울기 (판별식으로 구한 2a(x-p))."""
    return 2 * a * (x - p)

def learn(a, r, p=P, d0=D0, n=N_TERMS):
    """시작점 x(0) = p + d0 에서 학습 규칙을 n번 반복한 x 값들을 반환한다."""
    xs = [p + d0]
    for _ in range(n):
        x = xs[-1]
        xs.append(x - r * slope(x, a, p))   # 기울기의 반대 방향으로 r만큼 이동
        if abs(xs[-1] - p) > d0 * 1e6:      # 너무 커지면 발산으로 보고 중단
            break
    return xs

def judge(xs, p=P):
    """꼭짓점까지의 거리가 처음보다 줄었으면 수렴, 같으면 진동, 커졌으면 발산."""
    d_first = abs(xs[0] - p)
    d_last = abs(xs[-1] - p)
    if math.isclose(d_last, d_first, rel_tol=1e-6):
        return "진동"
    return "수렴" if d_last < d_first else "발산"

def steps_to_reach(xs, p=P):
    """꼭짓점까지 거리가 TOL보다 작아진 첫 번째 항의 번호. 도달하지 못하면 None."""
    for i, x in enumerate(xs):
        if abs(x - p) < TOL:
            return i
    return None

# ---------------- CSV 저장 ----------------
def save_csv(name, header, rows):
    with open(os.path.join(RES_DIR, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------------- 실험 1 : a와 r에 따른 판정 ----------------
def run_grid():
    """모든 (a, r) 조합의 판정 결과를 {(a, r): (판정, 도달 항)} 으로 반환한다."""
    table = {}
    for a in A_VALUES:
        for r in RATES:
            xs = learn(a, r)
            table[(a, r)] = (judge(xs), steps_to_reach(xs))
    return table

# ---------------- 실험 2 : 이론 경계와 실험 경계 비교 ----------------
def boundary_row(a, table):
    """a 하나에 대해 이론 상한과 실험에서 찾은 경계를 한 줄로 정리한다."""
    ok = [r for r in RATES if table[(a, r)][0] == "수렴"]
    first_fail = min(r for r in RATES if table[(a, r)][0] != "수렴")
    reach = {r: table[(a, r)][1] for r in ok if table[(a, r)][1] is not None}
    fastest = min(reach, key=reach.get)
    match = max(ok) < first_fail and math.isclose(first_fail, 1 / a) \
        and table[(a, first_fail)][0] == "진동"
    return [a, round(1 / a, 4), max(ok), first_fail, table[(a, first_fail)][0],
            "일치" if match else "불일치", round(1 / (2 * a), 4), fastest, reach[fastest]]

# ---------------- 실험 3 : 통제변인 검증 ----------------
def control_row(p, q, d0):
    """꼭짓점과 시작 거리를 바꿔도 r별 판정이 기준 조건과 같은지 확인한다."""
    results = [judge(learn(CONTROL_A, r, p, d0), p) for r in RATES]
    base = [judge(learn(CONTROL_A, r)) for r in RATES]
    last_ok = max(r for r, j in zip(RATES, results) if j == "수렴")
    return [CONTROL_A, p, q, d0, results.count("수렴"), results.count("진동"),
            results.count("발산"), last_ok, "같음" if results == base else "다름"]

# ---------------- 그림 4 : 이론 상한과 실험 판정 ----------------
def draw_boundary_map(table):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for row, a in enumerate(A_VALUES):
        for label, (color, marker) in STYLE.items():
            rs = [r for r in RATES if table[(a, r)][0] == label]
            ax.scatter(rs, [row] * len(rs), s=38, color=color, marker=marker,
                       edgecolors="white", linewidths=0.8, zorder=3,
                       label=label if row == 0 else None)
        ax.plot([1 / a] * 2, [row - 0.38, row + 0.38], color=INK, linewidth=2, zorder=4)
        ax.text(1 / a, row + 0.42, f"이론 상한 1/a = {1 / a:g}", ha="center",
                va="bottom", fontsize=9, color=INK)
    ax.set_yticks(range(len(A_VALUES)), [f"a = {a:g}" for a in A_VALUES])
    ax.set_ylim(-0.6, len(A_VALUES) - 0.1)
    ax.set_xlabel("학습률 r")
    ax.set_title("함수의 뾰족한 정도(a)에 따른 안전한 학습률의 범위")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=3)
    save_fig(fig, "그림4_이차함수_수렴경계.png")

# ---------------- 그림 5 : 같은 학습률, 다른 함수 ----------------
def draw_same_rate(r=0.22):
    fig, axes = plt.subplots(1, len(A_VALUES), figsize=(13, 3.8))
    for ax, a in zip(axes, A_VALUES):
        xs = learn(a, r, n=8)
        label = judge(learn(a, r))
        color, marker = STYLE[label]
        grid = [i / 50 for i in range(-120, 121)]            # -2.4 ~ 2.4
        ax.plot(grid, [a * x * x for x in grid], color=GRID, linewidth=2)
        ax.plot(xs, [a * x * x for x in xs], color=color, marker=marker,
                markersize=7, linewidth=1.5, markeredgecolor="white")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.5, 16)
        ax.set_title(f"a = {a:g}  →  {label}  (공비 {1 - 2 * a * r:.1f})", fontsize=11)
        ax.set_xlabel("x")
        style_axes(ax)
    axes[0].set_ylabel("f(x) = a·x²")
    fig.suptitle(f"같은 학습률 r = {r}이라도 함수가 뾰족하면 학습에 실패한다", fontsize=13)
    save_fig(fig, "그림5_같은학습률_다른함수.png")

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

# ---------------- 실행 ----------------
def report_grid(table):
    """실험 1, 2의 결과를 CSV로 저장하고 화면에 요약한다."""
    save_csv("2단계_판정표.csv", ["a", "학습률 r", "이론 공비 1-2ar", "판정", "도달한 항"],
             [[a, r, round(1 - 2 * a * r, 4), j, f"a{s}" if s is not None else "-"]
              for (a, r), (j, s) in table.items()])
    rows = [boundary_row(a, table) for a in A_VALUES]
    save_csv("2단계_경계비교.csv",
             ["a", "이론 상한 1/a", "실험: 수렴한 가장 큰 r", "실험: 처음 수렴 안 한 r",
              "그때 판정", "이론과 비교", "이론 최적 1/(2a)", "실험: 가장 빠른 r", "도달한 항"], rows)
    print("\n[경계 비교]")
    for row in rows:
        print(f"  a={row[0]:<4g} 이론 상한 {row[1]:<6g} 실험 경계 {row[2]:g}~{row[3]:g} "
              f"({row[4]}) {row[5]} | 최적 이론 {row[6]:g} 실험 {row[7]:g} (a{row[8]})")

def report_control():
    """실험 3의 결과를 CSV로 저장하고 화면에 요약한다."""
    ctrl = [control_row(p, q, d0) for p, q, d0 in CONTROL_CASES]
    save_csv("2단계_통제변인검증.csv",
             ["a", "p", "q", "시작 거리", "수렴 개수", "진동 개수", "발산 개수",
              "수렴한 가장 큰 r", "기준 조건과 비교"], ctrl)
    print(f"\n[통제변인 검증] a={CONTROL_A}")
    for row in ctrl:
        print(f"  p={row[1]:<3} q={row[2]:<3} 시작거리={row[3]:<5} 수렴 {row[4]} / 진동 {row[5]} "
              f"/ 발산 {row[6]}  경계 {row[7]}  {row[8]}")

def main():
    # 코드 검증: a = 1이면 1단계와 같아야 한다 (r = 0.1일 때 a20 = 0.0115)
    check = learn(1, 0.1, n=20)[20]
    print(f"[코드 검증] a=1, r=0.1의 20번째 항 = {check:.4f} (1단계 결과 0.0115)")
    table = run_grid()
    report_grid(table)
    report_control()
    draw_boundary_map(table)
    draw_same_rate()
    print("\n저장 완료: results/2단계_*.csv, figures/그림4, 그림5")

main()
