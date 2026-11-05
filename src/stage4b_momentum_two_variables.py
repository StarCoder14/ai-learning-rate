# -*- coding: utf-8 -*-
"""
진로주제 탐구 4b단계 - 2변수 함수에 모멘텀 적용 (3단계와 4단계를 같은 함수로 잇기)

질문: 3단계에서 지그재그로 느려졌던 f(x, y) = x^2 + 10y^2 를 관성으로 빠르게 할 수 있는가?

학습 규칙 (축마다 4단계의 3항 점화식이 따로 돈다):
  x(n+1) = x(n) - r·2a·x(n) + m·(x(n) - x(n-1))   ->  t^2 - (1 + m - 2ar)t + m = 0
  y(n+1) = y(n) - r·2b·y(n) + m·(y(n) - y(n-1))   ->  t^2 - (1 + m - 2br)t + m = 0

이론 예측:
  예측 1 (안전 상한): 4단계 조건 0 < r < 1 + m 에서 r 자리에 a·r, b·r 이 들어간다.
                      두 축이 모두 만족해야 하므로  0 < r < (1 + m)/max(a, b)
  예측 2 (최적): 허근이면 근의 크기가 √m 이므로, 두 축이 모두 허근(또는 중근)이 되는 가장 작은 m이 가장 빠르다.
      x축 D <= 0 :  1 + m - 2ar <= 2√m      ->  (1 - √m)^2 <= 2ar
      y축 D <= 0 :  1 + m - 2br >= -2√m     ->  2br <= (1 + √m)^2
      두 조건을 함께 만족하는 r이 있으려면  √b(1 - √m) <= √a(1 + √m)
      ->  √m >= (√b - √a)/(√b + √a),   이때  r = 2/(√a + √b)^2
      b = 10 이면 속도를 정하는 수가 0.818(관성 없음, 3단계) -> 0.520 으로 줄어든다.
  예측 3 (작은 학습률 r = 0.03): 느린 x축만 보면 4단계와 같아서 m = (1 - √(2ar))^2 근처가 가장 빠르다.
  예측 4 (코드 검증): m = 0 이면 3단계와 같다 (b = 10, r = 1/11 에서 35번째 항).

검증 방법: 특성방정식을 쓰지 않고 학습 규칙을 그대로 반복해서 판정한다.

실험에서 발견한 점: 이론 최적(두 축 모두 중근)은 '오래 반복했을 때 줄어드는 비율'이 가장 작지만,
중근이면 x(n)이 n × (0.52)^n 꼴이 되어 앞에 곱해진 n 때문에 초반에 손해를 본다.
그래서 0.001 도달 기준으로는 m이 조금 더 큰 조합(m ≈ 0.3)이 더 빨랐다. (4b단계_도달기준비교.csv)
"""

import os
import csv
import math
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
A, B = 1, 10                               # 두 축의 뾰족한 정도 (통제변인, 3단계의 b = 10)
X0, Y0 = 1.0, 1.0                          # 시작점, 처음에는 움직인 적이 없다 (통제변인)
N_TERMS = 3000                             # 최대 반복 횟수
TOL = 0.001                                # 도달 판정 기준

RANGE_MS = [0, 0.3, 0.6, 0.9]                              # 실험 1 : 관성 m
RANGE_RATES = [i / 200 for i in range(1, 51)]              # 실험 1 : r = 0.005 ~ 0.250
SEARCH_MS = [i / 100 for i in range(0, 96)]                # 실험 2 : m = 0.00 ~ 0.95
SEARCH_N = 400                                             # 실험 2 : 훑을 때 최대 반복
SMALL_RATE = 0.03                                          # 실험 3 : 3단계와 같은 작은 학습률
SMALL_MS = [i / 20 for i in range(0, 20)]                  # 실험 3 : m = 0 ~ 0.95

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"

# ---------------- 학습 과정 ----------------
def learn(r, m, n=N_TERMS):
    """두 좌표에 모멘텀 학습 규칙을 동시에 적용한 (x, y) 점들을 반환한다."""
    points, prev = [(X0, Y0)], (X0, Y0)
    for _ in range(n):
        x, y = points[-1]
        new_x = x - r * (2 * A * x) + m * (x - prev[0])   # 기울기만큼 빼고 관성만큼 더 간다
        new_y = y - r * (2 * B * y) + m * (y - prev[1])
        prev = (x, y)
        points.append((new_x, new_y))
        size = max(abs(new_x), abs(new_y))
        if size > 1e6 or size < 1e-9:                      # 충분히 커지거나 작아지면 중단
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
    """크기 < TOL 이 된 뒤 다시는 벗어나지 않는 첫 항. 출렁이며 다시 커질 수 있어서 마지막으로 벗어난 곳을 본다."""
    if size_of(points[-1]) >= TOL:
        return None
    last_out = max((i for i, p in enumerate(points) if size_of(p) >= TOL), default=-1)
    return last_out + 1

# ---------------- 이론값 계산 ----------------
def best_theory():
    """두 축이 모두 허근(중근)이 되는 가장 작은 m과 그때의 r."""
    root_m = (math.sqrt(B) - math.sqrt(A)) / (math.sqrt(B) + math.sqrt(A))
    r = 2 / (math.sqrt(A) + math.sqrt(B)) ** 2
    return r, root_m ** 2, root_m

def biggest_root_size(r, m, k):
    """뾰족한 정도 k인 축의 특성방정식에서 근의 절댓값 중 큰 값 (4단계와 같은 계산)."""
    c = 1 + m - 2 * k * r
    d = c * c - 4 * m
    return math.sqrt(m) if d < 0 else (abs(c) + math.sqrt(d)) / 2

# ---------------- CSV 저장 ----------------
def save_csv(name, header, rows):
    with open(os.path.join(RES_DIR, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------------- 실험 1 : 안전 상한 ----------------
def range_row(m):
    """m 하나에 대해 실제 수렴 경계를 이론 상한 (1 + m)/max(a, b)와 비교한다."""
    result = {r: judge(learn(r, m)) for r in RANGE_RATES}
    ok = [r for r in RANGE_RATES if result[r] == "수렴"]
    fail = min(r for r in RANGE_RATES if result[r] != "수렴")
    upper = (1 + m) / max(A, B)
    match = max(ok) < upper and math.isclose(fail, upper) and result[fail] == "진동"
    return [A, B, m, round(upper, 4), max(ok), fail, result[fail], "일치" if match else "불일치"]

# ---------------- 실험 2 : 가장 빠른 (r, m) 찾기 ----------------
def search_best():
    """m은 0.01 간격, r은 0.001 간격으로 훑어 도달 항이 가장 작은 조합들을 찾는다."""
    best, found = None, []
    for m in SEARCH_MS:
        for i in range(1, round((1 + m) / B * 1000)):
            r = i / 1000
            s = steps_to_reach(learn(r, m, n=SEARCH_N))
            if s is None or (best is not None and s > best):
                continue
            if best is None or s < best:
                best, found = s, []
            found.append((r, m))
    return best, found

def best_rows():
    """관성 없음(3단계 최선) / 이론 최적 / 격자 탐색 최선을 한 표로 정리한다."""
    r_t, m_t, root_t = best_theory()
    best, found = search_best()
    rows = [["관성 없음 (3단계 최선)", round(1 / (A + B), 4), 0, round((B - A) / (B + A), 4),
             steps_to_reach(learn(1 / (A + B), 0))],
            ["이론 최적", round(r_t, 4), round(m_t, 4), round(root_t, 4),
             steps_to_reach(learn(r_t, m_t))]]
    for r, m in found:
        size = max(biggest_root_size(r, m, A), biggest_root_size(r, m, B))
        rows.append(["격자 탐색 최선", r, m, round(size, 4), best])
    return rows

# ---------------- 실험 3 : 작은 학습률에서 관성의 효과 ----------------
def small_rows():
    rows = []
    for m in SMALL_MS:
        size = max(biggest_root_size(SMALL_RATE, m, A), biggest_root_size(SMALL_RATE, m, B))
        rows.append([SMALL_RATE, m, round(size, 4), steps_to_reach(learn(SMALL_RATE, m))])
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

def save_fig(fig, name, top=1.0):
    fig.tight_layout(rect=(0, 0, 1, top))       # top < 1 이면 전체 제목 자리를 남긴다
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close(fig)

# ---------------- 그림 11 : 관성 없음 vs 관성 있음 경로 ----------------
def draw_one_path(ax, r, m, title, color, gx, gy):
    points = learn(r, m)
    ax.contour(gx, gy, A * gx ** 2 + B * gy ** 2, levels=[0.05, 0.2, 0.5, 1, 2, 4],
               colors=GRID, linewidths=1.2)                 # 높이가 같은 점을 이은 선
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=1.5,
            marker="o", markersize=5, markeredgecolor="white", zorder=3)
    ax.plot(0, 0, marker="*", markersize=14, color=INK, zorder=4)
    ax.set_title(f"{title}\nr = {r:.3f}, m = {m:.2f} → {steps_to_reach(points)}번째 항에서 도달",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    style_axes(ax)

def draw_paths():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    grid = np.linspace(-1.45, 1.45, 300)
    gx, gy = np.meshgrid(grid, grid)
    r_t, m_t, _ = best_theory()
    draw_one_path(axes[0], 1 / (A + B), 0, "관성 없음 (3단계의 최선)", BLUE, gx, gy)
    draw_one_path(axes[1], r_t, m_t, "관성 있음 (판별식으로 구한 최적)", ORANGE, gx, gy)
    axes[0].set_ylabel("y")
    fig.suptitle(f"f(x, y) = x² + {B}y²에서 관성을 주면 지그재그가 남아도 훨씬 빨리 도착한다 (★ = 최솟값)", fontsize=13)
    save_fig(fig, "그림11_2변수_모멘텀_경로.png", top=0.93)

# ---------------- 그림 12 : 작은 학습률에서 관성에 따른 속도 ----------------
def draw_small(rows):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    best_m = (1 - math.sqrt(2 * A * SMALL_RATE)) ** 2       # 느린 x축이 중근이 되는 m
    ax.axvspan(best_m, 1, color=SHADE, zorder=0)
    ms = [row[1] for row in rows if row[3] is not None]
    steps = [row[3] for row in rows if row[3] is not None]
    ax.plot(ms, steps, color=ORANGE, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", zorder=3)
    ax.axvline(best_m, color=INK, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(best_m, max(steps), f" x축 중근 (D = 0)\n m = {best_m:.3f}", va="top", fontsize=10)
    ax.text(best_m / 2, max(steps) * 0.9, "x축 두 실근", ha="center", color=INK_SOFT)
    ax.text((best_m + 1) / 2, max(steps) * 0.75, "x축 허근", ha="center", color=INK_SOFT)
    ax.set(xlim=(-0.02, 1), ylim=(0, max(steps) * 1.08), xlabel="관성 m",
           ylabel=f"크기 < {TOL}에 도달한 항의 번호",
           title=f"f(x, y) = x² + {B}y², 작은 학습률 r = {SMALL_RATE}에서 관성에 따른 학습 속도")
    style_axes(ax)
    save_fig(fig, "그림12_2변수_모멘텀_작은학습률.png")

# ---------------- 실행 ----------------
def report_range():
    rows = [range_row(m) for m in RANGE_MS]
    save_csv("4b단계_안전상한.csv", ["a", "b", "관성 m", "이론 상한 (1+m)/max(a,b)", "실험: 수렴한 가장 큰 r",
                                   "실험: 처음 수렴 안 한 r", "그때 판정", "이론과 비교"], rows)
    print("\n[실험 1 안전 상한]")
    for row in rows:
        print(f"  m={row[2]:<4g} 이론 {row[3]:<6g} 실험 {row[4]:g}~{row[5]:g}({row[6]})  {row[7]}")

def report_best():
    rows = best_rows()
    save_csv("4b단계_최적비교.csv", ["구분", "학습률 r", "관성 m", "이론: 속도를 정하는 수 (큰 근의 크기)",
                                   "실험: 도달한 항"], rows)
    print("\n[실험 2 가장 빠른 조합]")
    for row in rows:
        print(f"  {row[0]:<16} r={row[1]:<7g} m={row[2]:<6g} 큰 근 {row[3]:<7g} → a{row[4]}")

def steps_to_tol(points, tol):
    """크기 < tol 이 된 뒤 다시는 벗어나지 않는 첫 항 (learn이 1e-9에서 멈추므로 tol >= 1e-8 까지만)."""
    out = [i for i, p in enumerate(points) if size_of(p) >= tol]
    return out[-1] + 1 if out else 0

def report_tolerance():
    """이론 최적과 격자 최선의 차이가 도달 기준을 엄격하게 해도 유지되는지 확인한다."""
    r_t, m_t, _ = best_theory()
    cases = [("관성 없음", 1 / (A + B), 0), ("이론 최적", r_t, m_t), ("격자 최선 예", 0.113, 0.3)]
    rows = [[name, round(r, 4), round(m, 4)] + [steps_to_tol(learn(r, m), t) for t in (1e-3, 1e-5, 1e-8)]
            for name, r, m in cases]
    save_csv("4b단계_도달기준비교.csv", ["구분", "학습률 r", "관성 m", "0.001 도달 항", "0.00001 도달 항",
                                     "0.00000001 도달 항"], rows)
    print("\n[추가 확인 도달 기준을 엄격하게]")
    for row in rows:
        print(f"  {row[0]:<10} 1e-3: a{row[3]:<4} 1e-5: a{row[4]:<4} 1e-8: a{row[5]}")

def report_small():
    rows = small_rows()
    save_csv("4b단계_작은학습률.csv", ["학습률 r", "관성 m", "이론: 큰 근의 크기", "실험: 도달한 항"], rows)
    best = min((row for row in rows if row[3] is not None), key=lambda row: row[3])
    print(f"\n[실험 3 작은 학습률 r={SMALL_RATE}] 관성 없음 a{rows[0][3]} → 가장 빠른 m={best[1]}: a{best[3]}"
          f"  (이론 중근 m={(1 - math.sqrt(2 * A * SMALL_RATE)) ** 2:.3f})")
    return rows

def main():
    # 코드 검증: m = 0 이면 3단계와 같아야 한다 (b = 10, r = 1/11 에서 35번째 항)
    print(f"[코드 검증] m=0, r=1/11 → a{steps_to_reach(learn(1 / 11, 0))} (3단계 결과 a35)")
    report_range()
    report_best()
    report_tolerance()
    rows = report_small()
    draw_paths()
    draw_small(rows)
    print("\n저장 완료: results/4b단계_*.csv, figures/그림11, 그림12")

main()
