# -*- coding: utf-8 -*-
"""
진로주제 탐구 5단계 - 로지스틱 사상과 혼돈

질문: 점화식의 모양을 조금 바꾸면(곱하기 하나 추가) 수렴 말고 무엇이 나타나는가?

점화식: a(n+1) = r · a(n) · (1 - a(n)),   0 < r <= 4,  0 < a(0) < 1
  (원래 생물 개체수 모형: 개체수가 많아지면 먹이가 부족해 다음 세대가 줄어든다)

이론 (미분 없이, 판별식과 근과 계수의 관계만 사용):
  [고정점] a = r·a(1 - a)  ->  a = 0  또는  a = 1 - 1/r
  [고정점 근처의 공비] 포물선 y = r·x(1 - x) 위 x = t 에서 접선의 기울기는 판별식 D = 0 으로 r(1 - 2t).
      고정점 근처에서는 '고정점까지의 거리'가 이 기울기를 공비로 하는 등비수열처럼 움직인다.
      a = 0 에서 기울기 r          ->  0 < r < 1 이면 0으로 수렴
      a = 1 - 1/r 에서 기울기 2 - r  ->  |2 - r| < 1,  즉 1 < r < 3 이면 1 - 1/r 로 수렴
      (1 < r < 2 는 공비가 양수라 한쪽에서, 2 < r < 3 은 음수라 번갈아 가며 다가간다)
  [주기 2의 탄생] 두 번 가면 제자리인 점 g(g(a)) = a 에서 고정점 두 개를 인수로 나누면
      r^2·a^2 - r(r + 1)·a + (r + 1) = 0,   판별식 D = r^2 (r + 1)(r - 3)
      -> r > 3 일 때만 서로 다른 두 실근 = 번갈아 오가는 두 값(주기 2)이 생긴다.
  [주기 2가 안정한 범위] 두 번 가는 동안의 공비 = 두 점에서의 기울기의 곱
      r(1 - 2p)·r(1 - 2q) = r^2 (1 - 2(p + q) + 4pq),  근과 계수의 관계  p + q = (r + 1)/r,  pq = (r + 1)/r^2
      = -r^2 + 2r + 4   ->  |-r^2 + 2r + 4| < 1   ->   3 < r < 1 + √6 ≈ 3.449
  [그 뒤] 주기 4, 8, ... 로 계속 갈라지다가 r ≈ 3.57 부근부터 규칙이 보이지 않는 혼돈이 나타난다고 알려져 있다.
      (이 값은 유도하지 않았고, 실험으로 관찰만 한다)

검증 방법: 공식을 쓰지 않고 점화식을 그대로 반복해서, 마지막 값들이 몇 개의 값을 도는지 센다.
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
A0 = 0.2                                           # 초기값 (통제변인)
FIXED_RATES = [i / 10 for i in range(1, 30) if i != 10]    # 실험 1 : r = 0.1 ~ 2.9 (경계 1.0 제외)
SCAN_RATES = [2.5 + i / 1000 for i in range(0, 1501)]      # 실험 2 : r = 2.500 ~ 4.000, 0.001 간격
BURN_IN = 10000                                    # 처음 값에 영향받지 않도록 버리는 항 수
KEEP = 128                                         # 그 뒤에 관찰하는 항 수
GAP = 1e-3                                         # 이보다 가까운 값은 같은 값으로 본다
SENSE_RATES = [2.8, 3.2, 3.5, 3.9]                 # 실험 3 : 초기값 민감성
NUDGE = 1e-7                                       # 초기값을 이만큼만 바꾼다

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# ---------------- 점화식 ----------------
def step(a, r):
    """로지스틱 점화식 한 번."""
    return r * a * (1 - a)

def sequence(r, a0=A0, n=100):
    """a(0) 부터 a(n) 까지의 값들."""
    values = [a0]
    for _ in range(n):
        values.append(step(values[-1], r))
    return values

def count_values(r):
    """충분히 반복한 뒤 KEEP개의 값이 서로 다른 값 몇 개를 도는지 센다 (가까운 값끼리 묶어서)."""
    a = A0
    for _ in range(BURN_IN):
        a = step(a, r)
    seen = sorted(sequence(r, a, KEEP)[1:])
    return 1 + sum(1 for x, y in zip(seen, seen[1:]) if y - x > GAP)

def period_label(count):
    """값의 개수를 주기 이름으로 바꾼다. 32개보다 많으면 규칙이 보이지 않는 것으로 본다."""
    return f"주기 {count}" if count <= 32 else "혼돈(32개 초과)"

# ---------------- 이론값 ----------------
def theory_fixed(r):
    """수렴해야 할 고정점과, 그 근처에서의 공비(접선의 기울기)."""
    return (0.0, r) if r < 1 else (1 - 1 / r, 2 - r)

def theory_two_cycle(r):
    """r^2 a^2 - r(r+1) a + (r+1) = 0 의 두 근 (주기 2로 오가는 두 값)."""
    d = r * r * (r + 1) * (r - 3)
    return ((r * (r + 1) - math.sqrt(d)) / (2 * r * r), (r * (r + 1) + math.sqrt(d)) / (2 * r * r))

# ---------------- CSV 저장 ----------------
def save_csv(name, header, rows):
    with open(os.path.join(RES_DIR, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------------- 실험 1 : 고정점과 공비 ----------------
def measured_ratio(values, target):
    """고정점까지의 거리가 1e-4 ~ 1e-7 사이일 때 '다음 거리 ÷ 지금 거리'의 평균."""
    ratios = []
    for e1, e2 in zip(values, values[1:]):
        d1, d2 = e1 - target, e2 - target
        if 1e-7 < abs(d1) < 1e-4:
            ratios.append(d2 / d1)
    return sum(ratios) / len(ratios) if ratios else None

def fixed_row(r):
    """마지막 값이 이론 고정점과 같은지, 다가가는 공비가 이론 기울기와 같은지 확인한다."""
    target, slope = theory_fixed(r)
    values = sequence(r, n=3000)
    ratio = measured_ratio(values, target)
    reached = abs(values[-1] - target) < 1e-9
    exact = next((i for i, v in enumerate(values[:10]) if v == target), None)   # 우연히 고정점에 딱 떨어진 경우
    if ratio is None and exact is not None:
        return [r, round(target, 6), round(slope, 4), round(values[-1], 6),
                f"측정 불가 (a{exact}이 고정점과 정확히 같음)", "일치 (공비는 확인 못 함)"]
    ratio_ok = ratio is None and abs(slope) < 0.05 or ratio is not None and abs(ratio - slope) < 0.01
    return [r, round(target, 6), round(slope, 4), round(values[-1], 6),
            "-" if ratio is None else round(ratio, 4), "일치" if reached and ratio_ok else "불일치"]

# ---------------- 실험 2 : 주기 판정과 분기점 ----------------
def scan():
    """r을 0.001 간격으로 훑어 각 r에서 돌고 있는 값의 개수를 센다."""
    return [(r, count_values(r)) for r in SCAN_RATES]

def first_rate(scanned, test):
    """조건을 처음 만족하는 r."""
    return next((round(r, 3) for r, c in scanned if test(c)), None)

def split_rows(scanned):
    """갈라지는 지점을 이론값과 비교한다."""
    at_three = dict((round(r, 3), c) for r, c in scanned)[3.0]
    return [["주기 1 → 주기 2", 3.0, first_rate(scanned, lambda c: c >= 2),
             f"판별식 r²(r+1)(r-3) > 0. r = 3.000에서는 값 {at_three}개로 셌다 (경계에서 수렴이 매우 느림)"],
            ["주기 2 → 주기 4", round(1 + math.sqrt(6), 4), first_rate(scanned, lambda c: c >= 4),
             "|-r²+2r+4| < 1 (근과 계수의 관계)"],
            ["주기 4 → 주기 8", "유도 안 함", first_rate(scanned, lambda c: c >= 8), "실험 관찰만"],
            ["혼돈 시작", "약 3.57 (알려진 값)", first_rate(scanned, lambda c: c > 32), "실험 관찰만"]]

def two_cycle_check(rates=(3.1, 3.2, 3.3, 3.4)):
    """주기 2 구간에서 실제로 오가는 두 값이 이차방정식의 두 근과 같은지 확인한다."""
    rows = []
    for r in rates:
        tail = sequence(r, n=BURN_IN)[-2:]
        low, high = theory_two_cycle(r)
        ok = abs(min(tail) - low) < 1e-6 and abs(max(tail) - high) < 1e-6
        rows.append([r, round(low, 6), round(high, 6), round(min(tail), 6), round(max(tail), 6),
                     "일치" if ok else "불일치"])
    return rows

# ---------------- 실험 3 : 초기값 민감성 ----------------
def sensitivity_row(r):
    """초기값을 0.0000001만 바꿨을 때 두 수열의 차이가 0.1을 넘는 첫 항."""
    s1, s2 = sequence(r, A0, 300), sequence(r, A0 + NUDGE, 300)
    first = next((n for n, (x, y) in enumerate(zip(s1, s2)) if abs(x - y) > 0.1), None)
    final_gap = abs(s1[-1] - s2[-1])
    return [r, NUDGE, "끝까지 0.1 이하" if first is None else first, f"{final_gap:.2e}"]

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

# ---------------- 그림 13 : 거미줄 그림 4장 ----------------
def draw_cobweb(ax, r, n=80):
    """포물선 y = r·x(1-x), 직선 y = x, 그리고 수열이 지나간 계단을 그린다."""
    xs = [i / 200 for i in range(201)]
    ax.plot(xs, [step(x, r) for x in xs], color=INK, linewidth=1.5)          # 점화식 그 자체
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linestyle="--", linewidth=0.8)   # y = x
    values = sequence(r, n=n)
    px, py = [values[0]], [0.0]
    for a, b in zip(values, values[1:]):
        px += [a, b]
        py += [b, b]
    ax.plot(px, py, color=BLUE, linewidth=0.8, alpha=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="a(n)", aspect="equal")
    label = period_label(count_values(r))
    ax.set_title(f"r = {r}  →  {label}", fontsize=11)
    style_axes(ax)

def draw_cobwebs(rates=(2.5, 3.2, 3.5, 3.9)):
    fig, axes = plt.subplots(1, len(rates), figsize=(14, 4.9))
    for ax, r in zip(axes, rates):
        draw_cobweb(ax, r)
    axes[0].set_ylabel("a(n+1)")
    fig.suptitle("로지스틱 점화식 a(n+1) = r·a(n)(1 - a(n))의 거미줄 그림 (a(0) = 0.2, 80번 반복)", fontsize=13)
    save_fig(fig, "그림13_로지스틱_거미줄.png", top=0.92)

# ---------------- 그림 14 : 분기 다이어그램 ----------------
def draw_theory_lines(ax):
    """고정점 1 - 1/r (r < 3)과 주기 2의 두 값 (3 < r < 1+√6)을 이론선으로 겹쳐 그린다."""
    r1 = [2.5 + i / 1000 for i in range(0, 501)]
    ax.plot(r1, [1 - 1 / r for r in r1], color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), zorder=3, label="이론: 고정점 1 - 1/r")
    r2 = [3.0 + i / 1000 for i in range(1, 450)]
    cycles = [theory_two_cycle(r) for r in r2]
    ax.plot(r2, [c[0] for c in cycles], color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3, label="이론: 주기 2의 두 값")
    ax.plot(r2, [c[1] for c in cycles], color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    for x, text in [(3.0, "r = 3"), (1 + math.sqrt(6), "r = 1+√6")]:
        ax.axvline(x, color=INK_SOFT, linestyle="--", linewidth=1, zorder=2)
        ax.text(x + 0.008, 1.01, text, ha="left", va="bottom", fontsize=9, color=INK)

def draw_bifurcation():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    rs, vals = [], []
    for r in [2.5 + i / 2000 for i in range(0, 3001)]:           # r을 0.0005 간격으로
        a = A0
        for _ in range(1000):
            a = step(a, r)
        tail = sequence(r, a, 100)[1:]
        rs += [r] * len(tail)
        vals += tail
    ax.scatter(rs, vals, s=0.3, color=BLUE, zorder=2, label="실험: 1000번 반복 뒤 100개의 값")
    draw_theory_lines(ax)
    ax.set(xlim=(2.5, 4), ylim=(0, 1.08), xlabel="r", ylabel="충분히 반복한 뒤의 a(n) 값",
           title="분기 다이어그램: r이 커지면 수렴 → 주기 2 → 주기 4 → ... → 혼돈")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=3, markerscale=25)
    save_fig(fig, "그림14_로지스틱_분기다이어그램.png")

# ---------------- 그림 15 : 초기값 민감성 ----------------
def draw_sensitivity(rates=(2.8, 3.9), n=60):
    fig, axes = plt.subplots(len(rates), 1, figsize=(10, 5.6), sharex=True)
    for ax, r in zip(axes, rates):
        s1, s2 = sequence(r, A0, n), sequence(r, A0 + NUDGE, n)
        ax.plot(range(n + 1), s1, color=BLUE, linewidth=1.8, label=f"a(0) = {A0}")
        ax.plot(range(n + 1), s2, color=ORANGE, linewidth=1.4, linestyle="--",
                label=f"a(0) = {A0 + NUDGE:.7f}")
        ax.set(ylim=(0, 1.05), ylabel="a(n)")
        ax.set_title(f"r = {r}  ({period_label(count_values(r))})", fontsize=11, loc="left")
        style_axes(ax)
    axes[-1].set_xlabel("항의 번호 n")
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    fig.suptitle("초기값을 0.0000001만 바꿨을 때: 혼돈에서는 결과가 완전히 달라진다", fontsize=13)
    save_fig(fig, "그림15_로지스틱_초기값민감성.png", top=0.93)

# ---------------- 실행 ----------------
def report_fixed():
    rows = [fixed_row(r) for r in FIXED_RATES]
    save_csv("5단계_고정점수렴.csv", ["r", "이론 고정점", "이론 공비(접선 기울기)", "실험: 3000번째 값",
                                   "실험: 거리의 비", "이론과 비교"], rows)
    wrong = [row for row in rows if row[5] != "일치"]
    partial = [row for row in wrong if row[5].startswith("일치")]
    print(f"\n[실험 1 고정점과 공비] {len(rows)}개 중 일치 {len(rows) - len(wrong)}개, "
          f"고정점만 확인 {len(partial)}개, 불일치 {len(wrong) - len(partial)}개")
    for row in rows[::4] + wrong:
        print(f"  r={row[0]:<4} 고정점 {row[1]:<9} 이론 공비 {row[2]:<7} 실험 {row[4]!s:<8} {row[5]}")

def report_scan():
    scanned = scan()
    save_csv("5단계_주기판정.csv", ["r", "돌고 있는 값의 개수", "판정"],
             [[round(r, 3), c, period_label(c)] for r, c in scanned])
    rows = split_rows(scanned)
    save_csv("5단계_분기점비교.csv", ["변화", "이론값", "실험: 처음 관찰된 r (0.001 간격)", "근거"], rows)
    print("\n[실험 2 갈라지는 지점]")
    for row in rows:
        print(f"  {row[0]:<10} 이론 {row[1]!s:<14} 실험 {row[2]}")
    cycle = two_cycle_check()
    save_csv("5단계_주기2확인.csv", ["r", "이론 작은 값", "이론 큰 값", "실험 작은 값", "실험 큰 값", "비교"], cycle)
    print("  주기 2의 두 값 (이차방정식의 근과 비교): " + ", ".join(f"r={row[0]} {row[5]}" for row in cycle))

def report_sensitivity():
    rows = [sensitivity_row(r) for r in SENSE_RATES]
    save_csv("5단계_초기값민감성.csv", ["r", "초기값 차이", "차이가 0.1을 넘은 첫 항", "300번째 항의 차이"], rows)
    print("\n[실험 3 초기값 민감성]")
    for row in rows:
        print(f"  r={row[0]:<4} 0.1을 넘은 첫 항: {row[2]!s:<14} 300번째 차이 {row[3]}")

def main():
    # 코드 검증: r = 2 이면 고정점 0.5 로 수렴해야 한다
    print(f"[코드 검증] r=2의 100번째 항 = {sequence(2, n=100)[-1]:.6f} (이론 0.5)")
    report_fixed()
    report_scan()
    report_sensitivity()
    draw_cobwebs()
    draw_bifurcation()
    draw_sensitivity()
    print("\n저장 완료: results/5단계_*.csv, figures/그림13~15")

main()
