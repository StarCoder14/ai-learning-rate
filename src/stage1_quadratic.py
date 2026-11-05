# -*- coding: utf-8 -*-
"""
진로주제 탐구 - 점화식과 등비수열의 수렴 조건을 이용한 인공지능 학습률 분석

이론:
  대상 함수 f(x) = x^2 의 최솟값(x=0)을 경사하강법으로 찾는다.
  x = t 에서의 접선의 기울기는 2t 이므로 학습 규칙은
      a(n+1) = a(n) - r * 2*a(n) = (1 - 2r) * a(n)
  즉 공비가 (1 - 2r) 인 등비수열이다.
  등비수열이 0으로 수렴할 조건은 |공비| < 1 이므로
      |1 - 2r| < 1   ->   0 < r < 1
"""

import csv
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

# ---------------- 설정값 ----------------
A0 = 1.0                                  # 초기값 (통제변인)
N_TERMS = 20                              # 계산할 항의 개수
RATES = [0.1, 0.25, 0.5, 0.9, 1.0, 1.2]   # 학습률 (조작변인)
TOL = 0.001                               # 수렴 판정 기준

# ---------------- 수열 계산 ----------------
def make_sequence(r, a0=A0, n=N_TERMS):
    """공비 (1-2r)인 등비수열의 항을 리스트로 반환한다."""
    ratio = 1 - 2 * r
    seq = [a0]
    for _ in range(n):
        seq.append(seq[-1] * ratio)
    return seq

def judge(r, seq):
    """수렴 / 진동 / 발산 을 판정하고, 수렴이면 몇 번째 항에서인지 반환한다."""
    ratio = abs(1 - 2 * r)
    if ratio < 1:
        for i, v in enumerate(seq):
            if abs(v) < TOL:
                return "수렴", i
        return "수렴", None
    elif ratio == 1:
        return "진동", None
    else:
        return "발산", None

# ---------------- 결과 표 ----------------
rows = []
print(f"{'학습률 r':>8} {'공비 1-2r':>10} {'|공비|':>8} {'판정':>6} {'수렴한 항':>10} {'20번째 항':>14}")
print("-" * 64)
for r in RATES:
    seq = make_sequence(r)
    ratio = 1 - 2 * r
    result, step = judge(r, seq)
    step_txt = f"a{step}" if step is not None else "-"
    print(f"{r:>8} {ratio:>10.2f} {abs(ratio):>8.2f} {result:>6} {step_txt:>10} {seq[N_TERMS]:>14.4f}")
    rows.append([r, round(ratio, 4), round(abs(ratio), 4), result, step_txt, seq[N_TERMS]])

with open("결과표.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["학습률 r", "공비 1-2r", "|공비|", "판정", "수렴한 항", f"a{N_TERMS} 값"])
    w.writerows(rows)

# ---------------- 그림 1 : 항의 변화 ----------------
fig, ax = plt.subplots(figsize=(8, 5))
for r in RATES:
    seq = make_sequence(r, n=12)
    ax.plot(range(len(seq)), seq, marker="o", markersize=4, label=f"r = {r}")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylim(-3, 3)
ax.set_xlabel("항의 번호 n")
ax.set_ylabel("a(n)")
ax.set_title("학습률에 따른 수열의 변화")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("그림1_항의변화.png", dpi=200)
plt.close(fig)

# ---------------- 그림 2 : 거미줄 그림 ----------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, r in zip(axes.flat, RATES):
    ratio = 1 - 2 * r
    lim = 1.6
    xs = [-lim, lim]
    ax.plot(xs, xs, color="gray", linestyle="--", linewidth=0.8)      # y = x
    ax.plot(xs, [ratio * x for x in xs], color="purple", linewidth=1.5)  # y = (1-2r)x

    x = A0
    px, py = [x], [0.0]
    for _ in range(12):
        y = ratio * x
        px += [x, y]
        py += [y, y]
        x = y
        if abs(x) > lim:
            break
    color = "green" if abs(ratio) < 1 else ("orange" if abs(ratio) == 1 else "red")
    ax.plot(px, py, color=color, linewidth=1.2)

    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("a(n)")
    ax.set_ylabel("a(n+1)")
    ax.set_title(f"r = {r}  (공비 {ratio:.1f})")
fig.suptitle("거미줄 그림으로 본 수렴과 발산", fontsize=14)
fig.tight_layout()
fig.savefig("그림2_거미줄그림.png", dpi=200)
plt.close(fig)

# ---------------- 그림 3 : 이론적 수렴 조건 ----------------
fig, ax = plt.subplots(figsize=(8, 5))
rs = [i / 200 for i in range(0, 301)]
ax.plot(rs, [abs(1 - 2 * r) for r in rs], color="purple", linewidth=1.8, label="|1 - 2r|")
ax.axhline(1, color="red", linestyle="--", linewidth=1, label="수렴 한계선")
ax.axvspan(0, 1, color="green", alpha=0.12)
ax.text(0.5, 1.45, "수렴 구간  0 < r < 1", ha="center", color="green")
for r in RATES:
    ax.plot(r, abs(1 - 2 * r), "o", color="black", markersize=5)
ax.set_xlabel("학습률 r")
ax.set_ylabel("|공비|")
ax.set_title("학습률에 따른 공비의 절댓값과 수렴 조건")
ax.set_ylim(0, 2)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("그림3_수렴조건.png", dpi=200)
plt.close(fig)

print("\n저장 완료: 결과표.csv, 그림1~3.png")
