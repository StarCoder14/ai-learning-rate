# -*- coding: utf-8 -*-
"""
발표 슬라이드를 HTML 파일 하나로 묶음.

slides/ 안의 슬라이드 조각을 deck.json 순서대로 이어 붙이고,
그림은 figures/ 폴더의 파일을 글자(base64)로 바꿔 파일 안에 직접 넣음.
그래서 인터넷이 없어도, 파일 하나만 USB에 담아 가도 그대로 열림.

실행: python 발표/build_slides.py   ->   발표/발표슬라이드.html 생성
"""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "..", "figures")
OUTPUT = os.path.join(HERE, "발표슬라이드.html")

# 슬라이드 안의 /_blob/ 주소가 어떤 그림인지 (claude.ai에 올릴 때 받은 주소)
BLOB_TO_FIGURE = {
    "5171c22f26414db9873776a2dae11703": "그림4_이차함수_수렴경계.png",
    "61252280f9911b4dd60ab7c8a81572a1": "그림9_2변수_지그재그.png",
    "33b21c74aecbfc7ac1dc426383ca282e": "그림6_모멘텀_경로모양.png",
    "ae245b5e44d988f52f99ec6b4bed81ca": "그림14_로지스틱_분기다이어그램.png",
    "91d42a73f36c47950fd0266795cff0d0": "애니1_거미줄그림.gif",
    "343dcc79afa1294ca0d8a2a969bc3076": "애니2_관성경주.gif",
}

def read_text(path):
    with open(path, encoding="utf-8") as file:
        return file.read()

def figure_as_text(name):
    """그림 파일을 브라우저가 바로 읽는 data 주소로 바꿈."""
    kind = "gif" if name.endswith(".gif") else "png"
    with open(os.path.join(FIGURES, name), "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:image/{kind};base64,{encoded}"

def load_slides():
    """deck.json의 순서대로 슬라이드 조각을 읽어 하나로 이어 붙임."""
    deck = json.loads(read_text(os.path.join(HERE, "deck.json")))
    pieces = [read_text(os.path.join(HERE, "slides", f"{slide_id}.html")) for slide_id in deck["order"]]
    return deck["title"], "\n".join(pieces)

def embed_figures(slides):
    for blob_id, name in BLOB_TO_FIGURE.items():
        slides = slides.replace(f"/_blob/{blob_id}", figure_as_text(name))
    return slides

def main():
    title, slides = load_slides()
    slides = embed_figures(slides)
    # 표에 없는 그림 주소가 남아 있으면 그 그림은 안 보이므로 알려 줌
    if "/_blob/" in slides:
        print("경고: BLOB_TO_FIGURE에 없는 그림 주소가 남아 있음")
    page = read_text(os.path.join(HERE, "template.html"))
    page = page.replace("{{TITLE}}", title).replace("{{SLIDES}}", slides)
    with open(OUTPUT, "w", encoding="utf-8") as file:
        file.write(page)
    print(f"저장 완료: {OUTPUT} ({os.path.getsize(OUTPUT) // 1024} KB)")

if __name__ == "__main__":
    main()
