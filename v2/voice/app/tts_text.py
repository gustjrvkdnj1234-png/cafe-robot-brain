"""Normalize every sentence before it is spoken (templates, model questions, free text).

- native Korean counters for 1-20: 1잔 -> 한 잔, 2잔 -> 두 잔, 3개 -> 세 개, 20잔 -> 스무 잔
- shot changes: 샷 -0.5 / -0.5샷 / shots_delta -0.5 -> 반 샷, +1 -> 한 샷 추가, -1 -> 한 샷 빼고
- list markup: numbering (1. 2) ①), bullets (- * •), [heading] lines, brackets, markdown symbols
The result is also what the screen shows, so the customer reads what the robot says.
"""
from __future__ import annotations

import re

NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
          11: "열한", 12: "열두", 13: "열세", 14: "열네", 15: "열다섯", 16: "열여섯", 17: "열일곱", 18: "열여덟",
          19: "열아홉", 20: "스무"}
COUNTERS = "잔|개|샷|명|컵|조각|가지|병"          # native-number counters (not 분/원/번, which take Sino numbers)
NUM = r"[+-]?\d+(?:\.\d+)?"


def native(n: int, counter: str) -> str:
    return f"{NATIVE[n]} {counter}" if n in NATIVE else f"{n}{counter}"


def shot_delta(v: float) -> str:
    """shots_delta value -> spoken phrase."""
    n = abs(v)
    whole, half = int(n), (n - int(n)) >= 0.5
    if n == 0:
        return "샷 기본"
    if whole == 0:
        base = "반 샷"
    else:
        base = f"{NATIVE.get(whole, str(whole))} 샷" + (" 반" if half else "")
    if v < 0:
        return base if whole == 0 else base + " 빼고"
    return base + " 추가"


def _shot_signed(m) -> str:
    return shot_delta(float(m.group(1)))


def _shot_count(m) -> str:
    """'2샷' (no sign) = two shots; '+1샷' / '-0.5샷' (signed) = a change."""
    s = m.group(1)
    if s[0] in "+-":
        return shot_delta(float(s))
    v = float(s)
    if v == 0.5:
        return "반 샷"
    return native(int(v), "샷") if v.is_integer() else f"{s}샷"


def _count(m) -> str:
    return native(int(m.group(1)), m.group(2))


def _paren(m) -> str:
    return " " + m.group(1).replace(",", " ") + " "


LIST_MARK = re.compile(r"^\s*(?:\d{1,2}\s*[.)]\s+|\d{1,2}\s*[.)](?=[가-힣])|[-*•·▪◦]\s*|[①-⑳]\s*)")
HEADING = re.compile(r"^\s*[\[【][^\]】]*[\]】]\s*:?\s*$")


def normalize(text: str) -> str:
    if not text:
        return ""
    lines = []
    for raw in str(text).replace("\r", "").split("\n"):
        line = raw.strip()
        if not line or HEADING.match(line):
            continue                                   # "[현재 주문 내역]" etc.
        line = LIST_MARK.sub("", line)
        lines.append(line)
    out = ""
    for line in lines:
        if out:
            out += " " if out[-1] in ".?!" else ", "
        out += line
    s = out
    s = re.sub(r"\*\*|__|`|#+\s*", "", s)                    # markdown
    s = re.sub(r"[\[\]【】{}<>]", " ", s)
    s = re.sub(r"\(([^()]*)\)", _paren, s)                   # 바닐라라떼(샷 -0.5, 덜 달게) -> 바닐라라떼 샷 -0.5 덜 달게
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"shots?_?delta\s*[:=]?\s*(" + NUM + ")", _shot_signed, s, flags=re.I)
    s = re.sub(r"샷\s*[:=]?\s*([+-]\d+(?:\.\d+)?)", _shot_signed, s)      # 샷 -0.5, 샷 +1
    s = re.sub(r"(" + NUM + r")\s*샷", _shot_count, s)                   # -0.5샷, 2샷
    s = re.sub(r"반샷", "반 샷", s)
    s = re.sub(r"(?<![\d.])(\d{1,3})\s*(" + COUNTERS + r")", _count, s)
    s = re.sub(r"(?<![가-힣])(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)(잔|개|샷)", r"\1 \2", s)
    s = re.sub(r"\s*(?:→|->|=>)\s*", ", ", s)
    s = re.sub(r"\s+/\s+|(?<=[가-힣])/(?=[가-힣])", ", ", s)
    s = re.sub(r"[*#|~=_]", " ", s)
    s = re.sub(r"(?<=\s)[-+](?=\s)", " ", s)                 # stray +/- between words
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.?!])", r"\1", s)
    s = re.sub(r",\s*,+", ",", s)
    s = re.sub(r"([,.?!])(?=[가-힣A-Za-z])", r"\1 ", s)
    return s.strip(" ,")


MIN_FIRST = 4   # a first piece shorter than this (e.g. "네.") is joined with the next sentence


def sentences(text: str) -> list[str]:
    """D3: split normalized spoken text at sentence ends (. ? ! followed by a space), so the first sentence can be
    synthesized and played while the rest is still being synthesized. Very short first pieces stay joined."""
    parts = [p.strip() for p in re.split(r"(?<=[.?!])\s+", text.strip()) if p.strip()]
    if len(parts) > 1 and len(re.sub(r"\W", "", parts[0])) < MIN_FIRST:
        parts = [parts[0] + " " + parts[1]] + parts[2:]
    return parts or [text]
