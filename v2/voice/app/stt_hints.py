"""C2: menu-word hint text for Whisper (initial_prompt / hotwords), built from data/menu.json.

Menu item names + nicknames (short aliases, or ones that imply a temperature; no spacing or spelling
variants such as "콜드 브루", "레모네이드", "라테", which would pull the output toward them) + option words.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPELLING_VARIANTS = {"라테"}          # same as bench_stt.VARIANTS (standard spelling of 라떼)
# spoken option words that are not literally in menu.json (it has the canonical Korean option values)
EXTRA_WORDS = ["반샷", "샷 추가", "큰 걸로", "테이크아웃", "포장", "매장", "캐리어", "덜 달게", "더 달게"]


def hint_text() -> str:
    m = json.loads((ROOT / "data" / "menu.json").read_text(encoding="utf-8"))
    names = [it["name"] for it in m["items"]]
    aliases = []
    for a in m["aliases"]:
        w = a["alias"]
        if (" " in w or w in names or w in aliases or w in SPELLING_VARIANTS or
                not (len(w) <= 3 or a.get("implies"))):
            continue
        aliases.append(w)
    opts = []
    for o in m["options"].values():
        ko = o.get("ko")
        if isinstance(ko, dict):
            opts += [v for v in ko.values() if v not in opts]
    opts += [w for w in EXTRA_WORDS if w not in opts]
    return "카페 주문. " + ", ".join(names) + ". " + ", ".join(aliases) + ". " + ", ".join(opts) + "."
