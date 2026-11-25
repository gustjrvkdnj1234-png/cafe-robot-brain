"""Demo-only prompt and tools for the voice app (the v1 experiment prompt/tools are not changed).

prompt = v1 build_system_prompt() text with demo/rules.txt inserted just before "[메뉴판]"
         (the "[현재 주문 내역]" block stays at the end, so the deploy/append layouts still work).
tools  = v1 data/tools.json + demo/inform_tool.json.
inform answers and the free-text guard read data/menu.json only (single source of truth).
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from cafe_common import build_system_prompt, canonical_menu, items_by_name, load_menu, load_tools  # noqa: E402

MENU_ANCHOR = "\n[메뉴판]\n"


# A3 can come from the prompt ("prompt": rule + edits below, and the app safety net) or from the app only
# ("app": v1 temperature wording, no [온도 묻기] section; the app holds a hot-or-iced add_item without a
# temperature and asks). Set by engines.RUN["a3"] / server --a3.
A3 = {"mode": "app"}   # default: app safety net only (2026-11-25 regression: prompt rule cost 8 other cafe items)
A3_SECTION = "[온도 묻기]"


def rules_text() -> str:
    text = (DEMO / "rules.txt").read_text(encoding="utf-8").strip()
    if A3["mode"] != "prompt":
        parts = re.split(r"\n\n(?=\[)", text)
        text = "\n\n".join(p for p in parts if not p.startswith(A3_SECTION))
    return text


# A3 (ask the temperature): the v1 text says "hot unless stated"; the demo copy says "ask" for items that
# come both hot and iced. Each replacement must match exactly once, so a v1 wording change fails loudly.
PROMPT_EDITS = [
    ("손님이 말하지 않은 옵션은 묻지 말고 기본값을 쓴다. (Don't ask about options the customer didn't mention; use defaults.)",
     "손님이 말하지 않은 옵션은 묻지 말고 기본값을 쓴다. 단, 온도는 예외다(아래 [온도 묻기]). "
     "(Don't ask about options the customer didn't mention; use defaults. Exception: temperature.)"),
    ("- 온도: 따뜻하게(hot). 단, 콜드브루와 아이스만 되는 메뉴(에이드)는 아이스(iced). 디저트는 온도 없음.",
     "- 온도: 기본값 없음. 따뜻한 것과 아이스가 둘 다 되는 메뉴는 손님이 말하지 않으면 묻는다(아래 [온도 묻기]). "
     "아이스만 되는 메뉴(에이드)는 아이스(iced), 따뜻하게만 되는 메뉴(에스프레소)는 따뜻하게(hot). 디저트는 온도 없음."),
]
BOTH_TEMPS = re.compile(r"온도 따뜻한/아이스\(기본 [^)]*\)")
TEMP_TOOL_DESC = ("Drink temperature, exactly as the customer said it. No default for items that come both hot and iced: "
                  "if the customer did not say, ask instead of adding. Ade: iced only; espresso: hot only.")


def system_prompt(order_state) -> str:
    sp = build_system_prompt(order_state)
    for old, new in (PROMPT_EDITS if A3["mode"] == "prompt" else []):
        assert sp.count(old) == 1, f"v1 prompt text changed: {old[:30]}"
        sp = sp.replace(old, new)
    if A3["mode"] == "prompt":
        sp = BOTH_TEMPS.sub("온도 따뜻한/아이스(말하지 않으면 묻기)", sp)
    i = sp.index(MENU_ANCHOR)
    return sp[:i] + "\n\n" + rules_text() + sp[i:]


def tools() -> list:
    return _tools(A3["mode"])


@lru_cache(maxsize=2)
def _tools(_mode) -> list:
    v1 = json.loads(json.dumps(load_tools()))               # deep copy: data/tools.json is not changed
    for t in v1:
        if t["function"]["name"] == "add_item" and A3["mode"] == "prompt":
            t["function"]["parameters"]["properties"]["temperature"]["description"] = TEMP_TOOL_DESC
    extra = [json.loads((DEMO / f).read_text(encoding="utf-8")) for f in ("inform_tool.json", "read_order_tool.json")]
    return v1 + extra


# Order-flow line appended to the "[현재 주문 내역]" message (A1/A2). Nothing is added while ordering,
# so the exams (which never have a phase) see exactly the same message as before.
PHASE_LINES = {
    "awaiting": "\n[주문 확인 대기] 위 주문을 손님에게 읽어 드리고 확인을 물었다. 손님이 맞다고 하거나 매장/포장을 말하면 "
                "confirm_order를 호출한다(포장이면 dine_option=\"takeout\"). 메뉴를 더하거나 바꾸면 그 도구를 호출한다.",
    "confirmed": "\n[주문 완료] 이 주문은 이미 들어갔다. 메뉴를 더하거나 바꿀 수 없다.",
}


def phase_line(phase) -> str:
    return PHASE_LINES.get(phase or "", "")


# ---------------------------------------------------------------- Korean particles
def _final(word: str):
    ch = str(word).strip()[-1:]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28
    return None


def _p(word, with_final, without, unknown):
    f = _final(word)
    return f"{word}{with_final if f else without}"   # non-Hangul ending: vowel form (never "은(는)" aloud)


def topic(w):
    return _p(w, "은", "는", "은(는)")


def ga(w):
    return _p(w, "이", "가", "이(가)")


def ul(w):
    return _p(w, "을", "를", "을(를)")


def listing(words) -> str:
    return ", ".join(words)


# ---------------------------------------------------------------- menu lookups
CATEGORY_WORDS = {"coffee": ["커피", "커피류"], "latte": ["라떼", "라떼류", "라테"], "tea": ["차", "티", "차류", "티류"],
                  "ade": ["에이드", "에이드류"], "dessert": ["디저트", "디저트류", "케이크", "케이크류", "먹을 거", "먹을거", "간식", "빵"]}


def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def resolve_menu(item):
    """Canonical menu item name (aliases included) or None."""
    if not item:
        return None
    name = canonical_menu(str(item).strip())
    return name if name in items_by_name() else None


def resolve_category(item):
    n = _norm(item)
    if not n:
        return None
    cats = load_menu()["categories"]
    for key, ko in cats.items():
        if n == _norm(ko):
            return key
    for key, words in CATEGORY_WORDS.items():
        if n in [_norm(w) for w in words]:
            return key
    return None


def resolve_not_offered(item):
    n = _norm(item)
    for x in load_menu()["not_offered"]:
        if any(_norm(k) in n for k in x["keywords"]) or _norm(x["ko"]) in n:
            return x
    return None


@lru_cache(maxsize=1)
def common_items() -> dict:
    d = json.loads((DEMO / "common_cafe_items.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def common_category(item):
    """Menu category to suggest for a common item that is not on the menu (or None)."""
    n = _norm(item)
    for cat, words in common_items().items():
        if any(_norm(w) == n for w in words):
            return cat if cat in load_menu()["categories"] else None
    for suffix, cat in (("라떼", "latte"), ("라테", "latte"), ("에이드", "ade"), ("케이크", "dessert"), ("차", "tea"), ("티", "tea")):
        if n.endswith(suffix):
            return cat
    return None


def items_in(cat) -> list:
    return [it["name"] for it in load_menu()["items"] if it["category"] == cat]


OPTION_WORDS = {w: k for k, ws in {
    "sweetness": ["sweetness", "당도", "단맛", "달기", "단정도", "당도조절"],
    "shots": ["shots", "shot", "shots_delta", "샷", "샷추가", "에스프레소샷"],
    "size": ["size", "사이즈", "크기", "용량"],
    "temperature": ["temperature", "온도"],
    "milk": ["milk", "우유", "오트", "오트우유", "오트밀크"],
}.items() for w in ws}


# ---------------------------------------------------------------- inform answers
def inform_answer(topic_: str, item=None) -> str:
    import replies  # templates live in replies.json
    t = replies.T()["inform"]
    menu = load_menu()
    cats = menu["categories"]
    name = resolve_menu(item)
    it = items_by_name().get(name) if name else None
    size_words = lambda sizes: listing(t["size_names"].get(s, s) for s in sizes)  # noqa: E731
    opt = OPTION_WORDS.get(_norm(item))
    if opt and not it:                    # a question about an option itself ("당도가 어느 정도예요?")
        return t["option_detail"][opt]

    if topic_ == "category_items":
        cat = resolve_category(item) or (it and it["category"])
        if cat:
            return t["category"].format(cat_topic=topic(cats[cat]), items_ga=ga(listing(items_in(cat))))
        return t["all_categories"].format(cats_ga=ga(listing(cats.values())))

    if topic_ == "not_offered" or (topic_ == "item_detail" and item and not it):
        if it:
            return t["offered"].format(menu=name)
        cat = resolve_category(item)
        if cat:
            return t["category"].format(cat_topic=topic(cats[cat]), items_ga=ga(listing(items_in(cat))))
        no = resolve_not_offered(item)
        thing = no["ko"] if no else (str(item).strip() or "그건")
        out = t["not_offered"].format(thing_topic=topic(thing))
        sug = common_category(item) if not no else None
        if sug:
            out += " " + t["suggest"].format(cat_topic=topic(cats[sug]), items_ga=ga(listing(items_in(sug))))
        return out

    if topic_ == "sizes":
        if it:
            key = "two" if len(it["sizes"]) > 1 else "one" if it["sizes"] else "none"
            return t[f"sizes_{key}"].format(menu_topic=topic(name), sizes_ga=ga(size_words(it["sizes"])))
        drinks = [x for x in menu["items"] if x["sizes"]]
        allsizes = sorted({s for x in drinks for s in x["sizes"]}, key=["regular", "large"].index)
        out = t["sizes_general"].format(sizes_ga=ga(size_words(allsizes)))
        one = [x["name"] for x in drinks if len(x["sizes"]) == 1]
        if one:
            out += " " + t["sizes_general_except"].format(items_topic=topic(listing(one)))
        return out

    if topic_ == "temperatures":
        if it:
            temps = it["temperatures"]
            key = "both" if len(temps) > 1 else ("iced" if temps == ["iced"] else "hot") if temps else "none"
            return t[f"temps_{key}"].format(menu_topic=topic(name))
        out = [t["temps_general"]]
        iced = [x["name"] for x in menu["items"] if x["temperatures"] == ["iced"]]
        hot = [x["name"] for x in menu["items"] if x["temperatures"] == ["hot"]]
        if iced:
            out.append(t["temps_general_iced"].format(items_topic=topic(listing(iced))))
        if hot:
            out.append(t["temps_general_hot"].format(items_topic=topic(listing(hot))))
        return " ".join(out)

    if topic_ == "options":
        names = t["option_names"]
        if it:
            opts = []
            if len(it["sizes"]) > 1:
                opts.append(names["size"])
            if len(it["temperatures"]) > 1:
                opts.append(names["temperature"])
            for k in ("shots", "sweetness", "milk"):
                if it[k]:
                    opts.append(names[k])
            if not opts:
                return t["options_none"].format(menu_topic=topic(name))
            return t["options_item"].format(menu_topic=topic(name), opts_ul=ul(listing(opts)))
        return t["options_general"].format(opts_ul=ul(listing(names.values())))

    if topic_ == "item_detail" and it:
        temps = it["temperatures"]
        tk = "both" if len(temps) > 1 else ("iced" if temps == ["iced"] else "hot") if temps else "none"
        sk = "two" if len(it["sizes"]) > 1 else "one" if it["sizes"] else "none"
        return t["detail"].format(menu_topic=topic(name), temp_phrase=t["detail_temp"][tk],
                                  size_phrase=t["detail_size"][sk].format(sizes_ga=ga(size_words(it["sizes"]))))
    return t["unknown"]


# ---------------------------------------------------------------- free-text guard
PARTICLES = ("이에요", "예요", "이요", "이랑", "랑", "와", "과", "이나", "나", "은", "는", "이", "가", "을", "를",
             "도", "만", "요", "로", "으로", "에", "의")
SUFFIXES = ("라떼", "라테", "에이드", "스무디", "프라페", "주스", "쿠키", "케이크", "티", "차")


@lru_cache(maxsize=1)
def menu_terms() -> set:
    m = load_menu()
    terms = {_norm(it["name"]) for it in m["items"]} | {_norm(a["alias"]) for a in m["aliases"]}
    terms |= {_norm(v) for v in m["categories"].values()}
    return terms


def _strip(word: str) -> str:
    w = re.sub(r"[^\w가-힣]", "", word)
    changed = True
    while changed and len(w) > 1:
        changed = False
        for p in PARTICLES:
            if w.endswith(p) and len(w) > len(p) + 1:
                w, changed = w[: -len(p)], True
                break
    return w


def unknown_terms(text: str) -> list:
    """Drink/food/option names in `text` that are not on the menu (common-item list + menu-like words)."""
    found = []
    norm = _norm(text)
    words = [_strip(w) for w in re.split(r"[\s,./?!·~]+", str(text)) if w]
    terms = menu_terms()
    for cat, items in common_items().items():
        for w in items:
            n = _norm(w)
            if n in terms or any(n in t for t in terms):
                continue                       # on the menu after all
            hit = (n in norm) if len(n) >= 3 else (n in words)
            if hit and n not in [_norm(x) for x in found]:
                found.append(w)
    for w in words:
        n = _norm(w)
        if len(n) < 3 or n in terms or any(n == _norm(x) for x in found):
            continue
        if n.endswith(SUFFIXES):
            best = max((difflib.SequenceMatcher(None, n, t).ratio() for t in terms), default=0)
            if best < 0.8:
                found.append(w)
    return found
