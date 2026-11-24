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


def rules_text() -> str:
    return (DEMO / "rules.txt").read_text(encoding="utf-8").strip()


def system_prompt(order_state) -> str:
    sp = build_system_prompt(order_state)
    i = sp.index(MENU_ANCHOR)
    return sp[:i] + "\n\n" + rules_text() + sp[i:]


@lru_cache(maxsize=1)
def tools() -> list:
    return load_tools() + [json.loads((DEMO / "inform_tool.json").read_text(encoding="utf-8"))]


# ---------------------------------------------------------------- Korean particles
def _final(word: str):
    ch = str(word).strip()[-1:]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28
    return None


def _p(word, with_final, without, unknown):
    f = _final(word)
    return f"{word}{with_final if f else without if f == 0 else unknown}"


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


# ---------------------------------------------------------------- inform answers
def inform_answer(topic_: str, item=None) -> str:
    import replies  # templates live in replies.json
    t = replies.T()["inform"]
    menu = load_menu()
    cats = menu["categories"]
    name = resolve_menu(item)
    it = items_by_name().get(name) if name else None
    size_words = lambda sizes: listing(t["size_names"].get(s, s) for s in sizes)  # noqa: E731

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
