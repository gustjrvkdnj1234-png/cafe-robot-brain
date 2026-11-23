"""Fixed Korean reply templates for the voice demo, and order-state updates.

Tool calls are applied in order, exactly like the v1 live page (scripts/robot_ui.apply_calls).
Each reply comes in two forms:
  spoken  - for TTS and the screen, native Korean counters ("두 잔")
  history - for the conversation sent back to the model, in the exam style ("2잔")
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from cafe_common import defaults, items_by_name, normalize_add  # noqa: E402
from robot_ui import apply_calls  # noqa: E402

FALLBACK = "죄송합니다, 다시 한 번 말씀해 주시겠어요?"
STAFF = "직원을 불러 드릴게요. 잠시만 기다려 주세요."

NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
          11: "열한", 12: "열두", 13: "열세", 14: "열네", 15: "열다섯", 20: "스무"}


def count(n, unit: str, native: bool) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return f"{n}{unit}"
    if native and n in NATIVE:
        return f"{NATIVE[n]} {unit}"
    return f"{n}{unit}"


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def shots_phrase(v, native: bool) -> str:
    v = _num(v)
    n = abs(v)
    whole, half = int(n), (n - int(n)) >= 0.5
    if whole == 0:
        amount = "반 개"
    else:
        amount = count(whole, "개", native) + (" 반" if half else "")
    return f"샷 {amount} {'추가' if v > 0 else '빼고'}"


def unit_of(menu) -> str:
    item = items_by_name().get(menu)
    return "개" if item and item["category"] == "dessert" else "잔"


def item_phrase(args: dict, native: bool) -> str:
    """'아이스 아메리카노 큰 사이즈 두 잔' (options between name and count)."""
    a = normalize_add(args)
    item = items_by_name().get(a["menu"])
    words = []
    if item and len(item["temperatures"]) > 1 and a.get("temperature") in ("hot", "iced"):
        words.append("아이스" if a["temperature"] == "iced" else "따뜻한")
    elif not item and a.get("temperature") == "iced":
        words.append("아이스")
    words.append(str(a["menu"]))
    if a.get("size") == "large":
        words.append("큰 사이즈")
    if _num(a.get("shots_delta")):
        words.append(shots_phrase(a["shots_delta"], native))
    if a.get("sweetness") == "less":
        words.append("덜 달게")
    elif a.get("sweetness") == "more":
        words.append("더 달게")
    if a.get("milk") == "oat":
        words.append("오트우유")
    words.append(count(a.get("quantity") or 1, unit_of(a["menu"]), native))
    return " ".join(words)


def name_only(args: dict) -> str:
    """'아이스 아메리카노' (no options, no count) for modify/remove sentences."""
    a = normalize_add(args)
    item = items_by_name().get(a["menu"])
    if item and len(item["temperatures"]) > 1 and a.get("temperature") in ("hot", "iced"):
        return ("아이스 " if a["temperature"] == "iced" else "따뜻한 ") + str(a["menu"])
    return str(a["menu"])


def _has_final(word: str) -> bool | None:
    ch = word.strip()[-1:]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    return None


def obj(word: str) -> str:
    f = _has_final(word)
    return word + ("을" if f else "를" if f is False else "을(를)")


def changes_phrase(ch: dict, native: bool, menu_before=None) -> str:
    out = []
    for k, v in (ch or {}).items():
        if v is None:
            continue
        if k == "menu":
            out.append(f"{v}{'으로' if _has_final(str(v)) and not str(v).endswith('ㄹ') else '로'}")
        elif k == "temperature":
            out.append({"iced": "아이스로", "hot": "따뜻하게"}.get(v, str(v)))
        elif k == "size":
            out.append({"large": "큰 사이즈로", "regular": "기본 사이즈로"}.get(v, str(v)))
        elif k == "shots_delta":
            out.append(shots_phrase(v, native) + "로" if _num(v) else "샷 기본으로")
        elif k == "sweetness":
            out.append({"less": "덜 달게", "more": "더 달게", "normal": "보통 당도로"}.get(v, str(v)))
        elif k == "milk":
            out.append({"oat": "오트우유로", "regular": "일반 우유로"}.get(v, str(v)))
        elif k == "quantity":
            out.append(count(v, unit_of(menu_before), native) + "으로")
        else:
            out.append(f"{k} {v}")
    return ", ".join(out)


def totals_phrase(state: list, native: bool) -> str:
    drinks = sum(int(_num(it.get("quantity"), 1)) for it in state if unit_of(it.get("menu")) == "잔")
    desserts = sum(int(_num(it.get("quantity"), 1)) for it in state if unit_of(it.get("menu")) == "개")
    parts = []
    if drinks and desserts:
        parts = [f"음료 {count(drinks, '잔', native)}", f"디저트 {count(desserts, '개', native)}"]
    elif drinks:
        parts = [count(drinks, "잔", native)]
    elif desserts:
        parts = [count(desserts, "개", native)]
    return ", ".join(parts)


def _index(a):
    try:
        return int(a.get("item_index"))
    except (TypeError, ValueError):
        return None


def build_reply(calls: list, state: list) -> dict:
    """Apply tool calls to the order state and build the reply.

    Returns {state, spoken, history, confirmed, dine_option, staff_reason, kinds}."""
    spoken, history, kinds = [], [], []
    adds_s, adds_h = [], []
    confirmed, dine, staff_reason = False, None, None
    cur = [dict(x) for x in state]

    def flush_adds():
        if adds_s:
            spoken.append(", ".join(adds_s) + " 담았습니다.")
            history.append(", ".join(adds_h) + " 담았습니다.")
            adds_s.clear()
            adds_h.clear()

    for c in calls or []:
        name, a = c.get("name"), c.get("arguments") or {}
        if name == "add_item":
            kinds.append("add")
            adds_s.append(item_phrase(a, True))
            adds_h.append(item_phrase(a, False))
            cur = apply_calls(cur, [c])
            continue
        flush_adds()
        if name in ("modify_item", "remove_item"):
            i = _index(a)
            if i is None or not 1 <= i <= len(cur):
                kinds.append(name.split("_")[0] + "_missing")
                s = f"{i if i is not None else '그'}번 메뉴를 찾지 못했습니다. 다시 말씀해 주시겠어요?"
                spoken.append(s)
                history.append(s)
                continue
            target = obj(f"{i}번 {name_only(cur[i - 1])}")
            if name == "modify_item":
                kinds.append("modify")
                ch = a.get("changes") if isinstance(a.get("changes"), dict) else {}
                menu_before = cur[i - 1].get("menu")
                spoken.append(f"{target} {changes_phrase(ch, True, menu_before)} 바꿨습니다.")
                history.append(f"{target} {changes_phrase(ch, False, menu_before)} 바꿨습니다.")
            else:
                kinds.append("remove")
                spoken.append(f"{target} 뺐습니다.")
                history.append(f"{target} 뺐습니다.")
            cur = apply_calls(cur, [c])
        elif name == "ask_clarification":
            kinds.append("ask")
            q = str(a.get("question") or "").strip()
            spoken.append(q or FALLBACK)
            history.append(q or FALLBACK)
        elif name == "call_staff":
            kinds.append("staff")
            staff_reason = str(a.get("reason") or "").strip() or "직원 호출"
            spoken.append(STAFF)
            history.append(STAFF)
        elif name == "confirm_order":
            kinds.append("confirm")
            dine = a.get("dine_option") if a.get("dine_option") in ("dine_in", "takeout") else defaults()["dine_option"]
            if not cur:
                s = "담긴 메뉴가 없습니다. 무엇을 드릴까요?"
                spoken.append(s)
                history.append(s)
                continue
            confirmed = True
            head = "포장 주문 확인했습니다." if dine == "takeout" else "주문 확인했습니다."
            spoken.append(f"{head} 총 {totals_phrase(cur, True)}입니다.")
            history.append(f"{head} 총 {totals_phrase(cur, False)}입니다.")
        else:
            kinds.append("unknown")
    flush_adds()
    if not spoken:
        spoken, history = [FALLBACK], [FALLBACK]
        kinds.append("fallback")
    return {"state": cur, "spoken": " ".join(spoken), "history": " ".join(history), "confirmed": confirmed,
            "dine_option": dine, "staff_reason": staff_reason, "kinds": kinds}


def order_lines(state: list) -> list[str]:
    return [item_phrase(it, True) for it in state]
