"""Reply wording for the voice demo (templates in replies.json) and order-state updates.

Tool calls are applied in order with the v1 live-page logic (scripts/robot_ui.apply_calls).
Replies are short. Item numbers are never read aloud; when an item must be named it is called by the
shortest description that is unique in the order ("바닐라라떼" -> "아이스 바닐라라떼" -> ... -> ordinal).
inform answers and the free-text guard come from demo/demo_config.py (menu.json only).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(APP / "demo"))
from cafe_common import defaults, items_by_name, normalize_add  # noqa: E402
from robot_ui import apply_calls  # noqa: E402
import demo_config  # noqa: E402
from tts_text import shot_delta  # noqa: E402
from demo_config import topic  # noqa: E402

TEMPLATES = APP / "replies.json"
_cache = {"mtime": None, "t": None}

NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
          11: "열한", 12: "열두", 13: "열세", 14: "열네", 15: "열다섯", 20: "스무"}
READ_ORDER = ("temp", "menu", "size", "shots", "sweet", "milk")


def T() -> dict:
    """replies.json, re-read when the file changes (edit wording without restarting)."""
    m = TEMPLATES.stat().st_mtime
    if _cache["mtime"] != m:
        _cache["t"], _cache["mtime"] = json.loads(TEMPLATES.read_text(encoding="utf-8")), m
    return _cache["t"]


def _final(word: str):
    ch = str(word).strip()[-1:]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28
    return None


def ro(word: str) -> str:
    f = _final(word)
    return word + ("로" if f in (0, 8) else "으로" if f else "(으)로")


def iyo(word: str) -> str:
    f = _final(word)
    return word + ("이요" if f else "요")


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _key(v) -> str:
    v = _num(v)
    return str(int(v)) if v.is_integer() else str(v)


def count(n, unit: str, native: bool = True) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return f"{n}{unit}"
    return f"{NATIVE[n]} {unit}" if native and n in NATIVE else f"{n}{unit}"


def unit_of(menu) -> str:
    item = items_by_name().get(menu)
    return "개" if item and item["category"] == "dessert" else "잔"


def _amount(v) -> str:
    n = abs(_num(v))
    whole, half = int(n), (n - int(n)) >= 0.5
    return "반" if whole == 0 else (count(whole, "개") + (" 반" if half else ""))


# ---------------------------------------------------------------- describing items
def _words(args: dict) -> dict:
    w = T()["item_words"]
    a = normalize_add(args)
    item = items_by_name().get(a["menu"])
    out = {k: "" for k in READ_ORDER}
    out["menu"] = str(a["menu"])
    out["qty"] = count(a.get("quantity") or 1, unit_of(a["menu"]))
    if a.get("size") == "large":
        out["size"] = w["size_large"]
    if _num(a.get("shots_delta")):
        v = a["shots_delta"]
        out["shots"] = w["shots"].get(_key(v)) or shot_delta(_num(v))
    out["sweet"] = w["sweetness"].get(a.get("sweetness"), "")
    if a.get("milk") == "oat":
        out["milk"] = w["milk_oat"]
    if (item and len(item["temperatures"]) > 1) or (not item and a.get("temperature") == "iced"):
        out["temp"] = w["temperature"].get(a.get("temperature"), "")
    return out


def _join(parts) -> str:
    return " ".join(p for p in parts if p)


def describe(args: dict, with_count: bool = True) -> str:
    """'아이스 아메리카노 큰 사이즈 반샷 두 잔'."""
    w = _words(args)
    return _join([w[k] for k in READ_ORDER] + ([w["qty"]] if with_count else []))


def name_in_order(state: list, i: int) -> str:
    """Shortest description of line i (0-based) that no other line in the order shares."""
    levels = [("menu",), ("temp", "menu"), ("temp", "menu", "size"), READ_ORDER]
    words = [_words(x) for x in state]
    for keys in levels:
        mine = _join(words[i][k] for k in keys)
        if sum(1 for w in words if _join(w[k] for k in keys) == mine) == 1:
            return mine
    full = _join(words[i][k] for k in READ_ORDER)
    same = [j for j, w in enumerate(words) if _join(w[k] for k in READ_ORDER) == full]
    ords = T()["item_words"]["ordinal"]
    k = same.index(i)
    return f"{ords[k] if k < len(ords) else str(k + 1) + ' 번째'} {full}"


def changes_text(ch: dict, menu_before) -> str:
    c = T()["changes"]
    forms = []
    for k, v in (ch or {}).items():
        if v is None:
            continue
        if k == "shots_delta":
            pair = c["shots"].get(_key(v)) or [s.format(n=_amount(v)) for s in
                                                (c["shots_more"] if _num(v) > 0 else c["shots_less"])]
        elif k == "quantity":
            pair = [s.format(count_ro=ro(count(v, unit_of(menu_before)))) for s in c["quantity"]]
        elif k == "menu":
            pair = [s.format(menu_ro=ro(str(v))) for s in c["menu"]]
        elif k in c and isinstance(c[k], dict) and str(v) in c[k]:
            pair = c[k][str(v)]
        else:
            pair = [f"{k} {v}(으)로 하고", f"{k} {v}(으)로 할게요"]
        forms.append(pair)
    if not forms:
        return "그대로 할게요"
    return " ".join([p[0] for p in forms[:-1]] + [forms[-1][1]])


# ---------------------------------------------------------------- one turn
def _index(a):
    try:
        return int(a.get("item_index"))
    except (TypeError, ValueError):
        return None


def ieyo(word: str) -> str:
    f = _final(word)
    return word + ("이에요" if f else "예요")


def order_listing(state: list) -> str:
    """'아이스 아메리카노 한 잔, 청포도에이드 덜 달게 한 잔' (from the order state, never from model text)."""
    return ", ".join(describe(it) for it in state)


def read_order_text(state: list) -> str:
    t = T()
    return t["read_order"].format(items_ieyo=ieyo(order_listing(state))) if state else t["read_order_empty"]


def readback_text(state: list, dine) -> str:
    t = T()
    ask = {"takeout": t["readback_ok_takeout"], "dine_in": t["readback_ok_dine_in"]}.get(dine, t["ask_dine"])
    return t["readback"].format(items_iyo=iyo(order_listing(state)), question=ask)


def needs_temperature(args: dict, temp_said: bool) -> bool:
    """add_item for an item that comes both hot and iced, with no temperature from the model or the customer."""
    if temp_said or str(args.get("temperature") or "").strip():
        return False
    item = items_by_name().get(normalize_add(args)["menu"])
    return bool(item) and len(item["temperatures"]) > 1


def listing_ko(words) -> str:
    return ", ".join(words)


ORDER_LIST = re.compile(r"주문\s*내역|^\s*\d+\s*[.)]", re.M)
CHANGES_ORDER = ("add_item", "modify_item", "remove_item")


def build_reply(calls: list, content: str, state: list, first_item_possible: bool = False,
                phase: str = "ordering", dine=None, temp_said: bool = True) -> dict:
    """Apply tool calls and build the spoken reply.

    first_item_possible: this customer has not ordered anything yet in this session, so the
    one-time "더 필요하신 거 있으세요?" may follow the first item.
    phase: ordering -> awaiting (confirm_order read the order back and asked) -> confirmed (locked).
      - confirm_order while ordering does not confirm: it reads the order back and asks dine-in/takeout
        when unknown (A1). It confirms only in "awaiting" (the customer said yes) with a known dine option.
      - a change while awaiting is applied and the order goes back to "ordering" (unconfirmed).
      - once confirmed, add/modify/remove are not applied: the robot says so and calls staff (A2).
    dine: dine option the customer stated in this session (None = not said).
    Returns state, spoken, kinds, phase, confirmed, dine_option, staff_reason, tool_results (one per call,
    for the chat-format history), blocked (guarded free text) and legacy_history (old plain text)."""
    t = T()
    sents, kinds, results = [], [], []
    adds = []
    confirmed, staff_reason, blocked = False, None, None
    new_phase, readback, changed = phase, False, False
    cur = [dict(x) for x in state]

    def flush_adds():
        if adds:
            sents.append(("add", t["add"].format(items_iyo=iyo(", ".join(adds)))))
            adds.clear()

    held, history_calls = [], []
    for c in calls or []:
        name, a = c.get("name"), c.get("arguments") or {}
        if phase == "confirmed" and name in CHANGES_ORDER:
            history_calls.append(c)
            flush_adds()
            if "locked" not in kinds:
                kinds += ["locked", "staff"]
                sents.append(("locked", t["locked"]))
                staff_reason = t["locked_staff_reason"]
            results.append({"ok": False, "error": "order already placed; staff called"})
            continue
        if name == "add_item" and needs_temperature(a, temp_said):
            # A3 safety net: the model added a hot-or-iced item although the customer said no temperature.
            # Not added; the robot asks, and the history records the question instead of the add.
            kinds.append("ask_temp")
            held.append(str(normalize_add(a)["menu"]))
            results.append({"ok": False, "not_added": "temperature not given; asked the customer"})
            history_calls.append(None)
            continue
        history_calls.append(c)
        if name == "add_item":
            kinds.append("add")
            adds.append(describe(a))
            cur = apply_calls(cur, [c])
            changed = True
            results.append({"ok": True, "added": describe(a)})
            continue
        flush_adds()
        if name in ("modify_item", "remove_item"):
            i = _index(a)
            if i is None or not 1 <= i <= len(cur):
                kinds.append(name.split("_")[0] + "_missing")
                sents.append(("missing", t["item_not_found"]))
                results.append({"ok": False, "error": f"item_index {a.get('item_index')} not in the current order"})
                continue
            target = topic(name_in_order(cur, i - 1))
            several = len(cur) > 1
            changed = True
            if name == "modify_item":
                kinds.append("modify")
                ch = a.get("changes") if isinstance(a.get("changes"), dict) else {}
                text = changes_text(ch, cur[i - 1].get("menu"))
                sents.append(("modify", t["modify_target" if several else "modify"].format(target=target, changes=text)))
                cur = apply_calls(cur, [c])
                results.append({"ok": True, "item_index": i, "now": describe(cur[i - 1])})
            else:
                kinds.append("remove")
                sents.append(("remove", t["remove_target" if several else "remove"].format(target=target)))
                results.append({"ok": True, "removed": describe(cur[i - 1])})
                cur = apply_calls(cur, [c])
        elif name == "read_order":
            kinds.append("read_order")
            sents.append(("read_order", read_order_text(cur)))
            results.append({"ok": True})
        elif name == "inform":
            kinds.append("inform")
            answer = demo_config.inform_answer(str(a.get("topic") or ""), a.get("item"))
            sents.append(("inform", answer))
            results.append({"ok": True, "told_customer": answer})
        elif name == "ask_clarification":
            kinds.append("ask")
            q = str(a.get("question") or "").strip()
            sents.append(("ask", q or t["fallback"]))
            results.append({"ok": True, "said_to_customer": q})
        elif name == "call_staff":
            kinds.append("staff")
            staff_reason = str(a.get("reason") or "").strip() or "직원 호출"
            sents.append(("staff", t["call_staff"]))
            results.append({"ok": True, "staff_called": True})
        elif name == "confirm_order":
            if phase == "confirmed":
                kinds.append("already_confirmed")
                sents.append(("already", t["already_confirmed"]))
                results.append({"ok": True, "already_placed": True})
                continue
            if not cur:
                kinds.append("confirm_empty")
                sents.append(("confirm_empty", t["confirm_empty"]))
                results.append({"ok": False, "error": "order is empty"})
                continue
            dine = dine or ("takeout" if a.get("dine_option") == "takeout" else None)
            if phase == "awaiting" and not changed:
                if dine:
                    kinds.append("confirm")
                    confirmed, new_phase = True, "confirmed"
                    sents.append(("confirm", t["confirm_takeout" if dine == "takeout" else "confirm"]))
                    results.append({"ok": True, "confirmed": True, "dine_option": dine})
                else:
                    kinds.append("ask_dine")
                    sents.append(("ask_dine", t["ask_dine"]))
                    results.append({"ok": True, "waiting_for": "dine_option"})
            else:
                kinds.append("readback")
                readback, new_phase = True, "awaiting"
                results.append({"ok": True, "read_back_and_asked": True})
        else:
            kinds.append("unknown")
            results.append({"ok": False, "error": f"unknown tool {name}"})
    flush_adds()
    if held:
        only = len(sents) == 0 and len(held) == 1
        q = t["ask_temp"] if only else t["ask_temp_item"].format(menu_topic=topic(listing_ko(held)))
        sents.append(("ask_temp", q))
        hc = {"name": "ask_clarification", "arguments": {"question": q, "reason": "ambiguous"}}
        results = [res for x, res in zip(history_calls, results) if x is not None] + [{"ok": True, "said_to_customer": q}]
        history_calls = [x for x in history_calls if x is not None] + [hc]
        if readback:                                   # cannot read back an order that is still missing a choice
            readback, new_phase = False, phase if phase != "awaiting" else "ordering"
            kinds.remove("readback")
    if changed and phase == "awaiting" and not readback:
        new_phase = "ordering"                     # changed instead of confirming: stays unconfirmed
    if readback:
        # the read-back names the whole order, so an add/modify sentence of the same turn would repeat it
        # ("아바라 테이크아웃이요" -> "네, 아이스 바닐라라떼 한 잔이요. 아이스 바닐라라떼 한 잔이요. ...")
        same = [s for s in sents if s[0] in ("add", "modify")]
        sents = [s for s in sents if s[0] not in ("add", "modify")]
        sents.append(("readback", ("네, " if same else "") + readback_text(cur, dine)))

    if not sents:
        text = (content or "").strip()
        if text and ORDER_LIST.search(text):
            kinds += ["read_order", "free_text_replaced"]   # model wrote the order out: read it from the state
            blocked = {"text": text, "replaced_by": "read_order"}
            sents.append(("read_order", read_order_text(cur)))
        elif text:
            bad = demo_config.unknown_terms(text)
            if bad:
                kinds.append("free_text_blocked")      # names things that are not on the menu
                blocked = {"text": text, "unknown_terms": bad}
                sents.append(("blocked", t["free_text_blocked"]))
            else:
                kinds.append("free_text")              # plain-text answer, menu-safe
                sents.append(("free_text", text))
        else:
            kinds.append("fallback")
            sents.append(("fallback", t["fallback"]))
    if first_item_possible and "add" in kinds and sents[-1][0] == "add":
        sents.append(("followup", t["first_item_followup"]))
    spoken = []
    for k, (_, s) in enumerate(sents):
        if k > 0:
            for p in t["ack_prefixes"]:
                if s.startswith(p):
                    s = s[len(p):]
                    break
        spoken.append(s)
    return {"state": cur, "spoken": " ".join(spoken), "kinds": kinds, "phase": new_phase, "confirmed": confirmed,
            "dine_option": dine, "staff_reason": staff_reason, "tool_results": results, "blocked": blocked,
            "history_calls": [x for x in history_calls if x is not None] if held else list(calls or []),
            "legacy_history": legacy_history(calls, content, state)}


def order_lines(state: list) -> list[str]:
    return [describe(it) for it in state]


# ---------------------------------------------------------------- old history text (comparison only)
def _legacy_item(a: dict) -> str:
    a = normalize_add(a)
    item = items_by_name().get(a["menu"])
    words = []
    if item and len(item["temperatures"]) > 1 and a.get("temperature") in ("hot", "iced"):
        words.append("아이스" if a["temperature"] == "iced" else "따뜻한")
    words.append(str(a["menu"]))
    if a.get("size") == "large":
        words.append("큰 사이즈")
    words.append(f"{a.get('quantity') or 1}{unit_of(a['menu'])}")
    return " ".join(words)


def legacy_history(calls: list, content: str, state: list) -> str:
    """What the first app version stored as the assistant turn (plain exam-style text)."""
    out, adds = [], []
    for c in calls or []:
        name, a = c.get("name"), c.get("arguments") or {}
        if name == "add_item":
            adds.append(_legacy_item(a))
            continue
        if adds:
            out.append(", ".join(adds) + " 담았습니다.")
            adds = []
        if name == "ask_clarification":
            out.append(str(a.get("question") or ""))
        elif name == "call_staff":
            out.append("직원을 불러 드릴게요. 잠시만 기다려 주세요.")
        elif name == "confirm_order":
            out.append("주문 확인했습니다.")
        elif name == "modify_item":
            out.append(f"{a.get('item_index')}번 메뉴를 바꿨습니다.")
        elif name == "remove_item":
            out.append(f"{a.get('item_index')}번 메뉴를 뺐습니다.")
    if adds:
        out.append(", ".join(adds) + " 담았습니다.")
    return " ".join(out) or "죄송합니다, 다시 한 번 말씀해 주시겠어요?"
