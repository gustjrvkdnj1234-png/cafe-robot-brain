"""Shared helpers for the cafe exam: menu loading, argument normalization,
Korean descriptions, and the system prompt given to models."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TOOL_NAMES = ["add_item", "modify_item", "remove_item", "ask_clarification", "confirm_order"]
ADD_KEYS = ["menu", "temperature", "size", "shots_delta", "sweetness", "milk", "quantity"]
REASONS = ["ambiguous", "impossible", "off_menu"]

KO_TEMP = {"hot": "따뜻한", "iced": "아이스"}
KO_SIZE = {"regular": "레귤러", "large": "라지"}
KO_SWEET = {"less": "덜 달게", "normal": "보통 당도", "more": "더 달게"}
KO_MILK = {"regular": "일반 우유", "oat": "오트밀크"}
KO_REASON = {"ambiguous": "무엇을 원하는지 불분명", "impossible": "불가능한 옵션 조합", "off_menu": "메뉴에 없음"}


@lru_cache(maxsize=1)
def load_menu() -> dict:
    return json.loads((DATA / "menu.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_tools() -> list:
    return json.loads((DATA / "tools.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def items_by_name() -> dict:
    return {it["name"]: it for it in load_menu()["items"]}


@lru_cache(maxsize=1)
def name_lookup() -> dict:
    """space-free surface form -> canonical name (canonical names + aliases + English)."""
    m = {}
    for it in load_menu()["items"]:
        m[it["name"].replace(" ", "")] = it["name"]
        m[it["name_en"].replace(" ", "").lower()] = it["name"]
    for a in load_menu()["aliases"]:
        m.setdefault(a["alias"].replace(" ", ""), a["menu"])
    return m


def canonical_menu(name):
    if not isinstance(name, str):
        return name
    key = name.replace(" ", "")
    return name_lookup().get(key, name_lookup().get(key.lower(), name.strip()))


def _as_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return v


def _as_str(v):
    return v.strip().lower() if isinstance(v, str) else v


def normalize_add(args: dict) -> dict:
    """Fill defaults so that an omitted option equals its default value."""
    args = dict(args or {})
    menu = canonical_menu(args.get("menu"))
    item = items_by_name().get(menu)
    out = {"menu": menu}
    temp = _as_str(args.get("temperature"))
    size = _as_str(args.get("size"))
    if item is not None:
        out["temperature"] = temp or item["default_temperature"]
        out["size"] = size or ("regular" if item["sizes"] else None)
        if not item["sizes"] and size == "regular":
            out["size"] = None
    else:
        out["temperature"] = temp
        out["size"] = size or "regular"
    out["shots_delta"] = _as_int(args.get("shots_delta"), 0)
    out["sweetness"] = _as_str(args.get("sweetness")) or "normal"
    out["milk"] = _as_str(args.get("milk")) or "regular"
    out["quantity"] = _as_int(args.get("quantity"), 1)
    return out


def normalize_changes(changes) -> dict:
    if isinstance(changes, str):
        try:
            changes = json.loads(changes)
        except json.JSONDecodeError:
            return {"_raw": changes}
    out = {}
    for k, v in (changes or {}).items():
        if v is None:
            continue
        if k == "menu":
            out[k] = canonical_menu(v)
        elif k in ("shots_delta", "quantity"):
            out[k] = _as_int(v)
        else:
            out[k] = _as_str(v)
    return out


def normalize_call(call: dict) -> dict:
    """Return {"name": ..., "args": {...key args only...}}."""
    name = call.get("name")
    args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if name == "add_item":
        key = normalize_add(args)
    elif name == "modify_item":
        key = {"item_index": _as_int(args.get("item_index")), "changes": normalize_changes(args.get("changes"))}
    elif name == "remove_item":
        key = {"item_index": _as_int(args.get("item_index"))}
    elif name == "ask_clarification":
        key = {"reason": _as_str(args.get("reason"))}  # question wording is ignored
    else:
        key = {}
    return {"name": name, "args": key}


# ---------- Korean descriptions ----------

def describe_item(args: dict) -> str:
    a = normalize_add(args)
    item = items_by_name().get(a["menu"])
    parts = []
    if item and len(item["temperatures"]) > 1:
        parts.append(KO_TEMP.get(a["temperature"], str(a["temperature"])))
    parts.append(str(a["menu"]))
    opts = []
    if a["size"] == "large":
        opts.append("라지")
    if a["shots_delta"]:
        opts.append(f"샷 {a['shots_delta']:+d}")
    if a["sweetness"] != "normal":
        opts.append(KO_SWEET[a["sweetness"]])
    if a["milk"] == "oat":
        opts.append("오트밀크")
    s = " ".join(parts)
    if opts:
        s += f"({', '.join(opts)})"
    unit = "개" if item and item["category"] == "dessert" else "잔"
    return f"{s} {a['quantity']}{unit}"


def describe_changes(changes: dict) -> str:
    c = normalize_changes(changes)
    out = []
    for k, v in c.items():
        if k == "menu":
            out.append(f"메뉴를 {v}(으)로")
        elif k == "temperature":
            out.append(KO_TEMP.get(v, v))
        elif k == "size":
            out.append(KO_SIZE.get(v, v))
        elif k == "shots_delta":
            out.append(f"샷 {v:+d}" if isinstance(v, int) else f"샷 {v}")
        elif k == "sweetness":
            out.append(KO_SWEET.get(v, v))
        elif k == "milk":
            out.append(KO_MILK.get(v, v))
        elif k == "quantity":
            out.append(f"수량 {v}")
        else:
            out.append(f"{k}={v}")
    return ", ".join(out)


def describe_call(call: dict) -> str:
    name, args = call.get("name"), call.get("arguments") or {}
    if name == "add_item":
        return f"추가: {describe_item(args)}"
    if name == "modify_item":
        return f"{args.get('item_index')}번 항목 변경 → {describe_changes(args.get('changes'))}"
    if name == "remove_item":
        return f"{args.get('item_index')}번 항목 삭제"
    if name == "ask_clarification":
        return f"되묻기({KO_REASON.get(args.get('reason'), args.get('reason'))})"
    if name == "confirm_order":
        return "주문 확정"
    return f"{name}({args})"


def describe_calls(calls: list) -> str:
    return " / ".join(describe_call(c) for c in calls) if calls else "(도구 호출 없음)"


def format_order_state(state: list) -> str:
    if not state:
        return "(비어 있음)"
    return "\n".join(f"{i}. {describe_item(it)}" for i, it in enumerate(state, 1))


# ---------- system prompt ----------

def _item_line(it: dict) -> str:
    opts = []
    if it["temperatures"]:
        temps = "/".join(KO_TEMP[t] for t in it["temperatures"])
        opts.append(f"온도 {temps}(기본 {KO_TEMP[it['default_temperature']]})")
    else:
        opts.append("온도 없음")
    opts.append("사이즈 레귤러/라지" if len(it["sizes"]) > 1 else ("사이즈 선택 없음" if it["sizes"] else "사이즈 없음"))
    opts.append("샷 조절 가능" if it["shots"] else "샷 불가")
    if it["sweetness"]:
        opts.append("당도 조절 가능")
    if it["milk"]:
        opts.append("우유 일반/오트")
    return f"- {it['name']} [{load_menu()['categories'][it['category']]}]: " + ", ".join(opts)


def build_system_prompt(order_state: list | None = None) -> str:
    menu = load_menu()
    alias_lines = []
    for a in menu["aliases"]:
        imp = a.get("implies") or {}
        extra = f" ({KO_TEMP[imp['temperature']]})" if "temperature" in imp else ""
        alias_lines.append(f"{a['alias']}={a['menu']}{extra}")
    return "\n".join(
        [
            "너는 카페 주문 로봇의 언어 두뇌다. 손님의 마지막 말을 듣고, 반드시 제공된 도구를 호출해서 처리한다.",
            "",
            "규칙:",
            "- 메뉴 이름은 아래 메뉴판의 정식 이름만 쓴다.",
            "- 손님이 말하지 않은 옵션은 넣지 않는다. 말하지 않은 옵션은 기본값으로 처리되며, 그것 때문에 되묻지 않는다.",
            "- 서로 다른 메뉴(또는 옵션이 다른 같은 메뉴)는 add_item을 각각 호출한다. 같은 메뉴·같은 옵션 여러 개는 quantity로 표현한다.",
            "- 메뉴판 규칙상 불가능한 요청이면 그 메뉴를 담지 말고 ask_clarification(reason=\"impossible\")을 호출한다.",
            "- 메뉴판에 없는 메뉴면 ask_clarification(reason=\"off_menu\")을 호출한다.",
            "- 무엇을 원하는지, 또는 어느 항목을 말하는지 확실하지 않으면 ask_clarification(reason=\"ambiguous\")을 호출한다.",
            "- 이미 담긴 항목의 변경·취소는 현재 주문 내역의 번호(item_index, 1부터 시작)로 modify_item / remove_item을 호출한다. modify_item의 changes에는 바뀌는 값만 넣는다.",
            "- 손님이 주문을 마치겠다고 하면 confirm_order를 호출한다.",
            "",
            "메뉴판:",
            *[_item_line(it) for it in menu["items"]],
            "",
            "메뉴 규칙:",
            *[f"- {c}" for c in menu["constraints"]],
            "",
            "줄임말/다른 이름: " + ", ".join(alias_lines),
            "",
            "현재 주문 내역:",
            format_order_state(order_state or []),
        ]
    )
