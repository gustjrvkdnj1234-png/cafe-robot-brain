"""Shared helpers for the cafe exam: menu loading, argument normalization,
Korean descriptions, and the system prompt given to models."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TOOL_NAMES = ["add_item", "modify_item", "remove_item", "ask_clarification", "call_staff", "confirm_order"]
ADD_KEYS = ["menu", "temperature", "size", "shots_delta", "sweetness", "milk", "quantity"]
REASONS = ["ambiguous", "impossible", "off_menu"]

KO_TEMP = {"hot": "따뜻한", "iced": "아이스"}
KO_SIZE = {"regular": "레귤러", "large": "라지"}
KO_SWEET = {"less": "덜 달게", "normal": "보통 당도", "more": "더 달게"}
KO_MILK = {"regular": "일반 우유", "oat": "오트밀크"}
KO_DINE = {"dine_in": "매장", "takeout": "포장"}
KO_REASON = {"ambiguous": "무엇을 원하는지 불분명", "impossible": "불가능한 옵션 조합", "off_menu": "제공하지 않는 메뉴/옵션"}


@lru_cache(maxsize=1)
def load_menu() -> dict:
    return json.loads((DATA / "menu.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_tools() -> list:
    return json.loads((DATA / "tools.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def items_by_name() -> dict:
    return {it["name"]: it for it in load_menu()["items"]}


def defaults() -> dict:
    return load_menu()["defaults"]


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


def _as_num(v, default=None):
    """Numbers in 0.5 steps; integral values become int so 1 == 1.0 in JSON too."""
    if v is None or v == "":
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() else f


def _as_str(v):
    return v.strip().lower() if isinstance(v, str) else v


def normalize_add(args: dict) -> dict:
    """Fill defaults so that an omitted option equals its default value."""
    args = dict(args or {})
    d = defaults()
    menu = canonical_menu(args.get("menu"))
    item = items_by_name().get(menu)
    out = {"menu": menu}
    temp = _as_str(args.get("temperature"))
    size = _as_str(args.get("size"))
    if item is not None:
        out["temperature"] = temp or item["default_temperature"]
        out["size"] = size or (d["size"] if item["sizes"] else None)
        if not item["sizes"] and size == d["size"]:
            out["size"] = None
    else:
        out["temperature"] = temp
        out["size"] = size or d["size"]
    out["shots_delta"] = _as_num(args.get("shots_delta"), d["shots_delta"])
    out["sweetness"] = _as_str(args.get("sweetness")) or d["sweetness"]
    out["milk"] = _as_str(args.get("milk")) or d["milk"]
    out["quantity"] = _as_int(args.get("quantity"), d["quantity"])
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
        elif k == "quantity":
            out[k] = _as_int(v)
        elif k == "shots_delta":
            out[k] = _as_num(v)
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
    elif name == "confirm_order":
        key = {"dine_option": _as_str(args.get("dine_option")) or defaults()["dine_option"]}
    else:  # call_staff: the free-text reason is ignored
        key = {}
    return {"name": name, "args": key}


# ---------- Korean descriptions ----------

def _shots_ko(v) -> str:
    v = _as_num(v)
    return f"샷 {v:+g}" if isinstance(v, (int, float)) else f"샷 {v}"


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
        opts.append(_shots_ko(a["shots_delta"]))
    if a["sweetness"] != "normal":
        opts.append(KO_SWEET.get(a["sweetness"], str(a["sweetness"])))
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
            out.append(_shots_ko(v))
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
        return f"되묻기/거절({KO_REASON.get(args.get('reason'), args.get('reason'))})"
    if name == "call_staff":
        return f"직원 호출({args.get('reason', '')})"
    if name == "confirm_order":
        opt = args.get("dine_option") or defaults()["dine_option"]
        return f"주문 확정({KO_DINE.get(opt, opt)})"
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
    if len(it["sizes"]) > 1:
        opts.append("사이즈 레귤러/라지(기본 레귤러)")
    else:
        opts.append("사이즈 선택 없음")
    opts.append("샷 조절 가능(0.5 단위)" if it["shots"] else "샷 불가")
    opts.append("당도 조절 가능(기본 보통)" if it["sweetness"] else "당도·시럽 조절 불가")
    if it["milk"]:
        opts.append("우유 일반/오트(기본 일반)")
    return f"- {it['name']} [{load_menu()['categories'][it['category']]}]: " + ", ".join(opts)


def build_system_prompt(order_state: list | None = None) -> str:
    menu = load_menu()
    cats = menu["categories"]
    cat_defaults = ", ".join(f"{cats[c]}={m}" for c, m in menu["category_default_items"].items())
    not_offered = ", ".join(x["ko"] for x in menu["not_offered"])
    alias_lines = []
    for a in menu["aliases"]:
        imp = a.get("implies") or {}
        extra = f" ({KO_TEMP[imp['temperature']]})" if "temperature" in imp else ""
        alias_lines.append(f"{a['alias']}={a['menu']}{extra}")
    return "\n".join(
        [
            "너는 카페 주문 로봇의 언어 두뇌다. 손님의 마지막 말을 듣고, 반드시 제공된 도구를 하나 이상 호출해서 처리한다.",
            "",
            "[기본값]",
            "손님이 말하지 않은 옵션은 묻지 말고 기본값을 쓴다. (Don't ask about options the customer didn't mention; use defaults.)",
            "- 온도: 따뜻하게(hot). 단, 콜드브루와 아이스만 되는 메뉴(에이드)는 아이스(iced). 디저트는 온도 없음.",
            "- 사이즈: 레귤러(regular)",
            "- 샷: 기본 그대로(shots_delta 0)",
            "- 당도: 보통(normal)",
            "- 우유: 일반 우유(regular)",
            "- 수량: 1",
            "- 매장/포장: 손님이 말하지 않으면 매장(dine_in)",
            "기본값과 같은 옵션은 도구 인자에서 빼도 된다.",
            "",
            "[주문 처리]",
            "- 메뉴 이름은 아래 메뉴판의 정식 이름만 쓴다.",
            "- 서로 다른 메뉴(또는 옵션이 다른 같은 메뉴)는 add_item을 각각 호출한다. 같은 메뉴·같은 옵션 여러 개는 quantity로 표현한다.",
            "- 샷은 0.5(반 샷) 단위로 더하거나 뺄 수 있다(shots_delta 예: 0.5, 1, -0.5).",
            "- 이미 담긴 항목의 변경·취소는 현재 주문 내역의 번호(item_index, 1부터 시작)로 modify_item / remove_item을 호출한다. changes에는 바뀌는 값만 넣는다.",
            "- 손님이 주문을 마치겠다고 하면 confirm_order를 호출한다. 대화 중 한 번이라도 포장(포장, 테이크아웃, 가져갈게요 등)이라고 했으면 dine_option=\"takeout\", 매장이라고 했거나 말이 없었으면 \"dine_in\".",
            "- 포장 여부는 먼저 묻지 않는다. 주문하면서 포장이라고 하면 그대로 받아들이고 주문을 계속 처리한다.",
            "",
            "[되묻기·거절: ask_clarification]",
            "- reason=\"ambiguous\": 무엇을 원하는지, 또는 현재 주문의 어느 항목을 말하는지 확실하지 않을 때.",
            "- reason=\"impossible\": 메뉴판에 있는 옵션이지만 그 메뉴에는 안 되는 조합일 때(예: 따뜻한 에이드, 차에 샷 추가, 에스프레소 라지·아이스, 라떼가 아닌 메뉴에 오트밀크, 디저트에 온도·사이즈). 그 메뉴는 담지 않는다.",
            "- 커피(에스프레소, 아메리카노, 콜드브루)에 시럽이나 단맛을 원하면 reason=\"impossible\"로 답하고, 시럽은 셀프바에서 넣을 수 있다고 안내한다.",
            f"- reason=\"off_menu\": 우리 카페가 아예 제공하지 않는 메뉴나 옵션일 때. 메뉴판에 없는 음료·음식과 다음 옵션은 제공하지 않는다: {not_offered}. 정중하게 안 된다고 안내한다.",
            "",
            "[조건부 요청]",
            "- \"X 있으면 X, 없으면 Y\"처럼 말하면: X가 가능하면 X를 담는다. X가 안 되면(메뉴에 없음, 불가능한 조합, 제공하지 않는 옵션) 되묻지 말고 Y를 담는다.",
            f"- Y가 \"아무거나\", \"비슷한 걸로\"처럼 구체적이지 않으면 그 종류의 기본 메뉴를 담는다: {cat_defaults}.",
            "- \"안 되면 그냥 그걸로\"처럼 안 되는 옵션만 빼라는 뜻이면 그 옵션 없이 같은 메뉴를 담는다.",
            "",
            "[직원 호출: call_staff]",
            "로봇이 정하거나 약속하면 안 되는 일은 call_staff로 직원에게 넘긴다:",
            *[f"- {s}" for s in menu["staff_only"]],
            "메뉴 주문과 함께 말하면 주문은 해당 도구로 처리하고 call_staff도 함께 호출한다.",
            "",
            "[메뉴판]",
            *[_item_line(it) for it in menu["items"]],
            "",
            "[메뉴 규칙]",
            *[f"- {c}" for c in menu["constraints"]],
            f"- 제공하지 않음: {not_offered}",
            f"- 셀프바: {menu['self_service_bar']}",
            "",
            "[줄임말/다른 이름]",
            ", ".join(alias_lines),
            "",
            "[현재 주문 내역]",
            format_order_state(order_state or []),
        ]
    )
