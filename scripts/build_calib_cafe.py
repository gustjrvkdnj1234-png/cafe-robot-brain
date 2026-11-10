"""Build data/calib_cafe.jsonl: ~600 calibration samples for later pruning.

Same task format as the exam, but generated from separate templates and
phrase banks. Any sample whose sentence is an exact/near duplicate of an exam
sentence, or whose intent matches an exam paraphrase group, is discarded
(see check_overlap.py, which verifies the result independently).

    python scripts/build_calib_cafe.py
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from cafe_common import DATA, describe_item, items_by_name, load_menu
from check_overlap import ExamIndex, intent_key, load_jsonl

SEED = 42
PER_CATEGORY = 75
CATEGORIES = ["basic", "options", "slang", "multi_item", "modify_cancel", "impossible", "ambiguous", "off_menu"]

ITEMS = items_by_name()
DRINKS = [n for n, it in ITEMS.items() if it["category"] != "dessert"]
DESSERTS = [n for n, it in ITEMS.items() if it["category"] == "dessert"]

QTY = {1: ["한 잔", "1잔"], 2: ["두 잔", "2잔"], 3: ["세 잔", "3잔"], 4: ["네 잔", "4잔"]}
QTY_D = {1: ["한 개", "1개"], 2: ["두 개", "2개", "두 조각"], 3: ["세 개", "3조각"]}
TEMP = {"iced": ["아이스로", "얼음 넣어서", "시원하게", "차가운 걸로"], "hot": ["뜨겁게", "핫으로", "따뜻한 걸로", "데워서"]}
SIZE = {"large": ["큰 사이즈로", "사이즈 크게", "제일 큰 컵으로"], "regular": ["기본 사이즈로", "레귤러 사이즈로"]}
SHOTS = {1: ["샷 추가해서", "샷 하나 더 넣어서"], 2: ["샷 두 번 추가해서", "샷 2개 추가해서"], -1: ["샷 하나 줄여서"]}
SWEET = {"less": ["당도 낮춰서", "시럽 조금만 넣고", "단맛 줄여서"], "more": ["시럽 넉넉히 넣고", "당도 올려서", "좀 더 달달하게"]}
MILK = {"oat": ["우유는 오트로", "오트 우유로 해서"]}

FRAMES = [
    "{x} 부탁할게요", "저기요, {x} 가능할까요?", "{x} 시킬게요", "여기 {x} 주문이요",
    "{x} 받을 수 있을까요?", "{x} 준비해 주세요", "{x} 될까요?", "음, {x} 주문할게요",
    "일단 {x} 할게요", "안녕하세요, {x} 부탁드려요", "{x}로 결정했어요", "저는 {x}요",
]
JOINERS = [" 그리고 ", ", 그리고 ", "에다가 ", ", 또 ", " 하고 또 "]
SURFACE = {  # extra spellings that are still the canonical item (no slang)
    "카페라떼": ["카페라떼", "카페 라떼"], "바닐라라떼": ["바닐라라떼", "바닐라 라떼"],
    "카라멜마키아또": ["카라멜마키아또", "카라멜 마키아또"], "카페모카": ["카페모카", "카페 모카"],
    "초콜릿케이크": ["초콜릿케이크", "초콜릿 케이크"], "치즈케이크": ["치즈케이크", "치즈 케이크"],
    "콜드브루": ["콜드브루", "콜드 브루"],
}
PRIOR_ASSISTANT = ["주문 확인했어요: {items}.", "네, {items} 준비할게요.", "{items}, 받았습니다."]
CONFIRM_TEXTS = ["네 이걸로 확정할게요", "더 시킬 건 없어요", "주문 넣어 주세요", "이렇게 계산해 주세요",
                 "네 이게 전부예요", "이상으로 주문 마칠게요", "그렇게 진행해 주세요", "이제 됐어요, 진행해 주세요"]
MOD_FRAMES = ["{ref} {chg} 바꿔 주실래요?", "{ref} 말인데요, {chg} 해 주세요", "아 {ref}는 {chg} 부탁드려요",
              "{ref} {chg} 수정할게요", "혹시 {ref} {chg} 변경 되나요?"]
RM_FRAMES = ["{ref}는 안 할게요", "{ref} 주문에서 빼 주실래요?", "{ref} 없애 주세요", "생각해 보니 {ref}는 필요 없어요",
             "{ref} 지워 주세요"]
CHG = {
    ("temperature", "iced"): ["아이스로", "차가운 걸로"], ("temperature", "hot"): ["따뜻한 걸로", "뜨거운 걸로"],
    ("size", "large"): ["큰 사이즈로", "라지 사이즈로"], ("size", "regular"): ["작은 사이즈로", "레귤러 사이즈로"],
    ("sweetness", "less"): ["당도 낮게"], ("sweetness", "more"): ["더 달달하게"], ("milk", "oat"): ["오트 우유로"],
}
OFF_MENU = ["아이스티", "핫초코", "밀크티", "버블티", "플랫화이트", "마카롱", "베이글", "스콘", "쿠키", "딸기라떼",
            "망고스무디", "바닐라쉐이크", "콜라", "오렌지주스", "티라미수", "크로플", "와플", "말차라떼", "고구마라떼",
            "페퍼민트티", "아인슈페너", "돌체라떼", "토스트", "생수"]
AMBIG_TOPICS = [("커피", "커피"), ("차 종류", "차"), ("음료", "음료"), ("디저트", "디저트"), ("시원한 음료", "시원한음료"),
                ("달달한 음료", "달달한음료"), ("아무 음료나", "아무거나"), ("제일 잘 나가는 메뉴", "추천"),
                ("추천해 주시는 걸로", "추천"), ("라떼 종류 중에 아무거나", "라떼종류")]


def call_add(menu, **o):
    return {"name": "add_item", "arguments": {"menu": menu, **o}}


def call_clar(reason, q):
    return {"name": "ask_clarification", "arguments": {"question": q, "reason": reason}}


def surface(rng, menu):
    return rng.choice(SURFACE.get(menu, [menu]))


def qty_text(rng, menu, q):
    bank = QTY_D if ITEMS[menu]["category"] == "dessert" else QTY
    return rng.choice(bank[min(q, max(bank))])


def random_options(rng, menu, k):
    it = ITEMS[menu]
    pool = []
    if len(it["temperatures"]) > 1:
        pool.append(("temperature", rng.choice(it["temperatures"])))
    if "large" in it["sizes"]:
        pool.append(("size", "large"))
    if it["shots"]:
        pool.append(("shots_delta", rng.choice([1, 2, -1])))
    if it["sweetness"]:
        pool.append(("sweetness", rng.choice(["less", "more"])))
    if it["milk"]:
        pool.append(("milk", "oat"))
    rng.shuffle(pool)
    return dict(pool[:k])


def option_phrases(rng, o):
    bank = {"temperature": TEMP, "size": SIZE, "shots_delta": SHOTS, "sweetness": SWEET, "milk": MILK}
    return [rng.choice(bank[k][v]) for k, v in o.items()]


def item_phrase(rng, menu, o, q, menu_text=None):
    parts = [menu_text or surface(rng, menu)] + option_phrases(rng, o)
    if q != 1 or rng.random() < 0.5:
        parts.append(qty_text(rng, menu, q))
    return " ".join(parts)


def random_state(rng, n):
    menus = rng.sample(DRINKS + DESSERTS, n)
    state = []
    for m in menus:
        it = ITEMS[m]
        s = {"menu": m, "quantity": rng.choice([1, 1, 2])}
        if it["temperatures"]:
            s["temperature"] = rng.choice(it["temperatures"])
        if it["sizes"]:
            s["size"] = rng.choice(it["sizes"])
        state.append(s)
    return state


def prior_turns(rng, state):
    items = ", ".join(describe_item(s) for s in state)
    return [
        {"role": "user", "content": "주문할게요"},
        {"role": "assistant", "content": rng.choice(PRIOR_ASSISTANT).format(items=items)},
    ]


# ---------------------------------------------------------------- generators


def gen_basic(rng):
    m = rng.choice(DRINKS + DESSERTS)
    q = rng.choice([1, 1, 2, 3])
    x = item_phrase(rng, m, {}, q)
    return [], rng.choice(FRAMES).format(x=x), [call_add(m, **({"quantity": q} if q > 1 else {}))], None


def gen_options(rng):
    m = rng.choice(DRINKS)
    o = random_options(rng, m, rng.choice([1, 2, 2, 3]))
    if not o:
        return None
    q = rng.choice([1, 1, 2])
    x = item_phrase(rng, m, o, q)
    args = dict(o, **({"quantity": q} if q > 1 else {}))
    return [], rng.choice(FRAMES).format(x=x), [call_add(m, **args)], None


def gen_slang(rng):
    a = rng.choice(load_menu()["aliases"])
    if " " in a["alias"]:
        return None
    m = a["menu"]
    o = dict(a.get("implies") or {})
    extra = random_options(rng, m, rng.choice([0, 0, 1]))
    extra.pop("temperature", None)
    o.update(extra)
    q = rng.choice([1, 1, 2, 3])
    x = item_phrase(rng, m, extra, q, menu_text=a["alias"])
    args = dict(o, **({"quantity": q} if q > 1 else {}))
    return [], rng.choice(FRAMES).format(x=x), [call_add(m, **args)], None


def gen_multi(rng):
    menus = rng.sample(DRINKS + DESSERTS, rng.choice([2, 2, 3]))
    phrases, calls = [], []
    for m in menus:
        o = random_options(rng, m, rng.choice([0, 1])) if m in DRINKS else {}
        q = rng.choice([1, 1, 2])
        phrases.append(item_phrase(rng, m, o, q))
        calls.append(call_add(m, **dict(o, **({"quantity": q} if q > 1 else {}))))
    text = phrases[0]
    for p in phrases[1:]:
        text += rng.choice(JOINERS) + p
    return [], rng.choice(FRAMES).format(x=text), calls, None


def ref_for(rng, state, i):
    if rng.random() < 0.4:
        return rng.choice([f"{i}번", f"{i}번 거", f"{i}번 메뉴"])
    return surface(rng, state[i - 1]["menu"])


def gen_modify(rng):
    state = random_state(rng, rng.choice([2, 3]))
    prior = prior_turns(rng, state) if rng.random() < 0.4 else []
    kind = rng.choice(["modify", "modify", "remove", "confirm", "add"])
    if kind == "confirm":
        return state, rng.choice(CONFIRM_TEXTS), [{"name": "confirm_order", "arguments": {}}], None, prior
    if kind == "add":
        m = rng.choice([x for x in DRINKS + DESSERTS if x not in {s["menu"] for s in state}])
        q = rng.choice([1, 2])
        x = item_phrase(rng, m, {}, q)
        text = rng.choice(["{x}도 추가해 주세요", "아 그리고 {x}도요", "{x} 더 넣어 주세요"]).format(x=x)
        return state, text, [call_add(m, **({"quantity": q} if q > 1 else {}))], None, prior
    i = rng.randint(1, len(state))
    ref = ref_for(rng, state, i)
    if kind == "remove":
        return state, rng.choice(RM_FRAMES).format(ref=ref), [{"name": "remove_item", "arguments": {"item_index": i}}], None, prior
    it = ITEMS[state[i - 1]["menu"]]
    options = []
    if len(it["temperatures"]) > 1:
        options.append(("temperature", "iced" if state[i - 1].get("temperature") == "hot" else "hot"))
    if len(it["sizes"]) > 1:
        options.append(("size", "regular" if state[i - 1].get("size") == "large" else "large"))
    if it["sweetness"]:
        options.append(("sweetness", rng.choice(["less", "more"])))
    if it["milk"]:
        options.append(("milk", "oat"))
    if not options:
        return None
    k, v = rng.choice(options)
    chg = rng.choice(CHG[(k, v)])
    text = rng.choice(MOD_FRAMES).format(ref=ref, chg=chg)
    return state, text, [{"name": "modify_item", "arguments": {"item_index": i, "changes": {k: v}}}], None, prior


EXAM_ONLY_IMPOSSIBLE = {("레몬에이드", "temperature"), ("얼그레이", "shots"), ("에스프레소", "size"), ("아메리카노", "milk")}


def gen_impossible(rng):
    cands = []
    for m, it in ITEMS.items():
        if "hot" not in it["temperatures"] and it["category"] != "dessert":
            cands.append((m, "temperature", {"temperature": "hot"}))
        if "iced" not in it["temperatures"] and it["category"] != "dessert":
            cands.append((m, "temperature", {"temperature": "iced"}))
        if it["category"] == "dessert":
            cands.append((m, "temperature", {"temperature": rng.choice(["hot", "iced"])}))
            cands.append((m, "size", {"size": "large"}))
        elif "large" not in it["sizes"]:
            cands.append((m, "size", {"size": "large"}))
        if not it["shots"]:
            cands.append((m, "shots", {"shots_delta": 1}))
        if not it["sweetness"] and it["category"] != "dessert":
            cands.append((m, "sweetness", {"sweetness": rng.choice(["less", "more"])}))
        if not it["milk"] and it["category"] != "dessert":
            cands.append((m, "milk", {"milk": "oat"}))
    cands = [c for c in cands if (c[0], c[1]) not in EXAM_ONLY_IMPOSSIBLE]
    m, field, o = rng.choice(cands)
    x = item_phrase(rng, m, o, rng.choice([1, 1, 2]))
    return [], rng.choice(FRAMES).format(x=x), [call_clar("impossible", f"{m}는 그 옵션이 안 돼요. 다르게 드릴까요?")], f"impossible:{m}:{field}"


def gen_ambiguous(rng):
    r = rng.random()
    if r < 0.55:
        phrase, topic = rng.choice(AMBIG_TOPICS)
        x = phrase + rng.choice(["", " 하나", " 한 잔"])
        return [], rng.choice(FRAMES).format(x=x), [call_clar("ambiguous", "어떤 메뉴로 드릴까요?")], f"ambiguous:{topic}"
    if r < 0.8:
        state = random_state(rng, rng.choice([2, 3]))
        text = rng.choice(["같은 거 하나 더 주세요", "이거 큰 사이즈로 해 주실래요?", "그거 하나 빼 주세요", "하나만 아이스로 바꿔 주세요"])
        return state, text, [call_clar("ambiguous", "어떤 메뉴를 말씀하시는 걸까요?")], f"ambiguous:which_item:{text}"
    m = rng.choice([x for x in DRINKS if len(ITEMS[x]["temperatures"]) > 1 and x != "아메리카노"])
    state = [{"menu": m, "temperature": "hot", "size": "regular", "quantity": 1},
             {"menu": m, "temperature": "iced", "size": "regular", "quantity": 1}]
    rng.shuffle(state)
    text = rng.choice(RM_FRAMES).format(ref=surface(rng, m))
    return state, text, [call_clar("ambiguous", f"어느 {m}를 말씀하시는 걸까요?")], f"ambiguous:duplicate_in_state:{m}"


def gen_off_menu(rng):
    off = rng.choice(OFF_MENU)
    x = off + rng.choice(["", " 하나", " 두 개"])
    return [], rng.choice(FRAMES).format(x=x), [call_clar("off_menu", f"{off}는 메뉴에 없어요. 다른 메뉴는 어떠세요?")], f"off_menu:{off}"


GEN = {
    "basic": gen_basic, "options": gen_options, "slang": gen_slang, "multi_item": gen_multi,
    "modify_cancel": gen_modify, "impossible": gen_impossible, "ambiguous": gen_ambiguous, "off_menu": gen_off_menu,
}


def main():
    rng = random.Random(SEED)
    exam = load_jsonl(DATA / "exam_cafe.jsonl")
    idx = ExamIndex(exam)
    seen_text = set()
    samples, rejected = [], {"near_dup": 0, "intent": 0, "dup_text": 0}
    for cat in CATEGORIES:
        n, attempts = 0, 0
        while n < PER_CATEGORY and attempts < 20000:
            attempts += 1
            out = GEN[cat](rng)
            if out is None:
                continue
            if len(out) == 4:
                state, text, expected, intent = out
                prior = []
            else:
                state, text, expected, intent, prior = out
            conv = prior + [{"role": "user", "content": text}]
            if text in seen_text:
                rejected["dup_text"] += 1
                continue
            if any(idx.near_duplicate(m["content"]) for m in conv):
                rejected["near_dup"] += 1
                continue
            case = {"order_state": state, "conversation": conv, "expected": expected}
            if intent:
                case["intent"] = intent
            key = intent_key(case)
            if key in idx.intents:
                rejected["intent"] += 1
                continue
            seen_text.add(text)
            n += 1
            gid = "c-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
            samples.append({"id": f"calib-{len(samples) + 1:04d}", "category": cat, "paraphrase_group": gid, **case})
    out = DATA / "calib_cafe.jsonl"
    out.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n", encoding="utf-8")
    counts = {c: sum(1 for s in samples if s["category"] == c) for c in CATEGORIES}
    print(f"{len(samples)} samples -> data/calib_cafe.jsonl")
    print("per category:", counts)
    print("rejected during generation:", rejected)


if __name__ == "__main__":
    main()
