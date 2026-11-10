"""Build data/calib_cafe.jsonl: 600 calibration samples for later pruning
(12 categories x 50, same categories as the exam).

Same task format as the exam, but generated from separate templates and
phrase banks. Any sample whose sentence is an exact/near duplicate of an exam
sentence, or whose intent matches an exam paraphrase group, is discarded
(check_overlap.py verifies the result independently).

    python scripts/build_calib_cafe.py
"""
from __future__ import annotations

import hashlib
import json
import random

from cafe_common import DATA, describe_item, items_by_name, load_menu
from check_overlap import ExamIndex, intent_key, load_jsonl

SEED = 42
PER_CATEGORY = 50
CATEGORIES = ["basic", "options", "slang", "multi_item", "modify_cancel", "impossible", "ambiguous",
              "off_menu", "call_staff", "takeout", "conditional", "unavailable"]

MENU = load_menu()
ITEMS = items_by_name()
DRINKS = [n for n, it in ITEMS.items() if it["category"] != "dessert"]
DESSERTS = [n for n, it in ITEMS.items() if it["category"] == "dessert"]
LATTES = [n for n, it in ITEMS.items() if it["category"] == "latte"]
COFFEES = [n for n, it in ITEMS.items() if it["category"] == "coffee"]

QTY = {1: ["한 잔", "1잔"], 2: ["두 잔", "2잔"], 3: ["세 잔", "3잔"], 4: ["네 잔", "4잔"]}
QTY_D = {1: ["한 개", "1개"], 2: ["두 개", "2개", "두 조각"], 3: ["세 개", "3조각"]}
TEMP = {"iced": ["아이스로", "얼음 넣어서", "시원하게", "차가운 걸로"], "hot": ["뜨겁게", "핫으로", "따뜻한 걸로", "데워서"]}
SIZE = {"large": ["큰 사이즈로", "사이즈 크게", "제일 큰 컵으로"], "regular": ["기본 사이즈로", "레귤러 사이즈로"]}
SHOTS = {1: ["샷 추가해서", "샷 하나 더 넣어서"], 2: ["샷 두 번 추가해서", "샷 2개 추가해서"], -1: ["샷 하나 줄여서"],
         0.5: ["샷 반 개 더 넣어서", "반 샷만 더 넣어서"], -0.5: ["샷 반 개 덜 넣어서", "샷을 반 개 줄여서"]}
SWEET = {"less": ["당도 낮춰서", "시럽 조금만 넣고", "단맛 줄여서"], "more": ["시럽 넉넉히 넣고", "당도 올려서", "좀 더 달달하게"]}
MILK = {"oat": ["우유는 오트로", "오트 우유로 해서"]}

FRAMES = [
    "{x} 부탁할게요", "저기요, {x} 가능할까요?", "{x} 시킬게요", "여기 {x} 주문이요",
    "{x} 받을 수 있을까요?", "{x} 준비해 주세요", "{x} 될까요?", "음, {x} 주문할게요",
    "일단 {x} 할게요", "안녕하세요, {x} 부탁드려요", "{x}로 결정했어요", "저는 {x}요",
]
JOINERS = [" 그리고 ", ", 그리고 ", "에다가 ", ", 또 ", " 하고 또 "]
SURFACE = {
    "카페라떼": ["카페라떼", "카페 라떼"], "바닐라라떼": ["바닐라라떼", "바닐라 라떼"],
    "카라멜마키아또": ["카라멜마키아또", "카라멜 마키아또"], "카페모카": ["카페모카", "카페 모카"],
    "초콜릿케이크": ["초콜릿케이크", "초콜릿 케이크"], "치즈케이크": ["치즈케이크", "치즈 케이크"],
    "콜드브루": ["콜드브루", "콜드 브루"],
}
PRIOR_ASSISTANT = ["주문 확인했어요: {items}.", "네, {items} 준비할게요.", "{items}, 받았습니다."]
CONFIRM_TEXTS = ["네 이걸로 확정할게요", "더 시킬 건 없어요", "주문 넣어 주세요", "이렇게 계산해 주세요",
                 "네 이게 전부예요", "이상으로 주문 마무리할게요", "그렇게 진행해 주세요", "이제 됐어요, 진행해 주세요"]
TAKEOUT_PHRASES = ["가지고 갈 거예요", "포장으로 부탁해요", "테이크아웃 할게요", "들고 나갈게요", "싸 주세요"]
DINE_IN_PHRASES = ["여기서 먹고 갈게요", "매장에서 마실게요", "먹고 갈 거예요"]
MOD_FRAMES = ["{ref} {chg} 바꿔 주실래요?", "{ref} 말인데요, {chg} 해 주세요", "아 {ref}는 {chg} 부탁드려요",
              "{ref} {chg} 수정할게요", "혹시 {ref} {chg} 변경 되나요?"]
RM_FRAMES = ["{ref}는 안 할게요", "{ref} 주문에서 빼 주실래요?", "{ref} 없애 주세요", "생각해 보니 {ref}는 필요 없어요",
             "{ref} 지워 주세요"]
CHG = {
    ("temperature", "iced"): ["아이스로", "차가운 걸로"], ("temperature", "hot"): ["따뜻한 걸로", "뜨거운 걸로"],
    ("size", "large"): ["큰 사이즈로", "라지 사이즈로"], ("size", "regular"): ["작은 사이즈로", "레귤러 사이즈로"],
    ("sweetness", "less"): ["당도 낮게"], ("sweetness", "more"): ["더 달달하게"], ("milk", "oat"): ["오트 우유로"],
    ("shots_delta", 0.5): ["샷 반 개 추가로"], ("shots_delta", -0.5): ["샷 반 개 덜 들어가게"], ("shots_delta", 1): ["샷 하나 추가로"],
}
OFF_MENU = ["아이스티", "핫초코", "밀크티", "버블티", "플랫화이트", "마카롱", "베이글", "스콘", "쿠키", "딸기라떼",
            "망고스무디", "바닐라쉐이크", "콜라", "오렌지주스", "티라미수", "크로플", "와플", "말차라떼", "고구마라떼",
            "페퍼민트티", "아인슈페너", "돌체라떼", "토스트", "생수"]
OFF_CATEGORY = {"딸기라떼": "latte", "말차라떼": "latte", "고구마라떼": "latte", "돌체라떼": "latte",
                "아이스티": "tea", "밀크티": "tea", "페퍼민트티": "tea", "플랫화이트": "coffee", "아인슈페너": "coffee",
                "티라미수": "dessert", "마카롱": "dessert"}
AMBIG_TOPICS = [("커피", "커피"), ("차 종류", "차"), ("음료", "음료"), ("디저트", "디저트"), ("시원한 음료", "시원한음료"),
                ("달달한 음료", "달달한음료"), ("아무 음료나", "아무거나"), ("제일 잘 나가는 메뉴", "추천"),
                ("추천해 주시는 걸로", "추천"), ("라떼 종류 중에 아무거나", "라떼종류")]
STAFF_TOPICS = {
    "bag": ["봉투 하나 받을 수 있나요?", "비닐봉투 좀 주세요", "쇼핑백에 담아 주실 수 있어요?"],
    "straw": ["빨대 하나 더 주실래요?", "빨대가 없어요"],
    "napkin": ["물티슈 좀 주실 수 있나요?", "냅킨 좀 더 주세요"],
    "restroom": ["화장실 어디예요?", "화장실 비밀번호 알려 주세요"],
    "wifi": ["와이파이 비밀번호가 뭐예요?", "여기 와이파이 돼요?"],
    "pickup_time": ["20분 뒤에 찾으러 올 건데 그때 맞춰 주실 수 있어요?", "3시 반에 픽업해도 될까요?"],
    "delivery": ["배달도 되나요?", "회사로 배달해 주실 수 있어요?"],
    "table": ["테이블 좀 닦아 주실 수 있나요?", "옆 테이블 좀 치워 주세요"],
    "charger": ["휴대폰 충전 좀 할 수 있을까요?", "충전기 빌릴 수 있나요?"],
    "carrier_alone": ["캐리어만 하나 주세요", "컵 캐리어 있나요?"],
    "cup_holder": ["컵홀더 하나 끼워 주세요", "컵 홀더 좀 주세요"],
}
STAFF_WITH_ORDER = {"bag": "봉투도 하나 주세요", "straw": "빨대 두 개 주세요", "napkin": "물티슈도 부탁해요",
                    "cup_holder": "컵홀더도 끼워 주세요"}
NOT_OFFERED = {x["key"]: x for x in MENU["not_offered"]}
NOT_OFFERED_PHRASE = {
    "decaf": ["디카페인으로 된 {m}", "{m} 디카페인으로"],
    "bean_selection": ["{m} 원두를 브라질 걸로", "{m} 원두 진한 걸로 골라서"],
    "whipped_cream": ["{m} 위에 휘핑 듬뿍 올려서", "휘핑 올린 {m}"],
    "lactose_free": ["{m} 락토프리 우유로", "유당 없는 우유로 만든 {m}"],
}
NOT_OFFERED_TARGETS = {"decaf": COFFEES + LATTES, "bean_selection": COFFEES, "whipped_cream": LATTES, "lactose_free": LATTES}


def call_add(menu, **o):
    return {"name": "add_item", "arguments": {"menu": menu, **o}}


def call_clar(reason, q):
    return {"name": "ask_clarification", "arguments": {"question": q, "reason": reason}}


def call_staff(reason):
    return {"name": "call_staff", "arguments": {"reason": reason}}


def call_confirm(opt=None):
    return {"name": "confirm_order", "arguments": {"dine_option": opt} if opt else {}}


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
        pool.append(("shots_delta", rng.choice([1, 2, -1, 0.5, -0.5])))
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


def add_args(o, q):
    return dict(o, **({"quantity": q} if q > 1 else {}))


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


def prior_turns(rng, state, user_text="주문할게요"):
    items = ", ".join(describe_item(s) for s in state)
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": rng.choice(PRIOR_ASSISTANT).format(items=items)},
    ]


def out(state, text, expected, intent=None, prior=None, tag=None):
    return {"state": state, "text": text, "expected": expected, "intent": intent, "prior": prior or [], "tag": tag}


# ---------------------------------------------------------------- generators


def gen_basic(rng):
    m = rng.choice(DRINKS + DESSERTS)
    q = rng.choice([1, 1, 2, 3])
    return out([], rng.choice(FRAMES).format(x=item_phrase(rng, m, {}, q)), [call_add(m, **add_args({}, q))])


def gen_options(rng):
    m = rng.choice(DRINKS)
    o = random_options(rng, m, rng.choice([1, 2, 2, 3]))
    if not o:
        return None
    q = rng.choice([1, 1, 2])
    return out([], rng.choice(FRAMES).format(x=item_phrase(rng, m, o, q)), [call_add(m, **add_args(o, q))])


def gen_slang(rng):
    a = rng.choice(MENU["aliases"])
    if " " in a["alias"]:
        return None
    m = a["menu"]
    extra = random_options(rng, m, rng.choice([0, 0, 1]))
    extra.pop("temperature", None)
    o = dict(a.get("implies") or {}, **extra)
    q = rng.choice([1, 1, 2, 3])
    x = item_phrase(rng, m, extra, q, menu_text=a["alias"])
    return out([], rng.choice(FRAMES).format(x=x), [call_add(m, **add_args(o, q))])


def gen_multi(rng):
    menus = rng.sample(DRINKS + DESSERTS, rng.choice([2, 2, 3]))
    phrases, calls = [], []
    for m in menus:
        o = random_options(rng, m, rng.choice([0, 1])) if m in DRINKS else {}
        q = rng.choice([1, 1, 2])
        phrases.append(item_phrase(rng, m, o, q))
        calls.append(call_add(m, **add_args(o, q)))
    text = phrases[0]
    for p in phrases[1:]:
        text += rng.choice(JOINERS) + p
    return out([], rng.choice(FRAMES).format(x=text), calls)


def ref_for(rng, state, i):
    if rng.random() < 0.4:
        return rng.choice([f"{i}번", f"{i}번 거", f"{i}번 메뉴"])
    return surface(rng, state[i - 1]["menu"])


def gen_modify(rng):
    state = random_state(rng, rng.choice([2, 3]))
    prior = prior_turns(rng, state) if rng.random() < 0.4 else []
    kind = rng.choice(["modify", "modify", "remove", "add"])
    if kind == "add":
        m = rng.choice([x for x in DRINKS + DESSERTS if x not in {s["menu"] for s in state}])
        q = rng.choice([1, 2])
        text = rng.choice(["{x}도 추가해 주세요", "아 그리고 {x}도요", "{x} 더 넣어 주세요"]).format(x=item_phrase(rng, m, {}, q))
        return out(state, text, [call_add(m, **add_args({}, q))], prior=prior)
    i = rng.randint(1, len(state))
    ref = ref_for(rng, state, i)
    if kind == "remove":
        return out(state, rng.choice(RM_FRAMES).format(ref=ref), [{"name": "remove_item", "arguments": {"item_index": i}}], prior=prior)
    cur = state[i - 1]
    it = ITEMS[cur["menu"]]
    options = []
    if len(it["temperatures"]) > 1:
        options.append(("temperature", "iced" if cur.get("temperature") == "hot" else "hot"))
    if len(it["sizes"]) > 1:
        options.append(("size", "regular" if cur.get("size") == "large" else "large"))
    if it["sweetness"]:
        options.append(("sweetness", rng.choice(["less", "more"])))
    if it["milk"]:
        options.append(("milk", "oat"))
    if it["shots"]:
        options.append(("shots_delta", rng.choice([0.5, -0.5, 1])))
    if not options:
        return None
    k, v = rng.choice(options)
    text = rng.choice(MOD_FRAMES).format(ref=ref, chg=rng.choice(CHG[(k, v)]))
    return out(state, text, [{"name": "modify_item", "arguments": {"item_index": i, "changes": {k: v}}}], prior=prior)


def gen_impossible(rng):
    cands = []
    for m, it in ITEMS.items():
        if it["category"] == "dessert":
            cands.append((m, "temperature", {"temperature": rng.choice(["hot", "iced"])}))
            cands.append((m, "size", {"size": "large"}))
            continue
        for t in ("hot", "iced"):
            if t not in it["temperatures"]:
                cands.append((m, "temperature", {"temperature": t}))
        if "large" not in it["sizes"]:
            cands.append((m, "size", {"size": "large"}))
        if not it["shots"]:
            cands.append((m, "shots", {"shots_delta": rng.choice([1, 0.5])}))
        if not it["sweetness"]:  # coffee + syrup (rule E)
            cands.append((m, "sweetness", {"sweetness": "more"}))
        if not it["milk"]:
            cands.append((m, "milk", {"milk": "oat"}))
    m, field, o = rng.choice(cands)
    x = item_phrase(rng, m, o, rng.choice([1, 1, 2]))
    q = "시럽은 셀프바에서 넣으실 수 있어요." if field == "sweetness" else f"{m}는 그 옵션이 안 돼요. 다르게 드릴까요?"
    return out([], rng.choice(FRAMES).format(x=x), [call_clar("impossible", q)], intent=f"impossible:{m}:{field}")


def gen_ambiguous(rng):
    r = rng.random()
    if r < 0.55:
        phrase, topic = rng.choice(AMBIG_TOPICS)
        x = phrase + rng.choice(["", " 하나", " 한 잔"])
        return out([], rng.choice(FRAMES).format(x=x), [call_clar("ambiguous", "어떤 메뉴로 드릴까요?")], intent=f"ambiguous:{topic}")
    if r < 0.8:
        state = random_state(rng, rng.choice([2, 3]))
        text = rng.choice(["같은 거 하나 더 주세요", "이거 큰 사이즈로 해 주실래요?", "그거 하나 빼 주세요", "하나만 아이스로 바꿔 주세요"])
        return out(state, text, [call_clar("ambiguous", "어떤 메뉴를 말씀하시는 걸까요?")], intent=f"ambiguous:which_item:{text}")
    m = rng.choice([x for x in DRINKS if len(ITEMS[x]["temperatures"]) > 1])
    state = [{"menu": m, "temperature": "hot", "size": "regular", "quantity": 1},
             {"menu": m, "temperature": "iced", "size": "regular", "quantity": 1}]
    rng.shuffle(state)
    text = rng.choice(RM_FRAMES).format(ref=surface(rng, m))
    return out(state, text, [call_clar("ambiguous", f"어느 {m}를 말씀하시는 걸까요?")], intent=f"ambiguous:duplicate_in_state:{m}")


def gen_off_menu(rng):
    off = rng.choice(OFF_MENU)
    x = off + rng.choice(["", " 하나", " 두 개"])
    return out([], rng.choice(FRAMES).format(x=x), [call_clar("off_menu", f"{off}는 메뉴에 없어요. 다른 메뉴는 어떠세요?")],
               intent=f"off_menu:{off}")


def gen_call_staff(rng):
    topic = rng.choice(list(STAFF_TOPICS))
    if topic in STAFF_WITH_ORDER and rng.random() < 0.35:
        m = rng.choice(DRINKS)
        q = rng.choice([1, 2])
        text = f"{item_phrase(rng, m, {}, q)} 주시고 {STAFF_WITH_ORDER[topic]}"
        return out([], text, [call_add(m, **add_args({}, q)), call_staff(f"{topic} 요청")], tag=f"multi+call_staff:{topic}")
    return out([], rng.choice(STAFF_TOPICS[topic]), [call_staff(f"{topic} 요청")], intent=f"call_staff:{topic}")


def gen_takeout(rng):
    kind = rng.choice(["add", "add", "earlier", "now_takeout", "now_dine_in", "plain"])
    if kind == "add":
        m = rng.choice(DRINKS + DESSERTS)
        o = random_options(rng, m, rng.choice([0, 1])) if m in DRINKS else {}
        q = rng.choice([1, 1, 2])
        x = item_phrase(rng, m, o, q)
        text = rng.choice(["{x} {t}", "{t}, {x}요", "{x} 주세요, {t}"]).format(x=x, t=rng.choice(TAKEOUT_PHRASES))
        return out([], text, [call_add(m, **add_args(o, q))], tag="takeout_mentioned")
    state = random_state(rng, rng.choice([1, 2]))
    if kind == "earlier":
        prior = prior_turns(rng, state, user_text=rng.choice(TAKEOUT_PHRASES))
        return out(state, rng.choice(CONFIRM_TEXTS), [call_confirm("takeout")], prior=prior, tag="takeout_earlier")
    if kind == "now_takeout":
        text = f"{rng.choice(CONFIRM_TEXTS)}, {rng.choice(TAKEOUT_PHRASES)}"
        return out(state, text, [call_confirm("takeout")], prior=prior_turns(rng, state), tag="takeout_now")
    if kind == "now_dine_in":
        text = f"{rng.choice(DINE_IN_PHRASES)}, {rng.choice(CONFIRM_TEXTS)}"
        return out(state, text, [call_confirm("dine_in")], prior=prior_turns(rng, state), tag="dine_in_now")
    return out(state, rng.choice(CONFIRM_TEXTS), [call_confirm()], prior=prior_turns(rng, state), tag="dine_not_mentioned")


def gen_conditional(rng):
    fallback_menu = rng.choice(DRINKS + DESSERTS)
    kind = rng.choice(["off_menu", "not_offered", "available", "any"])
    if kind == "available":
        x = rng.choice(DRINKS + DESSERTS)
        if x == fallback_menu:
            return None
        text = rng.choice(["{x} 가능하면 {x}로, 아니면 {y}로 할게요", "{x} 있다면 그걸로 하고 없다면 {y}요"]).format(
            x=surface(rng, x), y=surface(rng, fallback_menu))
        return out([], text, [call_add(x)], tag=f"conditional:{x}")
    if kind == "off_menu":
        x = rng.choice(OFF_MENU)
        text = rng.choice(["{x} 가능하면 그걸로, 아니면 {y}로 할게요", "{x} 있다면 그걸로 하고 없다면 {y}요"]).format(
            x=x, y=surface(rng, fallback_menu))
        return out([], text, [call_add(fallback_menu)], tag=f"conditional:{x}")
    if kind == "any":
        x = rng.choice([o for o in OFF_MENU if o in OFF_CATEGORY])
        cat = OFF_CATEGORY[x]
        word = {"latte": "라떼", "tea": "차", "coffee": "커피", "dessert": "디저트", "ade": "에이드"}[cat]
        text = rng.choice(["{x} 가능하면 그걸로, 아니면 아무 {w}나 주세요", "{x} 있다면 그걸로 하고 없다면 비슷한 {w}로요"]).format(x=x, w=word)
        return out([], text, [call_add(MENU["category_default_items"][cat])], tag=f"conditional:{x}:any")
    key = rng.choice(list(NOT_OFFERED_PHRASE))
    m = rng.choice(NOT_OFFERED_TARGETS[key])
    phrase = rng.choice(NOT_OFFERED_PHRASE[key]).format(m=surface(rng, m))
    text = rng.choice(["{p} 가능하면 그렇게 해 주시고, 아니면 그냥 일반 {m}로요", "{p} 된다면 좋겠고 안 된다면 그냥 기본으로요"]).format(
        p=phrase, m=surface(rng, m))
    return out([], text, [call_add(m)], tag=f"conditional:{key}:{m}")


def gen_unavailable(rng):
    key = rng.choice(list(NOT_OFFERED_PHRASE))
    m = rng.choice(NOT_OFFERED_TARGETS[key])
    phrase = rng.choice(NOT_OFFERED_PHRASE[key]).format(m=surface(rng, m))
    text = rng.choice(FRAMES).format(x=phrase)
    ko = NOT_OFFERED[key]["ko"]
    return out([], text, [call_clar("off_menu", f"죄송하지만 {ko}는 준비되어 있지 않아요.")], intent=f"off_menu:{key}:{m}")


GEN = {
    "basic": gen_basic, "options": gen_options, "slang": gen_slang, "multi_item": gen_multi,
    "modify_cancel": gen_modify, "impossible": gen_impossible, "ambiguous": gen_ambiguous, "off_menu": gen_off_menu,
    "call_staff": gen_call_staff, "takeout": gen_takeout, "conditional": gen_conditional, "unavailable": gen_unavailable,
}


def main():
    rng = random.Random(SEED)
    exam = load_jsonl(DATA / "exam_cafe.jsonl")
    idx = ExamIndex(exam)
    seen_text = set()
    samples, rejected = [], {"near_dup": 0, "intent": 0, "dup_text": 0}
    short = {}
    for cat in CATEGORIES:
        n, attempts = 0, 0
        while n < PER_CATEGORY and attempts < 30000:
            attempts += 1
            o = GEN[cat](rng)
            if o is None:
                continue
            conv = o["prior"] + [{"role": "user", "content": o["text"]}]
            if o["text"] in seen_text:
                rejected["dup_text"] += 1
                continue
            if any(idx.near_duplicate(m["content"]) for m in conv):
                rejected["near_dup"] += 1
                continue
            case = {"order_state": o["state"], "conversation": conv, "expected": o["expected"]}
            if o["intent"]:
                case["intent"] = o["intent"]
            if o["tag"]:
                case["intent_tag"] = o["tag"]
            key = intent_key(case)
            if key in idx.intents:
                rejected["intent"] += 1
                continue
            seen_text.add(o["text"])
            n += 1
            gid = "c-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
            samples.append({"id": f"calib-{len(samples) + 1:04d}", "category": cat, "paraphrase_group": gid, **case})
        if n < PER_CATEGORY:
            short[cat] = n
    (DATA / "calib_cafe.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n", encoding="utf-8")
    counts = {c: sum(1 for s in samples if s["category"] == c) for c in CATEGORIES}
    print(f"{len(samples)} samples -> data/calib_cafe.jsonl")
    print("per category:", counts)
    print("rejected during generation:", rejected)
    if short:
        print("WARNING: categories below target:", short)


if __name__ == "__main__":
    main()
