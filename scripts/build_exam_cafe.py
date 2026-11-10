"""Build data/exam_cafe.jsonl (120 hand-written cases in 37 paraphrase groups,
12 categories) and data/exam_cafe_review.md (20 sampled cases for manual checking).

    python scripts/build_exam_cafe.py
"""
from __future__ import annotations

import json
import random

from cafe_common import DATA, describe_calls, format_order_state

CATEGORY_KO = {
    "basic": "기본 주문",
    "options": "옵션(반 샷 포함)",
    "slang": "줄임말/은어",
    "multi_item": "여러 메뉴",
    "modify_cancel": "변경/취소",
    "impossible": "불가능한 조합(커피 시럽 포함)",
    "ambiguous": "모호한 요청",
    "off_menu": "메뉴에 없음",
    "call_staff": "직원 호출",
    "takeout": "포장/매장",
    "conditional": "조건부 요청",
    "unavailable": "제공하지 않는 옵션",
}


def add(menu, **o):
    return {"name": "add_item", "arguments": {"menu": menu, **o}}


def mod(i, **c):
    return {"name": "modify_item", "arguments": {"item_index": i, "changes": c}}


def rm(i):
    return {"name": "remove_item", "arguments": {"item_index": i}}


def clar(reason, q):
    return {"name": "ask_clarification", "arguments": {"question": q, "reason": reason}}


def staff(reason):
    return {"name": "call_staff", "arguments": {"reason": reason}}


def confirm(dine_option=None):
    return {"name": "confirm_order", "arguments": {"dine_option": dine_option} if dine_option else {}}


STATE_A = [
    {"menu": "아메리카노", "temperature": "iced", "size": "regular", "quantity": 1},
    {"menu": "카페라떼", "temperature": "hot", "size": "regular", "quantity": 1},
]
PRIOR_A = [
    {"role": "user", "content": "아아 하나랑 따뜻한 라떼 하나요"},
    {"role": "assistant", "content": "아이스 아메리카노 1잔, 따뜻한 카페라떼 1잔 담았습니다."},
]
PRIOR_TAKEOUT = [
    {"role": "user", "content": "아아 하나랑 따뜻한 라떼 하나 포장이요"},
    {"role": "assistant", "content": "포장으로 아이스 아메리카노 1잔, 따뜻한 카페라떼 1잔 담았습니다."},
]
STATE_B = [
    {"menu": "바닐라라떼", "temperature": "hot", "size": "regular", "quantity": 1},
    {"menu": "치즈케이크", "quantity": 1},
]
STATE_C = [
    {"menu": "아메리카노", "temperature": "hot", "size": "regular", "quantity": 1},
    {"menu": "아메리카노", "temperature": "iced", "size": "large", "quantity": 1},
]

# (group, category, expected, variants[(style, text)], extra)
#   extra: order_state, prior, intent (explicit label) or intent_tag (label + expected signature)
GROUPS = [
    # ---------------- basic (10)
    ("B1", "basic", [add("아메리카노")], [
        ("formal", "아메리카노 한 잔 주시겠어요?"),
        ("casual", "아메리카노 하나요"),
        ("slang", "아메 하나 ㄱㄱ"),
        ("typo", "아메리까노 한잔 주세여")], {}),
    ("B2", "basic", [add("치즈케이크")], [
        ("formal", "치즈케이크 한 조각 주시겠어요?"),
        ("casual", "치즈케이크 하나요"),
        ("typo", "치즈캐이크 하나 주세여")], {}),
    ("B3", "basic", [add("유자차")], [
        ("formal", "유자차 한 잔 주실 수 있을까요?"),
        ("casual", "유자차 하나 줘요"),
        ("typo", "유쟈차 하나요")], {}),

    # ---------------- options (10), incl. half shots (B)
    ("O1", "options", [add("아메리카노", temperature="iced", size="large")], [
        ("formal", "아이스 아메리카노 라지 사이즈로 한 잔 주세요."),
        ("casual", "아이스 아메리카노 큰 걸로 하나요"),
        ("slang", "아아 라지 하나"),
        ("typo", "아이스 아메리카노 라쥐로 하나요")], {}),
    ("O2", "options", [add("바닐라라떼", sweetness="less", shots_delta=1)], [
        ("formal", "바닐라라떼 덜 달게 해서 샷 하나 추가해 주세요."),
        ("slang", "바라 샷추 덜달"),
        ("typo", "바닐라라때 덜 달게 샷 추가요")], {}),
    ("O3", "options", [add("카페라떼", temperature="iced", shots_delta=0.5)], [
        ("formal", "아이스 카페라떼에 샷 반 개만 추가해 주세요."),
        ("casual", "아이스 라떼 반 샷 추가요"),
        ("slang", "아이스라떼 샷 0.5 추가 ㄱ")], {}),

    # ---------------- slang (10)
    ("S1", "slang", [add("아메리카노", temperature="iced", quantity=2)], [
        ("formal", "아이스 아메리카노 두 잔 부탁드립니다."),
        ("casual", "아아 두 잔이요"),
        ("slang", "아아 2"),
        ("typo", "아아 두잔여")], {}),
    ("S2", "slang", [add("바닐라라떼", temperature="iced")], [
        ("formal", "아이스 바닐라라떼 한 잔 주세요."),
        ("casual", "아바라 하나요"),
        ("typo", "아바러 하나")], {}),
    ("S3", "slang", [add("카라멜마키아또")], [
        ("formal", "카라멜 마키아또 한 잔 주세요."),
        ("slang", "카라마끼 하나요"),
        ("typo", "캬라멜 마끼아또 하나")], {}),

    # ---------------- multi_item (9)
    ("M1", "multi_item", [add("아메리카노", temperature="iced"), add("치즈케이크")], [
        ("formal", "아이스 아메리카노 한 잔이랑 치즈케이크 한 조각 주세요."),
        ("casual", "아이스 아메리카노하고 치즈케이크 하나씩이요"),
        ("slang", "아아 하나랑 치케 하나")], {}),
    ("M2", "multi_item", [add("카페라떼", quantity=2), add("레몬에이드")], [
        ("formal", "카페라떼 두 잔하고 레몬에이드 한 잔 주세요."),
        ("slang", "라떼 2 레몬에이드 1"),
        ("typo", "카페라떼 두잔이랑 레몬에이드 하나여")], {}),
    ("M3", "multi_item", [add("아메리카노", temperature="iced", quantity=2), staff("캐리어 요청")], [
        ("formal", "아이스 아메리카노 두 잔이랑 캐리어 하나 부탁드려요."),
        ("casual", "아아 둘에 캐리어요"),
        ("typo", "아아 2잔 캐리어에 담아주세여")], {"intent_tag": "multi+call_staff:carrier"}),

    # ---------------- modify_cancel (9)
    ("X1", "modify_cancel", [mod(1, size="large")], [
        ("formal", "첫 번째 아메리카노를 라지로 바꿔 주시겠어요?"),
        ("casual", "아아 큰 걸로 바꿔줘"),
        ("typo", "아메리까노 라지로 변경해주세여")], {"order_state": STATE_A, "prior": PRIOR_A}),
    ("X2", "modify_cancel", [rm(2)], [
        ("formal", "카페라떼는 취소할게요."),
        ("casual", "라떼 빼줘"),
        ("ordinal", "두 번째 메뉴 취소해 주세요.")], {"order_state": STATE_A, "prior": PRIOR_A}),
    ("X3", "modify_cancel", [mod(1, shots_delta=-0.5)], [
        ("formal", "바닐라라떼에서 샷 반 개만 빼 주실 수 있나요?"),
        ("casual", "바라 샷 반 개 빼줘"),
        ("typo", "바닐라라떼 샷 0.5 줄여 주세여")], {"order_state": STATE_B}),

    # ---------------- impossible (10), incl. syrup on coffee (E)
    ("I1", "impossible", [clar("impossible", "레몬에이드는 아이스로만 가능한데, 아이스로 드릴까요?")], [
        ("formal", "따뜻한 레몬에이드 한 잔 주세요."),
        ("casual", "레몬에이드 뜨겁게 하나요"),
        ("slang", "핫 레몬에이드 ㄱ"),
        ("typo", "레몬에이드 따듯하게 해주세여")], {"intent": "impossible:레몬에이드:temperature"}),
    ("I2", "impossible", [clar("impossible", "차에는 샷을 넣을 수 없어요. 그냥 얼그레이로 드릴까요?")], [
        ("formal", "얼그레이에 에스프레소 샷 추가해 주실 수 있나요?"),
        ("slang", "얼그레이 샷추"),
        ("typo", "얼그래이에 샷 추가요")], {"intent": "impossible:얼그레이:shots"}),
    ("I3", "impossible", [clar("impossible", "아메리카노는 당도 조절이 안 돼요. 시럽은 셀프바에서 넣으실 수 있어요.")], [
        ("formal", "아메리카노에 시럽 추가해 주세요."),
        ("casual", "아아 시럽 많이 넣어줘"),
        ("typo", "아메리카노 달달하게 해주세여")], {"intent": "impossible:아메리카노:sweetness"}),

    # ---------------- ambiguous (10)
    ("A1", "ambiguous", [clar("ambiguous", "어떤 메뉴를 말씀하시는 걸까요?")], [
        ("formal", "그거 하나 주세요."),
        ("casual", "그거요 그거 하나"),
        ("slang", "늘 먹던 걸로 ㄱ"),
        ("typo", "아까 그거 하나 주세여")], {"intent": "ambiguous:no_context_reference"}),
    ("A2", "ambiguous", [clar("ambiguous", "치즈케이크와 초콜릿케이크 중 어떤 걸로 드릴까요?")], [
        ("formal", "케이크 한 조각 주시겠어요?"),
        ("casual", "케이크 하나요"),
        ("typo", "케잌 하나 주세요")], {"intent": "ambiguous:케이크"}),
    ("A3", "ambiguous", [clar("ambiguous", "따뜻한 아메리카노와 아이스 아메리카노 중 어떤 걸 취소할까요?")], [
        ("formal", "아메리카노 하나 취소해 주세요."),
        ("casual", "아메리카노 빼 줘"),
        ("typo", "아메리까노 취소요")], {"order_state": STATE_C, "intent": "ambiguous:duplicate_in_state:아메리카노"}),

    # ---------------- off_menu (10)
    ("F1", "off_menu", [clar("off_menu", "녹차라떼는 메뉴에 없어요. 다른 라떼로 드릴까요?")], [
        ("formal", "녹차라떼 한 잔 주세요."),
        ("casual", "녹차라떼 하나요"),
        ("slang", "그린티 라떼 하나"),
        ("typo", "녹차 라테 하나 주세여")], {"intent": "off_menu:녹차라떼"}),
    ("F2", "off_menu", [clar("off_menu", "샌드위치는 없고 케이크만 있어요. 케이크로 드릴까요?")], [
        ("formal", "샌드위치 하나 주시겠어요?"),
        ("casual", "샌드위치 하나요"),
        ("typo", "샌드위취 하나 주세요")], {"intent": "off_menu:샌드위치"}),
    ("F3", "off_menu", [clar("off_menu", "카푸치노는 메뉴에 없어요. 카페라떼는 어떠세요?")], [
        ("formal", "카푸치노 한 잔 부탁드립니다."),
        ("casual", "카푸치노 하나요"),
        ("typo", "카프치노 하나 주세요")], {"intent": "off_menu:카푸치노"}),

    # ---------------- call_staff (10), rule A
    ("C1", "call_staff", [staff("주문한 음료 준비 상태 문의")], [
        ("formal", "제가 아까 주문한 음료 언제 나오나요?"),
        ("casual", "아까 시킨 거 아직이에요?"),
        ("slang", "제 라떼 얼마나 더 걸려요?"),
        ("typo", "주문한 거 언제 나와여?")], {"intent": "call_staff:order_status"}),
    ("C2", "call_staff", [staff("얼음컵 요청")], [
        ("formal", "얼음컵 하나 따로 받을 수 있을까요?"),
        ("casual", "얼음컵 하나 주세요"),
        ("typo", "얼음 컵 하나만 주세여")], {"intent": "call_staff:ice_cup"}),
    ("C3", "call_staff", [staff("단체 주문 시간 문의")], [
        ("formal", "내일 오전 10시에 아메리카노 30잔 단체 주문 맞춰 주실 수 있나요?"),
        ("casual", "내일 아침에 아아 30잔 10시까지 돼요?"),
        ("typo", "낼 10시까지 아메리카노 30잔 가능해여?")], {"intent": "call_staff:bulk_order"}),

    # ---------------- takeout (10), rule C
    ("T1", "takeout", [add("아메리카노", temperature="iced")], [
        ("formal", "아이스 아메리카노 한 잔 포장해 주세요."),
        ("casual", "아아 하나 테이크아웃이요"),
        ("slang", "아아 하나 들고 갈게요"),
        ("typo", "아이스 아메리카노 포장이여")], {"intent_tag": "takeout_mentioned"}),
    ("T2", "takeout", [confirm("takeout")], [
        ("formal", "이대로 주문할게요."),
        ("casual", "그게 다예요, 결제할게요"),
        ("slang", "ㅇㅋ 끝")], {"order_state": STATE_A, "prior": PRIOR_TAKEOUT, "intent_tag": "takeout_earlier"}),
    ("T3", "takeout", [confirm()], [
        ("formal", "주문 이걸로 마칠게요."),
        ("casual", "다 시켰어요"),
        ("typo", "이게 끝이에요 결제해 주세여")], {"order_state": STATE_A, "prior": PRIOR_A, "intent_tag": "dine_not_mentioned"}),

    # ---------------- conditional (10), rule D
    ("D1", "conditional", [add("카페라떼")], [
        ("formal", "녹차라떼 있으면 그걸로 주시고, 없으면 카페라떼로 주세요."),
        ("casual", "녹차라떼 되나요? 안 되면 그냥 라떼요"),
        ("slang", "녹차라떼 없으면 라떼 ㄱ"),
        ("typo", "녹차라떼 있으면 그거, 없으면 카페라떼 주세여")], {"intent_tag": "conditional:녹차라떼"}),
    ("D2", "conditional", [add("자몽에이드")], [
        ("formal", "자몽에이드 있으면 자몽에이드로, 없으면 레몬에이드로 주세요."),
        ("casual", "자몽에이드 돼요? 안 되면 레몬에이드요"),
        ("slang", "자몽에이드 없으면 레몬에이드 ㄱ")], {"intent_tag": "conditional:자몽에이드"}),
    ("D3", "conditional", [add("아메리카노")], [
        ("formal", "디카페인 되면 디카페인으로 하고, 안 되면 그냥 아메리카노 주세요."),
        ("casual", "디카페인 있으면 디카페인 아메리카노요, 없으면 걍 아메리카노"),
        ("typo", "디카페인 없으면 그냥 아메리카노 주세여")], {"intent_tag": "conditional:decaf"}),

    # ---------------- unavailable (12), rule F
    ("U1", "unavailable", [clar("off_menu", "죄송하지만 디카페인은 준비되어 있지 않아요. 일반 아메리카노로 드릴까요?")], [
        ("formal", "디카페인 아메리카노 한 잔 주세요."),
        ("casual", "디카페인으로 아메리카노 하나요"),
        ("typo", "디카페인 아메리카노 되나여?")], {"intent": "off_menu:decaf:아메리카노"}),
    ("U2", "unavailable", [clar("off_menu", "죄송하지만 휘핑크림은 없어요. 그냥 카페모카로 드릴까요?")], [
        ("formal", "카페모카에 휘핑크림 올려 주세요."),
        ("casual", "모카 휘핑 추가요"),
        ("typo", "카페모카 휘핑 많이 주세여")], {"intent": "off_menu:whipped_cream:카페모카"}),
    ("U3", "unavailable", [clar("off_menu", "죄송하지만 락토프리 우유는 없어요. 오트밀크는 가능한데 괜찮으세요?")], [
        ("formal", "카페라떼를 락토프리 우유로 바꿔서 주실 수 있나요?"),
        ("casual", "라떼 락토프리로 하나요"),
        ("typo", "락토프리 라떼 하나 주세여")], {"intent": "off_menu:lactose_free:카페라떼"}),
    ("U4", "unavailable", [clar("off_menu", "죄송하지만 원두는 고르실 수 없어요. 기본 원두로 드릴까요?")], [
        ("formal", "아메리카노 산미 있는 원두로 주세요."),
        ("casual", "아아 원두 고소한 걸로요"),
        ("typo", "아메리카노 원두 골라서 주문 되나여?")], {"intent": "off_menu:bean_selection:아메리카노"}),
]


def build_cases() -> list[dict]:
    cases = []
    for gid, cat, expected, variants, extra in GROUPS:
        for i, (style, text) in enumerate(variants, 1):
            case = {
                "id": f"{gid}-{i}",
                "category": cat,
                "paraphrase_group": gid,
                "style": style,
                "order_state": extra.get("order_state", []),
                "conversation": list(extra.get("prior", [])) + [{"role": "user", "content": text}],
                "expected": expected,
            }
            for k in ("intent", "intent_tag"):
                if k in extra:
                    case[k] = extra[k]
            cases.append(case)
    return cases


def review_md(cases: list[dict], n: int = 20, seed: int = 42) -> str:
    rng = random.Random(seed)
    by_cat = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)
    picked = [rng.choice(by_cat[cat]) for cat in CATEGORY_KO]
    rest = [c for c in cases if c not in picked]
    picked += rng.sample(rest, n - len(picked))
    order = {g[0]: i for i, g in enumerate(GROUPS)}
    picked.sort(key=lambda c: (order[c["paraphrase_group"]], int(c["id"].split("-")[1])))

    def cell(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", "<br>")

    L = [
        "# 카페 시험 문제 검토용 샘플 (20문항)",
        "",
        f"전체 {len(cases)}문항 중 12개 유형에서 1문항씩 + 나머지에서 무작위 {n - len(CATEGORY_KO)}문항(seed {seed})을 뽑았습니다.",
        "'기대 답'이 맞는지 직접 확인하고, 틀린 문항은 '확인' 칸에 표시해 주세요.",
        "",
        "채점 기준 요약:",
        "- 손님이 말하지 않은 옵션은 기본값으로 봅니다: 따뜻하게(콜드브루·에이드는 아이스), 레귤러, 샷 그대로, 보통 당도, 일반 우유, 수량 1, 매장.",
        "- 되묻기·거절(ask_clarification)은 질문 문장과 상관없이 이유만 맞으면 정답입니다.",
        "  - 모호함: 무엇/어느 항목인지 불분명",
        "  - 불가능한 조합: 메뉴판에 있는 옵션이지만 그 메뉴에는 안 됨 (예: 따뜻한 에이드, 커피에 시럽)",
        "  - 제공하지 않음: 메뉴에 없는 음료·음식, 디카페인, 원두 선택, 휘핑크림, 락토프리 우유",
        "- 직원 호출(call_staff)은 호출 사유 문장과 상관없이 직원을 불렀으면 정답입니다.",
        "- 이미 담긴 항목을 바꾸거나 취소할 때는 '현재 주문'의 번호(1부터)를 씁니다.",
        "",
        "| ID | 유형 | 현재 주문 | 대화 | 기대 답 | 확인 |",
        "|---|---|---|---|---|---|",
    ]
    for c in picked:
        conv = "<br>".join(
            f"{'손님' if m['role'] == 'user' else '점원'}: {m['content']}" for m in c["conversation"]
        )
        L.append(
            f"| {c['id']} | {CATEGORY_KO[c['category']]} | {cell(format_order_state(c['order_state']))} | "
            f"{cell(conv)} | {cell(describe_calls(c['expected']))} | ☐ |"
        )
    return "\n".join(L) + "\n"


def main():
    cases = build_cases()
    (DATA / "exam_cafe.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8"
    )
    (DATA / "exam_cafe_review.md").write_text(review_md(cases), encoding="utf-8")
    cats = {}
    for c in cases:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    print(f"{len(cases)} cases, {len(GROUPS)} groups -> data/exam_cafe.jsonl")
    print(cats)


if __name__ == "__main__":
    main()
