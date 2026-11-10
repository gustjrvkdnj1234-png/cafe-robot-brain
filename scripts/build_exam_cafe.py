"""Build data/exam_cafe.jsonl (120 hand-written cases in 32 paraphrase groups)
and data/exam_cafe_review.md (20 sampled cases for manual checking).

    python scripts/build_exam_cafe.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cafe_common import DATA, describe_calls, format_order_state

CATEGORY_KO = {
    "basic": "기본 주문",
    "options": "옵션",
    "slang": "줄임말/은어",
    "multi_item": "여러 메뉴",
    "modify_cancel": "변경/취소/확정",
    "impossible": "불가능한 요청",
    "ambiguous": "모호한 요청",
    "off_menu": "메뉴에 없음",
}


def add(menu, **o):
    return {"name": "add_item", "arguments": {"menu": menu, **o}}


def mod(i, **c):
    return {"name": "modify_item", "arguments": {"item_index": i, "changes": c}}


def rm(i):
    return {"name": "remove_item", "arguments": {"item_index": i}}


def clar(reason, q):
    return {"name": "ask_clarification", "arguments": {"question": q, "reason": reason}}


def confirm():
    return {"name": "confirm_order", "arguments": {}}


STATE_A = [
    {"menu": "아메리카노", "temperature": "iced", "size": "regular", "quantity": 1},
    {"menu": "카페라떼", "temperature": "hot", "size": "regular", "quantity": 1},
]
PRIOR_A = [
    {"role": "user", "content": "아아 하나랑 따뜻한 라떼 하나요"},
    {"role": "assistant", "content": "아이스 아메리카노 1잔, 따뜻한 카페라떼 1잔 담았습니다."},
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
GROUPS = [
    # ---------------- basic
    ("B1", "basic", [add("아메리카노")], [
        ("formal", "아메리카노 한 잔 주시겠어요?"),
        ("casual", "아메리카노 하나요"),
        ("slang", "아메 하나 ㄱㄱ"),
        ("typo", "아메리까노 한잔 주세여")], {}),
    ("B2", "basic", [add("카페라떼")], [
        ("formal", "카페라떼 한 잔 부탁드립니다."),
        ("casual", "라떼 하나 주세요"),
        ("slang", "라떼 ㄱ"),
        ("typo", "까페라떼 하나요")], {}),
    ("B3", "basic", [add("치즈케이크")], [
        ("formal", "치즈케이크 한 조각 주시겠어요?"),
        ("casual", "치즈케이크 하나요"),
        ("slang", "치케 하나"),
        ("typo", "치즈캐이크 하나 주세여")], {}),
    ("B4", "basic", [add("유자차")], [
        ("formal", "유자차 한 잔 주실 수 있을까요?"),
        ("casual", "유자차 하나 줘요"),
        ("typo", "유쟈차 하나요")], {}),

    # ---------------- options
    ("O1", "options", [add("아메리카노", temperature="iced", size="large")], [
        ("formal", "아이스 아메리카노 라지 사이즈로 한 잔 주세요."),
        ("casual", "아이스 아메리카노 큰 걸로 하나요"),
        ("slang", "아아 라지 하나"),
        ("typo", "아이스 아메리카노 라쥐로 하나요")], {}),
    ("O2", "options", [add("카페라떼", temperature="iced", milk="oat")], [
        ("formal", "아이스 카페라떼를 오트밀크로 변경해서 한 잔 주세요."),
        ("casual", "아이스 라떼 오트로 하나"),
        ("slang", "아이스라떼 귀리우유로 ㄱ"),
        ("typo", "아이스 카페라떼 오트밀크로 하나 주세용")], {}),
    ("O3", "options", [add("바닐라라떼", sweetness="less", shots_delta=1)], [
        ("formal", "바닐라라떼 덜 달게 해서 샷 하나 추가해 주세요."),
        ("casual", "바닐라라떼 샷 추가하고 덜 달게요"),
        ("slang", "바라 샷추 덜달"),
        ("typo", "바닐라라때 덜 달게 샷 추가요")], {}),
    ("O4", "options", [add("아메리카노", temperature="hot", shots_delta=-1)], [
        ("formal", "따뜻한 아메리카노 샷 하나 빼서 한 잔 주세요."),
        ("casual", "뜨거운 아메리카노 하나, 샷 하나 빼 줘"),
        ("slang", "뜨아 샷 하나 덜 넣어서요")], {}),

    # ---------------- slang
    ("S1", "slang", [add("아메리카노", temperature="iced", quantity=2)], [
        ("formal", "아이스 아메리카노 두 잔 부탁드립니다."),
        ("casual", "아아 두 잔이요"),
        ("slang", "아아 2"),
        ("typo", "아아 두잔여")], {}),
    ("S2", "slang", [add("바닐라라떼", temperature="iced")], [
        ("formal", "아이스 바닐라라떼 한 잔 주세요."),
        ("casual", "아바라 하나요"),
        ("slang", "아바라 ㄱ"),
        ("typo", "아바러 하나")], {}),
    ("S3", "slang", [add("카라멜마키아또")], [
        ("formal", "카라멜 마키아또 한 잔 주세요."),
        ("slang", "카라마끼 하나요"),
        ("typo", "캬라멜 마끼아또 하나")], {}),
    ("S4", "slang", [add("아메리카노", temperature="hot")], [
        ("formal", "따뜻한 아메리카노 한 잔 주세요."),
        ("casual", "뜨아 하나요"),
        ("slang", "핫아메 하나"),
        ("typo", "뜨아 하나여")], {}),

    # ---------------- multi_item
    ("M1", "multi_item", [add("아메리카노", temperature="iced"), add("치즈케이크")], [
        ("formal", "아이스 아메리카노 한 잔이랑 치즈케이크 한 조각 주세요."),
        ("casual", "아이스 아메리카노하고 치즈케이크 하나씩이요"),
        ("slang", "아아 하나랑 치케 하나"),
        ("typo", "아이스 아메리카노랑 치즈캐잌 하나요")], {}),
    ("M2", "multi_item", [add("카페라떼", quantity=2), add("레몬에이드")], [
        ("formal", "카페라떼 두 잔하고 레몬에이드 한 잔 주세요."),
        ("casual", "라떼 둘, 레몬에이드 하나요"),
        ("slang", "라떼 2 레몬에이드 1"),
        ("typo", "카페라떼 두잔이랑 레몬에이드 하나여")], {}),
    ("M3", "multi_item", [add("카페모카", temperature="iced", size="large"), add("얼그레이", temperature="hot"), add("초콜릿케이크")], [
        ("formal", "아이스 카페모카 라지 하나, 따뜻한 얼그레이 하나, 초콜릿 케이크 하나 주세요."),
        ("casual", "아이스 모카 큰 거 하나랑 얼그레이 따뜻하게 하나, 초코케이크 하나요"),
        ("slang", "아이스모카 라지 + 핫 얼그레이 + 초코케익"),
        ("typo", "아이스 카페모카 라지 하나 따듯한 얼그레이 하나 초콜렛케이크 하나요")], {}),
    ("M4", "multi_item", [add("콜드브루", quantity=2), add("자몽에이드", sweetness="less")], [
        ("formal", "콜드브루 두 잔이랑 자몽에이드 덜 달게 한 잔 부탁해요."),
        ("casual", "콜드브루 둘, 자몽에이드 하나는 덜 달게요"),
        ("slang", "콜브 2 자몽에이드 1 덜달게")], {}),

    # ---------------- modify_cancel
    ("X1", "modify_cancel", [mod(1, size="large")], [
        ("formal", "첫 번째 아메리카노를 라지로 바꿔 주시겠어요?"),
        ("casual", "아아 큰 걸로 바꿔줘"),
        ("slang", "아메리카노 사이즈업이요"),
        ("typo", "아메리까노 라지로 변경해주세여")], {"order_state": STATE_A, "prior": PRIOR_A}),
    ("X2", "modify_cancel", [rm(2)], [
        ("formal", "카페라떼는 취소할게요."),
        ("casual", "라떼 빼줘"),
        ("slang", "라떼 ㄴㄴ 취소"),
        ("ordinal", "두 번째 메뉴 취소해 주세요.")], {"order_state": STATE_A, "prior": PRIOR_A}),
    ("X3", "modify_cancel", [mod(1, temperature="iced")], [
        ("formal", "바닐라라떼를 아이스로 변경해 주실 수 있나요?"),
        ("casual", "바라 아이스로 바꿔요"),
        ("slang", "바닐라라떼 차갑게 해줘")], {"order_state": STATE_B}),
    ("X4", "modify_cancel", [confirm()], [
        ("formal", "이대로 주문할게요."),
        ("casual", "그게 다예요, 결제할게요"),
        ("slang", "ㅇㅋ 끝"),
        ("typo", "이대루 주문해주세여")], {"order_state": STATE_A, "prior": PRIOR_A}),

    # ---------------- impossible
    ("I1", "impossible", [clar("impossible", "레몬에이드는 아이스로만 가능한데, 아이스로 드릴까요?")], [
        ("formal", "따뜻한 레몬에이드 한 잔 주세요."),
        ("casual", "레몬에이드 뜨겁게 하나요"),
        ("slang", "핫 레몬에이드 ㄱ"),
        ("typo", "레몬에이드 따듯하게 해주세여")], {"intent": "impossible:레몬에이드:temperature"}),
    ("I2", "impossible", [clar("impossible", "차에는 샷을 넣을 수 없어요. 그냥 얼그레이로 드릴까요?")], [
        ("formal", "얼그레이에 에스프레소 샷 추가해 주실 수 있나요?"),
        ("casual", "얼그레이 샷 하나 넣어줘"),
        ("slang", "얼그레이 샷추"),
        ("typo", "얼그래이에 샷 추가요")], {"intent": "impossible:얼그레이:shots"}),
    ("I3", "impossible", [clar("impossible", "에스프레소는 사이즈 선택이 없어요. 기본으로 드릴까요?")], [
        ("formal", "에스프레소 라지 사이즈로 한 잔 주세요."),
        ("casual", "에스프레소 큰 걸로 하나"),
        ("slang", "에소 라지"),
        ("typo", "에스프래소 라지로요")], {"intent": "impossible:에스프레소:size"}),
    ("I4", "impossible", [clar("impossible", "오트밀크는 라떼 종류만 가능해요. 카페라떼로 드릴까요?")], [
        ("formal", "아메리카노에 오트밀크 넣어 주세요."),
        ("casual", "아메리카노 귀리우유로 하나요"),
        ("slang", "아아 오트로")], {"intent": "impossible:아메리카노:milk"}),

    # ---------------- ambiguous
    ("A1", "ambiguous", [clar("ambiguous", "어떤 메뉴를 말씀하시는 걸까요?")], [
        ("formal", "그거 하나 주세요."),
        ("casual", "그거요 그거 하나"),
        ("slang", "늘 먹던 걸로 ㄱ"),
        ("typo", "아까 그거 하나 주세여")], {"intent": "ambiguous:no_context_reference"}),
    ("A2", "ambiguous", [clar("ambiguous", "치즈케이크와 초콜릿케이크 중 어떤 걸로 드릴까요?")], [
        ("formal", "케이크 한 조각 주시겠어요?"),
        ("casual", "케이크 하나요"),
        ("slang", "케익 ㄱ"),
        ("typo", "케잌 하나 주세요")], {"intent": "ambiguous:케이크"}),
    ("A3", "ambiguous", [clar("ambiguous", "레몬, 자몽, 청포도 에이드 중 어떤 걸로 드릴까요?")], [
        ("formal", "에이드 한 잔 주세요."),
        ("casual", "에이드 하나요"),
        ("slang", "에이드 ㄱㄱ"),
        ("typo", "애이드 하나")], {"intent": "ambiguous:에이드"}),
    ("A4", "ambiguous", [clar("ambiguous", "따뜻한 아메리카노와 아이스 아메리카노 중 어떤 걸 취소할까요?")], [
        ("formal", "아메리카노 하나 취소해 주세요."),
        ("casual", "아메리카노 빼 줘"),
        ("typo", "아메리까노 취소요")], {"order_state": STATE_C, "intent": "ambiguous:duplicate_in_state:아메리카노"}),

    # ---------------- off_menu
    ("F1", "off_menu", [clar("off_menu", "녹차라떼는 메뉴에 없어요. 다른 라떼로 드릴까요?")], [
        ("formal", "녹차라떼 한 잔 주세요."),
        ("casual", "녹차라떼 하나요"),
        ("slang", "그린티 라떼 하나"),
        ("typo", "녹차 라테 하나 주세여")], {"intent": "off_menu:녹차라떼"}),
    ("F2", "off_menu", [clar("off_menu", "스무디는 메뉴에 없어요. 에이드는 어떠세요?")], [
        ("formal", "딸기 스무디 한 잔 주실 수 있나요?"),
        ("casual", "딸기스무디 하나"),
        ("slang", "딸기 스무디 ㄱㄱ"),
        ("typo", "딸기 스무듸 하나요")], {"intent": "off_menu:딸기스무디"}),
    ("F3", "off_menu", [clar("off_menu", "샌드위치는 없고 케이크만 있어요. 케이크로 드릴까요?")], [
        ("formal", "샌드위치 하나 주시겠어요?"),
        ("casual", "샌드위치 하나요"),
        ("slang", "샌드위치 ㄱ"),
        ("typo", "샌드위취 하나 주세요")], {"intent": "off_menu:샌드위치"}),
    ("F4", "off_menu", [clar("off_menu", "카푸치노는 메뉴에 없어요. 카페라떼는 어떠세요?")], [
        ("formal", "카푸치노 한 잔 부탁드립니다."),
        ("casual", "카푸치노 하나요"),
        ("typo", "카프치노 하나 주세요")], {"intent": "off_menu:카푸치노"}),
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
            if "intent" in extra:
                case["intent"] = extra["intent"]
            cases.append(case)
    return cases


def review_md(cases: list[dict], n: int = 20, seed: int = 42) -> str:
    rng = random.Random(seed)
    by_cat = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)
    picked = []
    for cat in CATEGORY_KO:
        picked += rng.sample(by_cat[cat], 2)
    rest = [c for c in cases if c not in picked]
    picked += rng.sample(rest, n - len(picked))
    picked.sort(key=lambda c: [c["id"].split("-")[0], int(c["id"].split("-")[1])])

    def cell(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", "<br>")

    L = [
        "# 카페 시험 문제 검토용 샘플 (20문항)",
        "",
        f"전체 {len(cases)}문항 중 유형별 2문항씩 + 무작위 4문항(seed {seed})을 뽑았습니다.",
        "'기대 답'이 맞는지 직접 확인하고, 틀린 문항은 '확인' 칸에 표시해 주세요.",
        "",
        "채점 기준 요약:",
        "- 손님이 말하지 않은 옵션은 기본값(사이즈 레귤러, 수량 1, 메뉴별 기본 온도 등)으로 봅니다.",
        "- 되묻기(ask_clarification)는 질문 문장과 상관없이 이유(모호함 / 불가능 / 메뉴에 없음)만 맞으면 정답입니다.",
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
    out = DATA / "exam_cafe.jsonl"
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    (DATA / "exam_cafe_review.md").write_text(review_md(cases), encoding="utf-8")
    cats = {}
    for c in cases:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    print(f"{len(cases)} cases, {len(GROUPS)} groups -> {out.relative_to(Path(__file__).resolve().parent.parent)}")
    print(cats)


if __name__ == "__main__":
    main()
