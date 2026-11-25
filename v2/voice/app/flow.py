"""Order-flow helpers (A1): what the customer said about dine-in / takeout, and the few short answers
that need no language model (a whole utterance that is only "yes", "no", "that's all" or a dine option).

Anything longer or mixed ("네, 그리고 라떼 하나 더요") goes to the model as usual.
"""
from __future__ import annotations

import re

TAKEOUT = re.compile(r"포장|테이크\s*아웃|가져\s*갈|가지고\s*갈|들고\s*갈|싸\s*주")
DINE_IN = re.compile(r"매장|먹고\s*갈|마시고\s*갈|먹고\s*가|마시고\s*가|여기서\s*(먹|마시|드)")

FILLER = r"(?:아|음|어|그럼|그러면|그냥|저기|네네|예예)"
YES = (r"(?:네|넵|예|응|어|맞아요|맞습니다|맞아|맞네요|그래요|좋아요|괜찮아요|그렇게\s*해\s*주세요|그렇게\s*주세요|"
       r"그걸로\s*주세요|그대로\s*주세요|그대로요|주문할게요|주문할께요|주문해\s*주세요|부탁해요|부탁드려요|부탁합니다|"
       r"오케이|확인|감사합니다)")
DINE = (r"(?:(?:포장|테이크\s*아웃|매장)\s*(?:이요|이에요|요|으로요|으로\s*해\s*주세요|으로\s*할게요|할게요|할께요|"
        r"해\s*주세요|에서요|에서\s*먹을게요|에서\s*마실게요|에서\s*먹고\s*갈게요|이고요|입니다)?|"
        r"(?:여기서\s*)?(?:먹고|마시고)\s*갈게요|(?:여기서\s*)?(?:먹을게요|마실게요)|"
        r"(?:가져|가지고|들고)\s*갈게요)")
NO = r"(?:아니요|아뇨|아니에요|아니|잠깐만요|잠시만요|잠깐만)"
DONE = (r"(?:(?:이게|그게|이거|그거|이렇게|그렇게)?\s*다(?:예요|에요|입니다|야)|없어요|없습니다|없어|됐어요|됐습니다|"
        r"주문할게요|주문할께요|주문해\s*주세요|주문이요|이렇게\s*주세요|그렇게\s*주세요|이대로\s*주세요|끝이요|끝)")


def _clean(text: str) -> str:
    return re.sub(r"[\s,.?!~·]+", " ", str(text or "")).strip()


def _only(text: str, *parts: str) -> bool:
    """True when the whole utterance is a sequence of the given phrases (plus fillers)."""
    alt = "|".join(parts + (FILLER,))
    return bool(text) and re.fullmatch(rf"(?:\s*(?:{alt}))+\s*", text) is not None


def dine_of(text: str):
    """'takeout' / 'dine_in' when the customer said it in this utterance, else None."""
    t = str(text or "")
    if TAKEOUT.search(t):
        return "takeout"
    if DINE_IN.search(t):
        return "dine_in"
    return None


def awaiting_answer(text: str):
    """While the order is read back: 'yes' (yes and/or a dine option only), 'no' (only no), else None."""
    t = _clean(text)
    if _only(t, YES, DINE):
        return "yes"
    if _only(t, NO):
        return "no"
    return None


def is_done(text: str) -> bool:
    """While ordering: the customer says they are done ("아니 없어요", "이게 다예요", "주문할게요")."""
    t = _clean(text)
    return _only(t, DONE, NO, r"(?:네|예)") and re.search(DONE, t) is not None


TEMP_WORDS = re.compile(r"따뜻|따듯|뜨거|뜨겁|뜨끈|뜨아|핫|아이스|차가|차갑|시원|얼음|아아|아바라|아이스아메|hot|ice", re.I)


def temp_said(*texts: str) -> bool:
    """The customer named a temperature (this turn or the one before, e.g. an answer to 따뜻한 거요, 아이스요?)."""
    return any(TEMP_WORDS.search(str(t or "")) for t in texts)
