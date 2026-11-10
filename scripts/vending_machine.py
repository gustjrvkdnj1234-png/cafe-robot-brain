"""Rule-based baseline ("vending machine") for the cafe ordering task.

Keyword / synonym-dictionary / regex parser that emits the same tool calls as
the LLM. Written from data/menu.json and the menu rules only (not tuned on the
exam). Roughly what one engineer would build in a day:

  * menu detection: canonical names + aliases (space-tolerant), longest match
    first, with a compound-word guard for short generic aliases (e.g. "라떼"
    inside "녹차라떼"), plus jamo-level fuzzy matching for typos
  * multi-item splitting on separators / quantity words between menu mentions
  * option regexes (temperature, size, shots, sweetness, milk, quantity)
  * menu-rule validation -> ask_clarification(impossible)
  * modify / cancel / confirm intents resolved against the current order state

Usage:
    python scripts/vending_machine.py "아아 두 잔이랑 치즈케이크 하나"
    python scripts/vending_machine.py "라떼 빼줘" --state '[{"menu":"카페라떼"}]'
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys

from cafe_common import items_by_name, load_menu

# ---------------------------------------------------------------- vocabulary

NUM_WORD = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "석": 3,
            "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8}
UNIT = r"(?:잔|개|조각|컵|병|피스)"

RE_ICED = re.compile(r"(아이스|ice|iced|차갑|차가운|시원|얼음|콜드(?!브루))", re.I)
RE_HOT = re.compile(r"(핫|hot|뜨거|뜨겁|뜨끈|따뜻|따듯|따숩|데워|데운)", re.I)
RE_LARGE = re.compile(r"(라지|large|큰\s*(?:걸|거|것|사이즈|컵|잔|용량)|크게|사이즈\s*업|벤티|그란데)", re.I)
RE_REGULAR = re.compile(r"(레귤러|regular|작은\s*(?:걸|거|것|사이즈|컵)|기본\s*사이즈|스몰|small)", re.I)
RE_OAT = re.compile(r"(오트|귀리|oat)", re.I)
RE_SWEET_LESS = re.compile(
    r"(덜\s*달|안\s*달|달지\s*않|당도\s*(?:를|는)?\s*(?:낮|적|줄|약)|시럽\s*(?:을|은)?\s*(?:적게|조금|덜|빼|줄|반)|저당|반\s*당)")
RE_SWEET_MORE = re.compile(
    r"(더\s*달|많이\s*달|달게|달달하게|당도\s*(?:를|는)?\s*(?:높|올|많)|시럽\s*(?:을)?\s*(?:추가|많이|더))")
RE_SHOT_MINUS = re.compile(r"(샷\s*(?:을|은|는)?\s*(?:하나|한\s*개|1\s*개?)?\s*(?:빼|덜|줄|적게)|연하게)")
RE_SHOT_PLUS3 = re.compile(r"샷\s*(?:을|은|는)?\s*(?:세|셋|3)\s*(?:개|번)?\s*(?:추가|더|넣)")
RE_SHOT_PLUS2 = re.compile(r"((?:투|더블)\s*샷|샷\s*(?:을|은|는)?\s*(?:두|둘|2)\s*(?:개|번)?\s*(?:추가|더|넣))")
RE_SHOT_PLUS1 = re.compile(r"(샷\s*(?:을|은|는)?\s*(?:하나|한\s*개|1\s*개?)?\s*(?:추가|추|더|넣)|진하게)")
SHOT_PATTERNS = (RE_SHOT_PLUS3, RE_SHOT_PLUS2, RE_SHOT_PLUS1, RE_SHOT_MINUS)


def strip_shots(s: str) -> str:
    for p in SHOT_PATTERNS:
        s = p.sub(" ", s)
    return s

RE_QTY_WORD = re.compile(r"(한|두|세|석|네|다섯|여섯|일곱|여덟)\s*" + UNIT + r"|(하나|둘|셋|넷|다섯|여섯)")
RE_QTY_DIGIT = re.compile(r"(?<![\d번])(\d{1,2})\s*(?:" + UNIT + r")?(?!\s*(?:번|샷|\d))")

RE_ORDINAL = [
    (re.compile(r"(첫\s*번째|첫째|1\s*번)"), 1),
    (re.compile(r"(두\s*번째|둘째|2\s*번)"), 2),
    (re.compile(r"(세\s*번째|셋째|3\s*번)"), 3),
    (re.compile(r"(네\s*번째|넷째|4\s*번)"), 4),
]
RE_LAST = re.compile(r"마지막")

RE_CANCEL = re.compile(r"(취소|빼\s*(?:줘|주|고|요|버)|지워|삭제|없애|안\s*(?:먹|시킬|할래)|ㄴㄴ|필요\s*없)")
RE_MODIFY = re.compile(r"(바꿔|바꾸|바꿀|변경|말고|대신|수정|사이즈\s*업|(?:으로|로|게|하게)\s*(?:해|할게|부탁|요|주세요|줘))")
RE_INSTEAD = re.compile(r"(말고|대신)")
RE_CONFIRM = re.compile(
    r"(이대로|그대로\s*(?:해|주문)|그게\s*다|그거면\s*(?:돼|됐|충분)|다\s*(?:예요|에요|입니다|이에요)|결제|계산|"
    r"주문\s*(?:할게|하겠|해\s*주|완료|끝|넣어|확정)|이상입니다|이상이요|끝|됐어요|됐습니다|더\s*없|충분)")

RE_SEPARATOR = re.compile(r"(,|\+|/|그리고|이랑|하고|에다가|랑|및|&|\band\b)")
RE_OPTION_PREFIX = re.compile(r"(아이스|핫|따뜻한|따듯한|뜨거운|차가운|시원한|큰|라지|레귤러|오트)\s*$")

# 메뉴가 없을 때 '무엇인지 불분명'으로 볼 표현 (카테고리 이름, 지시어)
GENERIC_WORDS = ["커피", "라떼류", "에이드", "케이크", "케익", "케잌", "음료", "디저트", "마실거", "마실 거",
                 "아무거나", "추천", "그거", "이거", "저거", "아까", "평소", "저번", "늘 먹던", "같은 걸로", "똑같이",
                 "시원한 거", "따뜻한 거", "단 거", "달달한 거"]
GENERIC_STANDALONE = re.compile(r"(?:^|\s)(차|티|빵|그걸로|그걸)(?=\s|$|[요를좀한하는도])")

# 붙여 쓰면 다른 메뉴의 일부일 수 있는 짧은 별칭 (예: 녹차라떼의 '라떼')
GENERIC_ALIASES = {"라떼", "라테", "모카", "유자", "아메", "에소", "콜브", "바라"}
FUNCTION_WORDS = {"저", "제", "그럼", "음", "어", "네", "예", "그냥", "일단", "여기", "저기", "저기요", "혹시", "그리고",
                  "또", "하고", "랑", "이랑", "좀", "요", "그", "이", "그거", "아", "그럼요", "그러면"}

# ---------------------------------------------------------------- helpers

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def jamo(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(CHO[code // 588])
            out.append(JUNG[(code % 588) // 28])
            if code % 28:
                out.append(JONG[code % 28])
        else:
            out.append(ch)
    return "".join(out)


def jamo_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, jamo(a), jamo(b)).ratio()


def spaced_pattern(surface: str) -> re.Pattern:
    chars = [re.escape(c) for c in surface.replace(" ", "")]
    return re.compile(r"\s*".join(chars), re.I)


def word_before(text: str, pos: int) -> tuple[str, bool]:
    """Return (hangul word immediately before pos, attached-without-space)."""
    i = pos
    attached = i > 0 and not text[i - 1].isspace()
    while i > 0 and text[i - 1].isspace():
        i -= 1
    j = i
    while j > 0 and re.match(r"[가-힣A-Za-z]", text[j - 1]):
        j -= 1
    return text[j:i], attached


# ---------------------------------------------------------------- machine

class VendingMachine:
    def __init__(self):
        self.items = items_by_name()
        surfaces = []
        for it in load_menu()["items"]:
            surfaces.append((it["name"], it["name"], {}))
            surfaces.append((it["name_en"], it["name"], {}))
        for a in load_menu()["aliases"]:
            surfaces.append((a["alias"], a["menu"], a.get("implies") or {}))
        # longest first so that "바닐라라떼" wins over "라떼"
        surfaces.sort(key=lambda s: -len(s[0].replace(" ", "")))
        self.surfaces = [(s, canon, imp, spaced_pattern(s)) for s, canon, imp in surfaces]

    # ---------- menu mentions

    def find_menus(self, text: str) -> list[dict]:
        taken = [False] * len(text)
        found = []
        for surface, canon, implies, pat in self.surfaces:
            for m in pat.finditer(text):
                s, e = m.span()
                if any(taken[s:e]):
                    continue
                compact = surface.replace(" ", "")
                if compact in GENERIC_ALIASES and not self._alias_boundary_ok(text, s):
                    continue
                for k in range(s, e):
                    taken[k] = True
                found.append({"start": s, "end": e, "menu": canon, "implies": implies, "surface": m.group(0)})
        found += self._fuzzy_menus(text, taken)
        return sorted(found, key=lambda f: f["start"])

    def _alias_boundary_ok(self, text: str, start: int) -> bool:
        prev, attached = word_before(text, start)
        if not prev:
            return True
        if RE_OPTION_PREFIX.search(prev) or RE_QTY_WORD.fullmatch(prev):
            return True
        if attached:
            return False  # e.g. 녹차라떼, 딸기라떼
        # spaced: "녹차 라떼" -> compound, "저 라떼" -> fine
        return prev in FUNCTION_WORDS or prev.endswith(("요", "고", "랑", "은", "는"))

    def _fuzzy_menus(self, text: str, taken: list[bool]) -> list[dict]:
        found = []
        tokens = list(re.finditer(r"[가-힣]+", text))
        candidates = []
        for i, t in enumerate(tokens):
            candidates.append((t.start(), t.end(), t.group()))
            if i + 1 < len(tokens):
                t2 = tokens[i + 1]
                candidates.append((t.start(), t2.end(), t.group() + t2.group()))
        best_by_start = {}
        for s, e, tok in candidates:
            if any(taken[s:e]):
                continue
            for surface, canon, implies, _ in self.surfaces:
                comp = surface.replace(" ", "")
                if len(comp) < 3 or not re.match(r"[가-힣]", comp):
                    continue
                for L in (len(comp) - 1, len(comp), len(comp) + 1):
                    if L < 3 or L > len(tok):
                        continue
                    r = jamo_ratio(tok[:L], comp)
                    if r >= 0.84 and r > best_by_start.get(s, (0,))[0]:
                        best_by_start[s] = (r, s, s + L if e - s == len(tok) else e, canon, implies, tok[:L])
        for r, s, e, canon, implies, surf in sorted(best_by_start.values(), key=lambda x: -x[0]):
            if any(taken[s:e]):
                continue
            for k in range(s, min(e, len(taken))):
                taken[k] = True
            found.append({"start": s, "end": e, "menu": canon, "implies": implies, "surface": surf, "fuzzy": True})
        return found

    # ---------- options

    @staticmethod
    def parse_options(seg: str, with_quantity: bool = True) -> dict:
        o = {}
        if RE_ICED.search(seg):
            o["temperature"] = "iced"
        elif RE_HOT.search(seg):
            o["temperature"] = "hot"
        if RE_LARGE.search(seg):
            o["size"] = "large"
        elif RE_REGULAR.search(seg):
            o["size"] = "regular"
        if RE_SHOT_MINUS.search(seg):
            o["shots_delta"] = -1
        elif RE_SHOT_PLUS3.search(seg):
            o["shots_delta"] = 3
        elif RE_SHOT_PLUS2.search(seg):
            o["shots_delta"] = 2
        elif RE_SHOT_PLUS1.search(seg):
            o["shots_delta"] = 1
        if RE_SWEET_LESS.search(seg):
            o["sweetness"] = "less"
        elif RE_SWEET_MORE.search(seg):
            o["sweetness"] = "more"
        if RE_OAT.search(seg):
            o["milk"] = "oat"
        if with_quantity:
            q = VendingMachine.parse_quantity(seg)
            if q is not None:
                o["quantity"] = q
        return o

    @staticmethod
    def parse_quantity(seg: str):
        s = strip_shots(seg)
        for pat, _ in RE_ORDINAL:
            s = pat.sub(" ", s)
        m = RE_QTY_DIGIT.search(s)
        if m:
            return int(m.group(1))
        m = RE_QTY_WORD.search(s)
        if m:
            return NUM_WORD[m.group(1) or m.group(2)]
        return None

    def violations(self, menu: str, o: dict) -> list[str]:
        it = self.items[menu]
        v = []
        if "temperature" in o and o["temperature"] not in it["temperatures"]:
            v.append("temperature")
        if o.get("size") == "large" and "large" not in it["sizes"]:
            v.append("size")
        if o.get("shots_delta") and not it["shots"]:
            v.append("shots")
        if o.get("sweetness") and not it["sweetness"]:
            v.append("sweetness")
        if o.get("milk") == "oat" and not it["milk"]:
            v.append("milk")
        return v

    # ---------- intents

    def resolve_target(self, text: str, state: list, mentions: list) -> int | None:
        for pat, idx in RE_ORDINAL:
            if pat.search(text) and idx <= len(state):
                return idx
        if RE_LAST.search(text) and state:
            return len(state)
        if mentions:
            names = {m["menu"] for m in mentions}
            idxs = [i for i, it in enumerate(state, 1) if it.get("menu") in names]
            return idxs[0] if len(idxs) == 1 else None
        return 1 if len(state) == 1 else None

    @staticmethod
    def clar(reason: str, question: str) -> dict:
        return {"name": "ask_clarification", "arguments": {"question": question, "reason": reason}}

    def predict_text(self, text: str, state: list | None = None) -> list[dict]:
        state = state or []
        text = text.strip()
        mentions = self.find_menus(text)
        no_shot = strip_shots(text)

        if not mentions and RE_CONFIRM.search(text) and not RE_CANCEL.search(no_shot):
            return [{"name": "confirm_order", "arguments": {}}]

        if state:
            in_state = [m for m in mentions if any(it.get("menu") == m["menu"] for it in state)]
            has_ordinal = any(p.search(text) for p, _ in RE_ORDINAL) or RE_LAST.search(text)
            refers_to_state = bool(in_state) or has_ordinal or not mentions
            if RE_CANCEL.search(no_shot) and refers_to_state:
                idx = self.resolve_target(text, state, in_state)
                if idx is None:
                    return [self.clar("ambiguous", "어떤 메뉴를 취소할까요?")]
                return [{"name": "remove_item", "arguments": {"item_index": idx}}]
            if RE_MODIFY.search(text) and refers_to_state:
                return self._modify(text, state, mentions, in_state)

        if not mentions:
            if RE_CONFIRM.search(text):
                return [{"name": "confirm_order", "arguments": {}}]
            if self._is_generic(text):
                return [self.clar("ambiguous", "어떤 메뉴로 드릴까요?")]
            return [self.clar("off_menu", "죄송하지만 메뉴에 없는 상품이에요. 다른 메뉴로 드릴까요?")]

        return self._order(text, mentions)

    def _is_generic(self, text: str) -> bool:
        if GENERIC_STANDALONE.search(text):
            return True
        compact = text.replace(" ", "")
        for w in GENERIC_WORDS:
            if w.replace(" ", "") in compact:
                return True
        for tok in re.findall(r"[가-힣]{3,}", text):
            for w in GENERIC_WORDS:
                wc = w.replace(" ", "")
                if len(wc) >= 3 and jamo_ratio(tok[: len(wc)], wc) >= 0.8:
                    return True
        return False

    def _modify(self, text, state, mentions, in_state):
        changes = {}
        if RE_INSTEAD.search(text) and len(mentions) >= 2:
            target_mentions = [mentions[0]]
            changes["menu"] = mentions[-1]["menu"]
            idx = self.resolve_target(text, state, target_mentions)
        else:
            idx = self.resolve_target(text, state, in_state)
            masked = text
            for m in mentions:
                masked = masked[: m["start"]] + " " * (m["end"] - m["start"]) + masked[m["end"]:]
            changes = self.parse_options(masked, with_quantity=False)
            q = re.search(r"(\d+|한|두|세|네|다섯)\s*(?:잔|개|조각)?\s*(?:으로|로)", masked)
            if q:
                changes["quantity"] = int(q.group(1)) if q.group(1).isdigit() else NUM_WORD[q.group(1)]
        if idx is None:
            return [self.clar("ambiguous", "어떤 메뉴를 바꿀까요?")]
        if not changes:
            return [self.clar("ambiguous", "어떻게 바꿔 드릴까요?")]
        menu = changes.get("menu") or state[idx - 1].get("menu")
        if menu in self.items and self.violations(menu, changes):
            return [self.clar("impossible", f"{menu}는 그 옵션이 안 돼요. 다른 옵션으로 드릴까요?")]
        return [{"name": "modify_item", "arguments": {"item_index": idx, "changes": changes}}]

    def _order(self, text, mentions):
        # split the utterance into one segment per menu mention
        bounds = [0]
        for k in range(1, len(mentions)):
            lo, hi = mentions[k - 1]["end"], mentions[k]["start"]
            region = text[lo:hi]
            cut = None
            for pat in (RE_SEPARATOR, RE_QTY_WORD, RE_QTY_DIGIT):
                ms = list(pat.finditer(region))
                if ms:
                    cut = lo + ms[-1].end()
                    break
            if cut is None:
                pm = RE_OPTION_PREFIX.search(region)
                cut = lo + pm.start() if pm else hi
            bounds.append(cut)
        bounds.append(len(text))

        calls, clars = [], []
        each_qty = None
        m_each = re.search(r"(하나|한\s*" + UNIT + r"|두\s*" + UNIT + r"|둘|\d+\s*" + UNIT + r"?)\s*씩", text)
        if m_each:
            each_qty = self.parse_quantity(m_each.group(1))
        for k, men in enumerate(mentions):
            seg = text[bounds[k]: bounds[k + 1]]
            s0 = men["start"] - bounds[k]
            masked = seg[:s0] + " " * (men["end"] - men["start"]) + seg[s0 + men["end"] - men["start"]:]
            o = dict(men["implies"])
            o.update(self.parse_options(masked))
            if each_qty and "quantity" not in o:
                o["quantity"] = each_qty
            bad = self.violations(men["menu"], o)
            if bad:
                clars.append(self.clar("impossible", f"죄송하지만 {men['menu']}는 그 옵션으로는 안 돼요. 다르게 드릴까요?"))
                continue
            args = {"menu": men["menu"], **{k2: v for k2, v in o.items() if not (k2 == "quantity" and v == 1)}}
            calls.append({"name": "add_item", "arguments": args})
        return calls + clars[:1]

    def predict(self, case: dict) -> list[dict]:
        """Exam-case interface: uses the last user turn and the order state."""
        user_turns = [m["content"] for m in case["conversation"] if m["role"] == "user"]
        return self.predict_text(user_turns[-1] if user_turns else "", case.get("order_state") or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--state", default="[]", help="current order as JSON list of add_item-style dicts")
    a = ap.parse_args()
    calls = VendingMachine().predict_text(a.text, json.loads(a.state))
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(calls, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
