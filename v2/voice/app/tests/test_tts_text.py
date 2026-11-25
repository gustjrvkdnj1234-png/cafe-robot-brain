"""C1: spoken-text normalization.  .venv-voice\\Scripts\\python.exe -m unittest discover -s v2\\voice\\app\\tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts_text import normalize, shot_delta  # noqa: E402


class Counters(unittest.TestCase):
    def test_native_1_to_20(self):
        self.assertEqual(normalize("아메리카노 1잔"), "아메리카노 한 잔")
        self.assertEqual(normalize("2잔"), "두 잔")
        self.assertEqual(normalize("치즈케이크 3개"), "치즈케이크 세 개")
        self.assertEqual(normalize("4잔이요"), "네 잔이요")
        self.assertEqual(normalize("11개"), "열한 개")
        self.assertEqual(normalize("20잔"), "스무 잔")

    def test_above_20_kept(self):
        self.assertEqual(normalize("25잔"), "25잔")

    def test_space_before_counter(self):
        self.assertEqual(normalize("아메리카노 1 잔"), "아메리카노 한 잔")
        self.assertEqual(normalize("아메리카노 한잔 주세요"), "아메리카노 한 잔 주세요")

    def test_sino_counters_untouched(self):
        self.assertEqual(normalize("5분 걸려요"), "5분 걸려요")
        self.assertEqual(normalize("4500원이에요"), "4500원이에요")


class Shots(unittest.TestCase):
    def test_delta_phrases(self):
        self.assertEqual(shot_delta(-0.5), "반 샷")
        self.assertEqual(shot_delta(1), "한 샷 추가")
        self.assertEqual(shot_delta(0.5), "반 샷 추가")
        self.assertEqual(shot_delta(-1), "한 샷 빼고")
        self.assertEqual(shot_delta(2), "두 샷 추가")
        self.assertEqual(shot_delta(1.5), "한 샷 반 추가")

    def test_in_text(self):
        self.assertEqual(normalize("바닐라라떼(샷 -0.5) 1잔"), "바닐라라떼 반 샷 한 잔")
        self.assertEqual(normalize("샷 +1"), "한 샷 추가")
        self.assertEqual(normalize("-0.5샷"), "반 샷")
        self.assertEqual(normalize("+1샷"), "한 샷 추가")
        self.assertEqual(normalize("shots_delta: -0.5"), "반 샷")
        self.assertEqual(normalize("2샷"), "두 샷")
        self.assertEqual(normalize("카페라떼 반샷 한 잔이요."), "카페라떼 반 샷 한 잔이요.")


class Markup(unittest.TestCase):
    def test_real_free_text_order_list(self):
        # model-written list from the 2026-11-24 session (f68fd4), turn 15
        src = "[현재 주문 내역]\n1. 따뜻한 아메리카노 1잔\n2. 청포도에이드(덜 달게) 1잔\n3. 아이스 바닐라라떼(샷 -0.5, 덜 달게) 1잔"
        self.assertEqual(normalize(src), "따뜻한 아메리카노 한 잔, 청포도에이드 덜 달게 한 잔, 아이스 바닐라라떼 반 샷 덜 달게 한 잔")

    def test_numbering_and_bullets(self):
        self.assertEqual(normalize("1. 아메리카노\n2. 카페라떼"), "아메리카노, 카페라떼")
        self.assertEqual(normalize("1) 아메리카노\n2) 카페라떼"), "아메리카노, 카페라떼")
        self.assertEqual(normalize("- 아메리카노\n- 카페라떼"), "아메리카노, 카페라떼")
        self.assertEqual(normalize("• 레몬에이드\n• 자몽에이드"), "레몬에이드, 자몽에이드")
        self.assertEqual(normalize("① 아메리카노 ② 라떼").startswith("아메리카노"), True)

    def test_markdown_and_symbols(self):
        self.assertEqual(normalize("**추천** 메뉴는 아메리카노예요."), "추천 메뉴는 아메리카노예요.")
        self.assertEqual(normalize("사이즈: 레귤러/라지"), "사이즈: 레귤러, 라지")
        self.assertNotIn("-", normalize("아메리카노 - 따뜻한"))

    def test_sentences_kept(self):
        for s in ("네, 아이스 아메리카노 한 잔이요. 더 필요하신 거 있으세요?",
                  "따뜻한 거요, 아이스요?",
                  "메뉴 안내는 '디저트 뭐 있어요?'처럼 물어봐 주세요.",
                  "주문이 이미 들어갔어요. 변경은 직원 불러 드릴게요."):
            self.assertEqual(normalize(s), s)

    def test_multiline_sentences(self):
        self.assertEqual(normalize("네.\n더 필요하신 거 있으세요?"), "네. 더 필요하신 거 있으세요?")


if __name__ == "__main__":
    unittest.main()


class Sentences(unittest.TestCase):
    def test_split(self):
        from tts_text import sentences
        self.assertEqual(sentences("네, 아이스 아메리카노 한 잔이요. 더 필요하신 거 있으세요?"),
                         ["네, 아이스 아메리카노 한 잔이요.", "더 필요하신 거 있으세요?"])
        self.assertEqual(sentences("따뜻한 거요, 아이스요?"), ["따뜻한 거요, 아이스요?"])
        self.assertEqual(sentences("네. 바꿔 드릴게요. 더 필요하신 거 있으세요?"),
                         ["네. 바꿔 드릴게요.", "더 필요하신 거 있으세요?"])
        self.assertEqual(sentences("1.5배요"), ["1.5배요"])
        self.assertEqual(sentences(""), [""])
