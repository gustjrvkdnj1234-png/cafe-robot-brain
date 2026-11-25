"""A1 short answers that skip the model.  .venv-voice\\Scripts\\python.exe -m unittest discover -s v2\\voice\\app\\tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flow import awaiting_answer, dine_of, is_done  # noqa: E402


class Flow(unittest.TestCase):
    def test_dine(self):
        self.assertEqual(dine_of("포장이요"), "takeout")
        self.assertEqual(dine_of("테이크아웃 할게요"), "takeout")
        self.assertEqual(dine_of("가져갈게요"), "takeout")
        self.assertEqual(dine_of("먹고 갈게요"), "dine_in")
        self.assertEqual(dine_of("매장에서 먹을게요"), "dine_in")
        self.assertIsNone(dine_of("아이스 아메리카노 한 잔"))

    def test_awaiting_yes(self):
        for s in ("네", "네.", "네 맞아요", "포장이요", "네, 포장이요", "먹고 갈게요", "매장에서 먹을게요", "네 그렇게 해 주세요",
                  "아 네 테이크아웃이요"):
            self.assertEqual(awaiting_answer(s), "yes", s)

    def test_awaiting_no(self):
        for s in ("아니요", "아니요.", "잠깐만요"):
            self.assertEqual(awaiting_answer(s), "no", s)

    def test_awaiting_other_goes_to_model(self):
        for s in ("네 그리고 라떼 하나 더요", "아니요 아이스로 바꿔 주세요", "포장이요 근데 사이즈 큰 걸로", "아메리카노 빼 주세요"):
            self.assertIsNone(awaiting_answer(s), s)

    def test_done(self):
        for s in ("아니 없어요", "없어요", "이게 다예요", "주문할게요", "네 주문할게요", "그게 다예요."):
            self.assertTrue(is_done(s), s)
        for s in ("안녕 없어요", "라떼 주문할게요", "아메리카노 없어요?", "네"):
            self.assertFalse(is_done(s), s)


if __name__ == "__main__":
    unittest.main()
