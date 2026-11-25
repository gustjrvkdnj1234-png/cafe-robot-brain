"""C2: hint text for Whisper.  .venv-voice\Scripts\python.exe -m unittest discover -s v2\voice\app\tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt_hints import hint_text  # noqa: E402


class Hints(unittest.TestCase):
    def test_words(self):
        h = hint_text()
        for w in ("아메리카노", "바닐라라떼", "아아", "뜨아", "아바라", "반샷", "캐리어", "포장", "덜 달게", "오트밀크"):
            self.assertIn(w, h)

    def test_no_spelling_variants(self):
        words = [w.strip(" .") for w in hint_text().replace(". ", ", ").split(", ")]
        for w in ("라테", "레모네이드", "콜드 브루", "카페 라떼"):
            self.assertNotIn(w, words)
