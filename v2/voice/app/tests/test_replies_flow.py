"""A1 / A2 / B1 reply logic without the model.  .venv-voice\\Scripts\\python.exe -m unittest discover -s v2\\voice\\app\\tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replies  # noqa: E402
from tts_text import normalize  # noqa: E402

AA = {"menu": "아메리카노", "temperature": "iced", "quantity": 1}
VL = {"menu": "바닐라라떼", "temperature": "iced", "shots_delta": -0.5, "sweetness": "less", "quantity": 1}
CONFIRM = [{"name": "confirm_order", "arguments": {"dine_option": "dine_in"}}]


def say(r):
    return normalize(r["spoken"])


class Readback(unittest.TestCase):
    def test_confirm_while_ordering_reads_back_and_asks_dine(self):
        r = replies.build_reply(CONFIRM, "", [AA, VL], phase="ordering", dine=None)
        self.assertFalse(r["confirmed"])
        self.assertEqual(r["phase"], "awaiting")
        self.assertEqual(say(r), "아이스 아메리카노 한 잔, 아이스 바닐라라떼 반 샷 덜 달게 한 잔이요. 드시고 가세요, 포장이세요?")

    def test_known_takeout(self):
        r = replies.build_reply(CONFIRM, "", [AA], phase="ordering", dine="takeout")
        self.assertEqual(say(r), "아이스 아메리카노 한 잔이요. 포장으로 주문할까요?")

    def test_add_and_done_in_one_turn_reads_back_once(self):
        calls = [{"name": "add_item", "arguments": {"menu": "바닐라라떼", "temperature": "iced"}},
                 {"name": "confirm_order", "arguments": {"dine_option": "takeout"}}]
        r = replies.build_reply(calls, "", [], phase="ordering", dine="takeout")
        self.assertEqual(r["phase"], "awaiting")
        self.assertEqual(say(r), "네, 아이스 바닐라라떼 한 잔이요. 포장으로 주문할까요?")

    def test_yes_without_dine_asks_again(self):
        r = replies.build_reply(CONFIRM, "", [AA], phase="awaiting", dine=None)
        self.assertFalse(r["confirmed"])
        self.assertEqual(r["phase"], "awaiting")
        self.assertEqual(say(r), "드시고 가세요, 포장이세요?")

    def test_yes_with_dine_confirms(self):
        r = replies.build_reply(CONFIRM, "", [AA], phase="awaiting", dine="dine_in")
        self.assertTrue(r["confirmed"])
        self.assertEqual(r["phase"], "confirmed")
        r = replies.build_reply([{"name": "confirm_order", "arguments": {"dine_option": "takeout"}}], "", [AA],
                                phase="awaiting", dine=None)
        self.assertTrue(r["confirmed"])
        self.assertEqual(r["dine_option"], "takeout")

    def test_change_while_awaiting_stays_unconfirmed(self):
        r = replies.build_reply([{"name": "modify_item", "arguments": {"item_index": 1, "changes": {"size": "large"}}}],
                                "", [AA], phase="awaiting", dine="dine_in")
        self.assertFalse(r["confirmed"])
        self.assertEqual(r["phase"], "ordering")
        self.assertEqual(r["state"][0]["size"], "large")

    def test_empty_order(self):
        r = replies.build_reply(CONFIRM, "", [], phase="ordering")
        self.assertEqual(r["phase"], "ordering")
        self.assertIn("confirm_empty", r["kinds"])


class Locked(unittest.TestCase):
    def test_changes_after_confirmation_not_applied(self):
        for call in ({"name": "add_item", "arguments": {"menu": "카페라떼", "temperature": "hot"}},
                     {"name": "modify_item", "arguments": {"item_index": 1, "changes": {"size": "large"}}},
                     {"name": "remove_item", "arguments": {"item_index": 1}}):
            r = replies.build_reply([call], "", [AA], phase="confirmed", dine="dine_in")
            self.assertEqual(r["state"], [AA], call["name"])
            self.assertEqual(say(r), "주문이 이미 들어갔어요. 변경은 직원 불러 드릴게요.")
            self.assertTrue(r["staff_reason"])
            self.assertEqual(r["phase"], "confirmed")


class ReadOrder(unittest.TestCase):
    def test_read_order_tool(self):
        r = replies.build_reply([{"name": "read_order", "arguments": {}}], "", [AA, VL])
        self.assertEqual(say(r), "지금 아이스 아메리카노 한 잔, 아이스 바닐라라떼 반 샷 덜 달게 한 잔이에요.")

    def test_model_written_list_is_replaced(self):
        text = "[현재 주문 내역]\n1. 따뜻한 아메리카노 1잔\n2. 청포도에이드(덜 달게) 1잔"
        r = replies.build_reply([], text, [AA])
        self.assertIn("free_text_replaced", r["kinds"])
        self.assertEqual(say(r), "지금 아이스 아메리카노 한 잔이에요.")

    def test_empty(self):
        r = replies.build_reply([{"name": "read_order", "arguments": {}}], "", [])
        self.assertEqual(say(r), "아직 담긴 게 없어요.")

    def test_dessert_counter(self):
        r = replies.build_reply([{"name": "read_order", "arguments": {}}], "", [{"menu": "치즈케이크", "quantity": 2}])
        self.assertEqual(say(r), "지금 치즈케이크 두 개예요.")


class Temperature(unittest.TestCase):
    LATTE = [{"name": "add_item", "arguments": {"menu": "카페라떼", "shots_delta": -0.5}}]

    def test_hot_or_iced_without_temperature_is_asked(self):
        r = replies.build_reply(self.LATTE, "", [AA], temp_said=False)
        self.assertEqual(r["state"], [AA])
        self.assertEqual(say(r), "따뜻한 거요, 아이스요?")
        self.assertEqual([c["name"] for c in r["history_calls"]], ["ask_clarification"])

    def test_with_other_items(self):
        calls = [{"name": "add_item", "arguments": {"menu": "청포도에이드"}}] + self.LATTE
        r = replies.build_reply(calls, "", [], temp_said=False)
        self.assertEqual(len(r["state"]), 1)
        self.assertEqual(say(r), "네, 청포도에이드 한 잔이요. 카페라떼는 따뜻한 거요, 아이스요?")

    def test_customer_said_it_or_single_temperature_item(self):
        self.assertEqual(len(replies.build_reply(self.LATTE, "", [], temp_said=True)["state"]), 1)
        r = replies.build_reply([{"name": "add_item", "arguments": {"menu": "에스프레소"}}], "", [], temp_said=False)
        self.assertEqual(len(r["state"]), 1)


class OptionQuestion(unittest.TestCase):
    def test_option_item(self):
        r = replies.build_reply([{"name": "inform", "arguments": {"topic": "item_detail", "item": "sweetness"}}], "", [])
        self.assertTrue(say(r).startswith("당도는 덜 달게, 보통, 더 달게"))


if __name__ == "__main__":
    unittest.main()
