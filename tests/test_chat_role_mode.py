from types import SimpleNamespace

from pandas import Timestamp

from weclone.data.models import ChatMessage, QaPair
from weclone.data.qa_generator import DataProcessor
from weclone.data.strategies import TimeWindowStrategy


def build_processor(assistant_sender: int) -> DataProcessor:
    processor = DataProcessor.__new__(DataProcessor)
    processor.assistant_sender = assistant_sender
    processor.qa_match_strategy = TimeWindowStrategy(time_window=300, is_single_chat=False)
    processor.system_prompt = "test system"
    processor.QaPair = QaPair
    processor.relations = {}
    processor.c = SimpleNamespace(add_time=False, add_relation=False)
    processor.config = SimpleNamespace(messages_max_length=2048, max_image_num=2)
    return processor


def build_message(is_sender: int, text: str, ts: str, talker: str) -> ChatMessage:
    return ChatMessage(
        id=1,
        MsgSvrID=f"{is_sender}-{ts}",
        type_name="text",
        is_sender=is_sender,
        talker=talker,
        msg=text,
        src="",
        CreateTime=Timestamp(ts),
        room_name=None,
        is_forward=False,
        modality=None,
    )


def test_match_qa_defaults_to_self_as_assistant():
    processor = build_processor(assistant_sender=1)
    messages = [
        build_message(is_sender=0, text="你晚上吃啥", ts="2025-01-01 10:00:00", talker="friend"),
        build_message(is_sender=1, text="我想喝粥", ts="2025-01-01 10:01:00", talker="me"),
    ]

    result = processor.match_qa(messages)

    assert len(result) == 1
    qa = result[0]
    assert qa.messages[0].role == "user"
    assert qa.messages[0].content == "你晚上吃啥"
    assert qa.messages[1].role == "assistant"
    assert qa.messages[1].content == "我想喝粥"


def test_match_qa_can_reverse_chat_role_direction():
    processor = build_processor(assistant_sender=0)
    messages = [
        build_message(is_sender=1, text="你晚上吃啥", ts="2025-01-01 10:00:00", talker="me"),
        build_message(is_sender=0, text="我想喝粥", ts="2025-01-01 10:01:00", talker="friend"),
    ]

    result = processor.match_qa(messages)

    assert len(result) == 1
    qa = result[0]
    assert qa.messages[0].role == "user"
    assert qa.messages[0].content == "你晚上吃啥"
    assert qa.messages[1].role == "assistant"
    assert qa.messages[1].content == "我想喝粥"