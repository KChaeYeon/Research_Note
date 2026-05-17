import json
import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from generate_note import parse_session_messages


def make_jsonl(messages: list[dict]) -> str:
    """임시 JSONL 파일을 생성하고 경로 반환"""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for msg in messages:
        f.write(json.dumps(msg) + '\n')
    f.close()
    return f.name


def test_parse_user_message_in_range():
    path = make_jsonl([
        {
            "type": "user",
            "timestamp": "2026-05-17T09:30:00.000Z",
            "message": {"role": "user", "content": "EEG 필터링 방법 알려줘"}
        }
    ])
    start = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    results = parse_session_messages([path], start, end)
    assert len(results) == 1
    assert results[0]['role'] == 'user'
    assert results[0]['content'] == 'EEG 필터링 방법 알려줘'
    os.unlink(path)


def test_parse_excludes_out_of_range():
    path = make_jsonl([
        {
            "type": "user",
            "timestamp": "2026-05-17T07:00:00.000Z",
            "message": {"role": "user", "content": "이 메시지는 범위 밖"}
        }
    ])
    start = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    results = parse_session_messages([path], start, end)
    assert len(results) == 0
    os.unlink(path)


def test_parse_assistant_message():
    path = make_jsonl([
        {
            "type": "assistant",
            "timestamp": "2026-05-17T09:31:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Butterworth 필터를 사용하세요."}]
            }
        }
    ])
    start = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    results = parse_session_messages([path], start, end)
    assert len(results) == 1
    assert results[0]['role'] == 'assistant'
    assert 'Butterworth' in results[0]['content']
    os.unlink(path)
