import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from generate_note import parse_session_messages


def test_parse_user_message_in_range(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({
        "type": "user",
        "timestamp": "2026-05-17T09:30:00.000Z",
        "message": {"role": "user", "content": "EEG 필터링 방법 알려줘"}
    }) + "\n", encoding='utf-8')
    start = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    results = parse_session_messages([str(path)], start, end)
    assert len(results) == 1
    assert results[0]['role'] == 'user'
    assert results[0]['content'] == 'EEG 필터링 방법 알려줘'


def test_parse_excludes_out_of_range(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({
        "type": "user",
        "timestamp": "2026-05-17T07:00:00.000Z",
        "message": {"role": "user", "content": "이 메시지는 범위 밖"}
    }) + "\n", encoding='utf-8')
    start = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    results = parse_session_messages([str(path)], start, end)
    assert len(results) == 0


def test_parse_assistant_message(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({
        "type": "assistant",
        "timestamp": "2026-05-17T09:31:00.000Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Butterworth 필터를 사용하세요."}]
        }
    }) + "\n", encoding='utf-8')
    start = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    results = parse_session_messages([str(path)], start, end)
    assert len(results) == 1
    assert results[0]['role'] == 'assistant'
    assert 'Butterworth' in results[0]['content']
