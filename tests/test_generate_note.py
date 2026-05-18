import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from generate_note import (
    parse_session_messages,
    build_conversation_text,
    generate_note_data,
    write_note_json,
    rebuild_index_html,
    SUMMARY_SYSTEM,
)


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


def test_build_conversation_text():
    messages = [
        {'role': 'user', 'content': 'EEG 필터링 어떻게 해?', 'timestamp': None},
        {'role': 'assistant', 'content': 'Butterworth 필터를 쓰세요.', 'timestamp': None},
    ]
    text = build_conversation_text(messages)
    assert '[사용자]' in text
    assert 'EEG 필터링' in text
    assert '[어시스턴트]' in text
    assert 'Butterworth' in text


def test_generate_note_data_returns_required_keys():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(
        text=json.dumps({
            "topics": ["EEG"],
            "summary": "오늘 EEG 필터링을 공부했다.",
            "qa_highlights": [],
            "code_highlights": [],
            "insights": []
        })
    )]

    with patch('generate_note.anthropic.Anthropic') as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_response

        result = generate_note_data(
            conversation_text="[사용자] EEG?\n[어시스턴트] Butterworth.",
            date="2026-05-17",
            start_str="09:00",
            end_str="18:00",
        )

    assert 'topics' in result
    assert 'summary' in result
    assert 'qa_highlights' in result
    assert result['date'] == '2026-05-17'
    assert result['start_time'] == '09:00'
    assert result['end_time'] == '18:00'

    instance.messages.create.assert_called_once()
    call_kwargs = instance.messages.create.call_args.kwargs
    assert call_kwargs['model'] == 'claude-haiku-4-5-20251001'
    assert call_kwargs['system'] == SUMMARY_SYSTEM
    assert call_kwargs['max_tokens'] == 4096


def test_generate_note_data_json_fallback():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="this is not json at all")]

    with patch('generate_note.anthropic.Anthropic') as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_response

        result = generate_note_data(
            conversation_text="some conversation",
            date="2026-05-17",
            start_str="09:00",
            end_str="18:00",
        )

    assert result['topics'] == []
    assert result['qa_highlights'] == []
    assert result['code_highlights'] == []
    assert result['insights'] == []
    assert result['date'] == '2026-05-17'
    assert 'summary_3lines' in result
    assert result.get('theme') == 'misc'


def test_write_note_json_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        note = {
            "date": "2026-05-17",
            "start_time": "09:00",
            "end_time": "18:00",
            "topics": ["EEG"],
            "summary": "테스트 요약",
            "qa_highlights": [],
            "code_highlights": [],
            "insights": [],
        }
        write_note_json(note, Path(tmpdir))
        out = data_dir / "2026-05-17.json"
        assert out.exists()
        loaded = json.loads(out.read_text(encoding='utf-8'))
        assert loaded['date'] == '2026-05-17'


def test_rebuild_index_html_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        data_dir = tmppath / "data"
        data_dir.mkdir()
        notes = {
            "2026-05-16": ["EEG", "필터링"],
            "2026-05-17": ["CNN", "MRI 분류"],
        }
        for d, topics in notes.items():
            (data_dir / f"{d}.json").write_text(
                json.dumps({"date": d, "topics": topics, "summary": ""}),
                encoding='utf-8'
            )
        rebuild_index_html(tmppath)
        assert (tmppath / "index.html").exists()
        html = (tmppath / "index.html").read_text(encoding='utf-8')
        assert "2026-05-17" in html
        assert "2026-05-16" in html
        assert "CNN" in html
        assert "EEG" in html
