import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path.home() / ".claude" / "projects" / "-mnt-d-00-Project"
NOTES_DIR = Path("/mnt/d/00_Project/research_notes")
STATE_FILE = Path.home() / ".claude" / "research_note_state.json"


def parse_session_messages(
    jsonl_paths: list[str | Path],
    start_time: datetime,
    end_time: datetime
) -> list[dict]:
    """JSONL 파일 목록에서 start_time~end_time 사이 메시지를 추출.

    Returns list of {"role": str, "content": str, "timestamp": datetime}
    """
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    messages = []
    for path in jsonl_paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get('type')
                ts_str = obj.get('timestamp')
                if msg_type not in ('user', 'assistant') or not ts_str:
                    continue

                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if not (start_time <= ts <= end_time):
                    continue

                message = obj.get('message', {})
                content_raw = message.get('content', '')

                if isinstance(content_raw, list):
                    content = ' '.join(
                        c.get('text', '') for c in content_raw
                        if isinstance(c, dict) and c.get('type') == 'text'
                    )
                else:
                    content = str(content_raw)

                if not content.strip():
                    continue

                messages.append({
                    'role': msg_type,
                    'content': content,
                    'timestamp': ts,
                })

    messages.sort(key=lambda m: m['timestamp'])
    return messages
