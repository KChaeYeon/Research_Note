import json
from datetime import datetime, timezone
from pathlib import Path

import anthropic


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


SUMMARY_SYSTEM = """당신은 의공학 대학원생의 연구 보조자입니다.
제공된 Claude 대화 내용을 분석하여 연구노트 형식의 JSON을 반환하세요.
JSON 외의 다른 텍스트는 절대 출력하지 마세요."""

SUMMARY_PROMPT_TMPL = """다음은 오늘({date}) 연구 대화 내용입니다:

{conversation}

위 대화를 아래 JSON 형식으로 정리하세요:
{{
  "topics": ["주제1", "주제2"],
  "summary": "하루 연구 전반 요약 (3-5문장)",
  "qa_highlights": [
    {{
      "q_original": "사용자 질문 원문",
      "a_summary": "핵심 답변 요약 (2-3문장)",
      "a_original": "어시스턴트 응답 원문 앞 500자"
    }}
  ],
  "code_highlights": ["구현/분석한 알고리즘 또는 코드 설명"],
  "insights": ["핵심 인사이트 또는 결론"]
}}"""


def build_conversation_text(messages: list[dict]) -> str:
    """메시지 목록을 API 프롬프트용 텍스트로 변환"""
    lines = []
    for msg in messages:
        role_label = '[사용자]' if msg['role'] == 'user' else '[어시스턴트]'
        content = msg['content'][:2000]
        lines.append(f"{role_label}\n{content}")
    return '\n\n'.join(lines)


def generate_note_data(
    conversation_text: str,
    date: str,
    start_str: str,
    end_str: str,
) -> dict:
    """Claude Haiku API를 호출하여 구조화된 연구노트 데이터 반환"""
    client = anthropic.Anthropic()
    prompt = SUMMARY_PROMPT_TMPL.format(
        date=date,
        conversation=conversation_text,
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "topics": [],
            "summary": raw[:500],
            "qa_highlights": [],
            "code_highlights": [],
            "insights": [],
        }
    data['date'] = date
    data['start_time'] = start_str
    data['end_time'] = end_str
    return data
