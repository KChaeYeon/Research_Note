import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic


PROJECT_DIR = Path.home() / ".claude" / "projects" / "-mnt-d-00-Project"
NOTES_DIR = Path("/mnt/d/00_Project/research_notes")
STATE_FILE = Path.home() / ".claude" / "research_note_state.json"

MAX_MSG_CHARS = 2000
MODEL = "claude-haiku-4-5-20251001"


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
        content = msg['content']
        if len(content) > MAX_MSG_CHARS:
            content = content[:MAX_MSG_CHARS] + '... [truncated]'
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
        model=MODEL,
        max_tokens=4096,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Strip markdown code fences that Claude sometimes adds despite instructions
        stripped = raw
        if stripped.startswith('```'):
            stripped = '\n'.join(stripped.split('\n')[1:])
        if stripped.endswith('```'):
            stripped = '\n'.join(stripped.split('\n')[:-1])
        stripped = stripped.strip()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            print(f"[research_note][warn] JSON parse failed. raw={raw[:200]}", file=sys.stderr)
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


def write_note_json(note_data: dict, output_dir: Path) -> None:
    """노트 데이터를 data/YYYY-MM-DD.json으로 저장"""
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / f"{note_data['date']}.json"
    out_path.write_text(
        json.dumps(note_data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>연구노트 캘린더</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #333; min-height: 100vh; }}
.page {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
.page-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 28px; }}
.page-title {{ font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }}
.cal-nav-row {{ display: flex; align-items: center; gap: 12px; }}
.cal-nav {{ background: #fff; border: 1px solid #d0d7de; border-radius: 6px; cursor: pointer; padding: 6px 14px; font-size: 0.95rem; color: #333; }}
.cal-nav:hover {{ background: #e8f0fe; }}
.month-label {{ font-size: 1.1rem; font-weight: 600; color: #1a1a2e; min-width: 120px; text-align: center; }}
.cal-grid {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; }}
.cal-day-name {{ text-align: center; font-size: 0.75rem; font-weight: 700; color: #888; padding: 8px 0; letter-spacing: 0.05em; }}
.cal-cell {{ background: #fff; border-radius: 10px; min-height: 100px; padding: 10px; border: 1px solid #e8ecf0; position: relative; cursor: default; transition: box-shadow 0.15s; }}
.cal-cell.has-note {{ cursor: pointer; border-color: #c5d8f5; }}
.cal-cell.has-note:hover {{ box-shadow: 0 2px 12px rgba(21,101,192,0.15); border-color: #1565c0; }}
.cal-cell.today {{ border-color: #1565c0; border-width: 2px; }}
.cal-cell.empty {{ background: transparent; border-color: transparent; }}
.day-num {{ font-size: 0.9rem; font-weight: 600; color: #555; margin-bottom: 6px; }}
.cal-cell.today .day-num {{ color: #1565c0; }}
.cal-cell.has-note .day-num {{ color: #1a1a2e; }}
.kw-list {{ display: flex; flex-wrap: wrap; gap: 3px; margin-top: 4px; }}
.kw-tag {{ font-size: 0.68rem; background: #e3f2fd; color: #1565c0; border-radius: 8px; padding: 2px 7px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }}
.modal-backdrop {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 100; align-items: center; justify-content: center; }}
.modal-backdrop.open {{ display: flex; }}
.modal {{ background: #fff; border-radius: 14px; width: 90%; max-width: 720px; max-height: 88vh; overflow-y: auto; padding: 32px; position: relative; box-shadow: 0 8px 40px rgba(0,0,0,0.18); }}
.modal-close {{ position: absolute; top: 16px; right: 20px; background: none; border: none; font-size: 1.4rem; cursor: pointer; color: #888; }}
.modal-close:hover {{ color: #333; }}
.note-title {{ font-size: 1.3rem; font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }}
.note-meta {{ font-size: 0.85rem; color: #888; margin-bottom: 16px; }}
.topic-tag {{ display: inline-block; background: #e3f2fd; color: #1565c0; border-radius: 12px; padding: 3px 11px; font-size: 0.82rem; margin: 0 4px 6px 0; font-weight: 500; }}
.section-title {{ font-size: 0.95rem; font-weight: 700; color: #333; margin: 22px 0 8px; border-left: 3px solid #1565c0; padding-left: 10px; }}
.summary-text {{ line-height: 1.75; color: #444; }}
.qa-item {{ background: #f8fafc; border: 1px solid #e8ecf0; border-radius: 8px; padding: 14px; margin-bottom: 10px; }}
.qa-q {{ font-weight: 600; color: #1a1a2e; margin-bottom: 6px; font-size: 0.92rem; }}
.qa-a-summary {{ color: #444; line-height: 1.65; font-size: 0.9rem; }}
.qa-toggle {{ font-size: 0.8rem; color: #1565c0; cursor: pointer; margin-top: 7px; display: inline-block; user-select: none; }}
.qa-original {{ display: none; background: #f1f3f5; border-radius: 4px; padding: 10px; margin-top: 8px; font-size: 0.82rem; color: #555; line-height: 1.6; white-space: pre-wrap; }}
.insight-item {{ background: #fff8e1; border-left: 3px solid #ffc107; border-radius: 4px; padding: 10px 14px; margin-bottom: 6px; font-size: 0.9rem; }}
.code-item {{ background: #f1f8e9; border-left: 3px solid #4caf50; border-radius: 4px; padding: 10px 14px; margin-bottom: 6px; font-size: 0.9rem; }}
.loading {{ text-align: center; color: #aaa; padding: 40px 0; }}
</style>
</head>
<body>
<div class="page">
  <div class="page-header">
    <div class="page-title">📓 연구노트</div>
    <div class="cal-nav-row">
      <button class="cal-nav" id="prev-month">&#8592;</button>
      <div class="month-label" id="month-label"></div>
      <button class="cal-nav" id="next-month">&#8594;</button>
    </div>
  </div>
  <div class="cal-grid" id="cal-grid"></div>
</div>

<div class="modal-backdrop" id="modal-backdrop" onclick="closeModal(event)">
  <div class="modal" id="modal">
    <button class="modal-close" onclick="closeModalDirect()">&#10005;</button>
    <div id="modal-content"><div class="loading">로딩 중...</div></div>
  </div>
</div>

<script>
const NOTES_META = {notes_meta_json};

let currentYear, currentMonth;

function init() {{
  const today = new Date();
  currentYear = today.getFullYear();
  currentMonth = today.getMonth();
  renderCalendar();
}}

function renderCalendar() {{
  document.getElementById('month-label').textContent =
    `${{currentYear}}년 ${{currentMonth + 1}}월`;
  const grid = document.getElementById('cal-grid');
  grid.innerHTML = '';

  ['일','월','화','수','목','금','토'].forEach(d => {{
    const h = document.createElement('div');
    h.className = 'cal-day-name';
    h.textContent = d;
    grid.appendChild(h);
  }});

  const firstDay = new Date(currentYear, currentMonth, 1).getDay();
  const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
  const today = new Date();

  for (let i = 0; i < firstDay; i++) {{
    const blank = document.createElement('div');
    blank.className = 'cal-cell empty';
    grid.appendChild(blank);
  }}

  for (let d = 1; d <= daysInMonth; d++) {{
    const dateStr = `${{currentYear}}-${{String(currentMonth+1).padStart(2,'0')}}-${{String(d).padStart(2,'0')}}`;
    const cell = document.createElement('div');
    cell.className = 'cal-cell';

    const isToday = currentYear === today.getFullYear() &&
                    currentMonth === today.getMonth() &&
                    d === today.getDate();
    if (isToday) cell.classList.add('today');

    const numEl = document.createElement('div');
    numEl.className = 'day-num';
    numEl.textContent = d;
    cell.appendChild(numEl);

    if (NOTES_META[dateStr]) {{
      cell.classList.add('has-note');
      const kwList = document.createElement('div');
      kwList.className = 'kw-list';
      NOTES_META[dateStr].slice(0, 3).forEach(kw => {{
        const tag = document.createElement('span');
        tag.className = 'kw-tag';
        tag.textContent = kw;
        kwList.appendChild(tag);
      }});
      cell.appendChild(kwList);
      cell.onclick = () => openNote(dateStr);
    }}

    grid.appendChild(cell);
  }}
}}

async function openNote(dateStr) {{
  document.getElementById('modal-backdrop').classList.add('open');
  document.getElementById('modal-content').innerHTML = '<div class="loading">로딩 중...</div>';
  try {{
    const res = await fetch(`./data/${{dateStr}}.json`);
    if (!res.ok) throw new Error('not found');
    const note = await res.json();
    renderModalNote(note);
  }} catch(e) {{
    document.getElementById('modal-content').innerHTML =
      '<div class="loading">노트를 불러올 수 없습니다.</div>';
  }}
}}

function renderModalNote(note) {{
  const topics = (note.topics||[]).map(t =>
    `<span class="topic-tag">${{escHtml(t)}}</span>`).join('');
  const qaHtml = (note.qa_highlights||[]).map((qa, i) => `
    <div class="qa-item">
      <div class="qa-q">Q. ${{escHtml(qa.q_original||'')}}</div>
      <div class="qa-a-summary">A. ${{escHtml(qa.a_summary||'')}}</div>
      ${{qa.a_original ? `<span class="qa-toggle" onclick="toggleOrig(${{i}})">▼ 원문 보기</span>
        <div class="qa-original" id="qa-orig-${{i}}">${{escHtml(qa.a_original)}}</div>` : ''}}
    </div>`).join('');
  const codes = (note.code_highlights||[]).map(c =>
    `<div class="code-item">🔧 ${{escHtml(c)}}</div>`).join('');
  const insights = (note.insights||[]).map(ins =>
    `<div class="insight-item">💡 ${{escHtml(ins)}}</div>`).join('');

  document.getElementById('modal-content').innerHTML = `
    <div class="note-title">${{escHtml(note.date)}} 연구노트</div>
    <div class="note-meta">${{escHtml(note.start_time)}} ~ ${{escHtml(note.end_time)}}</div>
    <div style="margin-bottom:12px">${{topics}}</div>
    <div class="section-title">오늘의 요약</div>
    <div class="summary-text">${{escHtml(note.summary||'')}}</div>
    ${{qaHtml ? `<div class="section-title">주요 Q&A</div>${{qaHtml}}` : ''}}
    ${{codes ? `<div class="section-title">코드 / 구현</div>${{codes}}` : ''}}
    ${{insights ? `<div class="section-title">핵심 인사이트</div>${{insights}}` : ''}}
  `;
}}

function closeModal(e) {{
  if (e.target === document.getElementById('modal-backdrop')) closeModalDirect();
}}
function closeModalDirect() {{
  document.getElementById('modal-backdrop').classList.remove('open');
}}
function toggleOrig(i) {{
  const el = document.getElementById(`qa-orig-${{i}}`);
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
}}
function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

document.getElementById('prev-month').onclick = () => {{
  currentMonth--; if (currentMonth < 0) {{ currentMonth = 11; currentYear--; }}
  renderCalendar();
}};
document.getElementById('next-month').onclick = () => {{
  currentMonth++; if (currentMonth > 11) {{ currentMonth = 0; currentYear++; }}
  renderCalendar();
}};

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModalDirect(); }});

init();
</script>
</body>
</html>'''


def rebuild_index_html(output_dir: Path) -> None:
    """data/ 디렉토리를 스캔하여 index.html을 전체 재생성.

    NOTES_META에 각 날짜의 topics를 포함해 캘린더 셀에 키워드 표시.
    """
    data_dir = output_dir / "data"
    notes_meta: dict[str, list[str]] = {}
    for p in sorted(data_dir.glob("*.json")):
        if len(p.stem) != 10:
            continue
        try:
            note = json.loads(p.read_text(encoding='utf-8'))
            notes_meta[p.stem] = note.get('topics', [])
        except (json.JSONDecodeError, OSError):
            notes_meta[p.stem] = []
    notes_meta_json = (
        json.dumps(notes_meta, ensure_ascii=False)
        .replace('</', '<\\/')
        .replace('<!--', '<\\!--')
    )
    html = HTML_TEMPLATE.format(notes_meta_json=notes_meta_json)
    (output_dir / "index.html").write_text(html, encoding='utf-8')


def main() -> None:
    """퇴근 시 호출되는 진입점. state file을 읽어 전체 파이프라인 실행."""
    if not STATE_FILE.exists():
        print("[research_note] state file not found. Did you send '출근'?", file=sys.stderr)
        sys.exit(1)

    try:
        state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        start_time = datetime.fromisoformat(state['start_time'].replace('Z', '+00:00'))
        date_str = state['date']
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[research_note] state file invalid: {e}", file=sys.stderr)
        sys.exit(1)
    end_time = datetime.now(tz=timezone.utc)
    start_str = start_time.astimezone().strftime('%H:%M')
    end_str = end_time.astimezone().strftime('%H:%M')

    jsonl_paths = list(str(p) for p in PROJECT_DIR.glob("*.jsonl"))
    if not jsonl_paths:
        print("[research_note] no session files found.", file=sys.stderr)
        sys.exit(1)

    print(f"[research_note] Parsing {len(jsonl_paths)} session files...")
    messages = parse_session_messages(jsonl_paths, start_time, end_time)
    if not messages:
        print("[research_note] No messages found in range.", file=sys.stderr)
        sys.exit(0)

    print(f"[research_note] {len(messages)} messages found. Calling Claude API...")
    conversation = build_conversation_text(messages)
    try:
        note_data = generate_note_data(conversation, date_str, start_str, end_str)
    except Exception as e:
        print(f"[research_note] API call failed: {e}", file=sys.stderr)
        sys.exit(1)

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    write_note_json(note_data, NOTES_DIR)
    rebuild_index_html(NOTES_DIR)

    print(f"[research_note] Done! {NOTES_DIR}/index.html updated.")


if __name__ == '__main__':
    main()
