import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic


PROJECT_DIR = Path.home() / ".claude" / "projects" / "-mnt-d-00-Project"
NOTES_DIR = Path("/mnt/d/00_Project/research_notes")
STATE_FILE = Path.home() / ".claude" / "research_note_state.json"
WEEKLY_GOALS_FILE = NOTES_DIR / "data" / "weekly_goals.json"

MAX_MSG_CHARS = 2000
MODEL = "claude-haiku-4-5-20251001"

THEME_COLORS = {
    "rsp":        {"bg": "#fce4ec", "text": "#c2185b", "border": "#f48fb1", "accent": "#e91e63", "label": "호흡 조절"},
    "ultrasound": {"bg": "#f1f8e9", "text": "#558b2f", "border": "#aed581", "accent": "#8bc34a", "label": "초음파·혈압"},
    "sleep":      {"bg": "#f3e5f5", "text": "#6a1b9a", "border": "#ce93d8", "accent": "#9c27b0", "label": "수면·초저주파"},
    "misc":       {"bg": "#f5f5f5", "text": "#616161", "border": "#bdbdbd", "accent": "#9e9e9e", "label": "기타"},
}


def parse_session_messages(
    jsonl_paths: list[str | Path],
    start_time: datetime,
    end_time: datetime
) -> list[dict]:
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

                messages.append({'role': msg_type, 'content': content, 'timestamp': ts})

    messages.sort(key=lambda m: m['timestamp'])
    return messages


SUMMARY_SYSTEM = """당신은 의공학 대학원생의 연구 보조자입니다.
제공된 Claude 대화 내용을 분석하여 연구노트 형식의 JSON을 반환하세요.
JSON 외의 다른 텍스트는 절대 출력하지 마세요."""

SUMMARY_PROMPT_TMPL = """다음은 오늘({date}) 연구 대화 내용입니다:

{conversation}

위 대화를 아래 JSON 형식으로 정리하세요:
{{
  "theme": "misc",
  "topics": ["주제1", "주제2"],
  "summary_3lines": [
    "오늘 수행한 핵심 작업 한 줄 요약",
    "주요 결과 또는 해결한 문제",
    "다음 단계 또는 남은 과제"
  ],
  "qa_highlights": [
    {{
      "q_original": "사용자 질문 원문",
      "a_summary": "핵심 답변 요약 (2-3문장)",
      "a_original": "어시스턴트 응답 원문 앞 500자"
    }}
  ],
  "code_highlights": ["구현/분석한 알고리즘 또는 코드 설명"],
  "insights": ["핵심 인사이트 또는 결론"]
}}

theme 분류 기준 (정확히 하나 선택):
- "rsp": 호흡 조절, hemodynamics, RSP_hemodynamics 폴더 관련
- "ultrasound": 초음파, 혈압 추정, BP estimation 관련
- "sleep": 수면, 초저주파 리듬, Sleep_InfraSlow_RSP 폴더 관련
- "misc": 위 세 카테고리 외 모든 작업

summary_3lines: 각각 40자 이내 한국어 한 문장, 정확히 3개"""


def build_conversation_text(messages: list[dict]) -> str:
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
    daily_goal: str = "",
) -> dict:
    client = anthropic.Anthropic()
    prompt = SUMMARY_PROMPT_TMPL.format(date=date, conversation=conversation_text)
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
                "theme": "misc",
                "topics": [],
                "summary_3lines": [raw[:100]] if raw else [],
                "qa_highlights": [],
                "code_highlights": [],
                "insights": [],
            }
    data.setdefault('theme', 'misc')
    data.setdefault('summary_3lines', [])
    data['date'] = date
    data['start_time'] = start_str
    data['end_time'] = end_str
    data['daily_goal'] = daily_goal
    return data


def write_note_json(note_data: dict, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / f"{note_data['date']}.json"
    out_path.write_text(json.dumps(note_data, ensure_ascii=False, indent=2), encoding='utf-8')


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chaeyeon\'s Research Note</title>
<link rel="manifest" href="manifest.json">
<meta name="theme-color" content="#1a1a2e">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="연구노트">
<link rel="apple-touch-icon" href="icons/icon-192.png">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", system-ui, -apple-system, sans-serif; background: #f0f4f8; color: #333; min-height: 100vh; }
.page { max-width: 1240px; margin: 0 auto; padding: 28px 20px; }
.page-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 16px; }
.page-title { font-size: 1.7rem; font-weight: 800; color: #1a1a2e; letter-spacing: -0.02em; }
.page-subtitle { font-size: 0.95rem; color: #999; margin-top: 3px; }
.cal-nav-row { display: flex; align-items: center; gap: 8px; }
.cal-nav { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; cursor: pointer; padding: 5px 16px; font-size: 1rem; color: #555; transition: all 0.15s; }
.cal-nav:hover { background: #e8f0fe; border-color: #aaa; }
.month-label { font-size: 1.2rem; font-weight: 700; color: #1a1a2e; min-width: 110px; text-align: center; }
.theme-legend { display: flex; align-items: center; gap: 16px; margin-bottom: 14px; flex-wrap: wrap; }
.legend-label { font-size: 0.85rem; font-weight: 700; color: #999; letter-spacing: 0.06em; text-transform: uppercase; margin-right: 4px; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.88rem; color: #666; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.cal-grid { display: grid; grid-template-columns: 120px repeat(7, 1fr); gap: 5px; }
.cal-week-header { display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; color: #aaa; padding: 8px 0; letter-spacing: 0.05em; text-transform: uppercase; }
.cal-day-name { text-align: center; font-size: 0.88rem; font-weight: 700; color: #999; padding: 8px 0; letter-spacing: 0.04em; }
.cal-week-goal { background: #fffde7; border: 1.5px solid #ffe082; border-radius: 10px; padding: 8px 7px; display: flex; align-items: center; justify-content: center; text-align: center; min-height: 96px; }
.week-goal-text { font-size: 0.85rem; color: #5d4037; line-height: 1.6; font-weight: 500; white-space: pre-line; text-align: left; }
.week-goal-empty { font-size: 0.78rem; color: #ccc; font-style: italic; }
.cal-cell { background: #fff; border-radius: 10px; min-height: 96px; padding: 8px 7px; border: 1.5px solid #e8ecf0; position: relative; cursor: default; transition: box-shadow 0.15s, border-color 0.15s; }
.cal-cell.empty { background: transparent; border-color: transparent; }
.cal-cell.today { border-color: #1565c0 !important; border-width: 2px; }
.day-num { font-size: 1rem; font-weight: 600; color: #777; margin-bottom: 5px; }
.cal-cell.today .day-num { color: #1565c0; font-weight: 800; }
.cal-cell.has-note { cursor: pointer; }
.cal-cell.has-note:hover { transform: translateY(-1px); }
@media (max-width: 700px) {
  .page { padding: 16px 10px; }
  .page-title { font-size: 1.2rem; }
  .page-subtitle { font-size: 0.82rem; }
  .cal-grid { grid-template-columns: 70px repeat(7, 1fr); gap: 3px; }
  .cal-week-goal { min-height: 64px; padding: 5px 4px; }
  .week-goal-text { font-size: 0.72rem; }
  .cal-cell { min-height: 64px; padding: 5px 4px; }
  .day-num { font-size: 0.85rem; }
  .kw-tag { font-size: 0.68rem; padding: 2px 5px; }
  .theme-legend { gap: 10px; }
  .legend-item { font-size: 0.78rem; }
  .modal-backdrop { padding: 0; align-items: flex-end; overflow: hidden; }
  .modal { border-radius: 18px 18px 0 0; max-height: 90vh; overflow-x: hidden; overflow-y: auto; -webkit-overflow-scrolling: touch; }
  .modal-header { padding: 20px 16px 14px; }
  .modal-body { padding: 14px 16px 28px; }
  .modal-date { font-size: 1.3rem; }
}
.cal-cell.has-note.theme-rsp        { border-color: #f48fb1; }
.cal-cell.has-note.theme-rsp:hover  { box-shadow: 0 4px 16px rgba(233,30,99,0.18); }
.cal-cell.has-note.theme-rsp .kw-tag { background: #fce4ec; color: #c2185b; }
.cal-cell.has-note.theme-ultrasound        { border-color: #aed581; }
.cal-cell.has-note.theme-ultrasound:hover  { box-shadow: 0 4px 16px rgba(139,195,74,0.22); }
.cal-cell.has-note.theme-ultrasound .kw-tag { background: #f1f8e9; color: #558b2f; }
.cal-cell.has-note.theme-sleep        { border-color: #ce93d8; }
.cal-cell.has-note.theme-sleep:hover  { box-shadow: 0 4px 16px rgba(156,39,176,0.18); }
.cal-cell.has-note.theme-sleep .kw-tag { background: #f3e5f5; color: #6a1b9a; }
.cal-cell.has-note.theme-misc        { border-color: #bdbdbd; }
.cal-cell.has-note.theme-misc:hover  { box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
.cal-cell.has-note.theme-misc .kw-tag { background: #f5f5f5; color: #616161; }
.kw-list { display: flex; flex-wrap: wrap; gap: 3px; }
.kw-tag { font-size: 0.78rem; border-radius: 8px; padding: 3px 7px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.modal-backdrop { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 100; align-items: flex-start; justify-content: center; padding: 48px 16px 48px; overflow-y: auto; }
.modal-backdrop.open { display: flex; }
.modal { background: #fff; border-radius: 18px; width: 100%; max-width: 740px; overflow-x: hidden; overflow-y: auto; box-shadow: 0 24px 64px rgba(0,0,0,0.22); position: relative; }
.modal-close { position: absolute; top: 14px; right: 16px; background: rgba(0,0,0,0.08); border: none; font-size: 1.1rem; cursor: pointer; color: #555; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; transition: background 0.15s; z-index: 1; }
.modal-close:hover { background: rgba(0,0,0,0.15); color: #111; }
.modal-header { padding: 26px 28px 20px; }
.modal-theme-badge { display: inline-flex; align-items: center; padding: 4px 13px; border-radius: 20px; font-size: 0.82rem; font-weight: 700; color: #fff; margin-bottom: 10px; letter-spacing: 0.04em; text-transform: uppercase; }
.modal-date { font-size: 1.65rem; font-weight: 800; color: #1a1a2e; line-height: 1.2; margin-bottom: 5px; }
.modal-time { font-size: 0.97rem; color: #888; }
.modal-goal { font-size: 1rem; color: #333; font-weight: 500; margin-top: 10px; padding: 10px 16px; background: rgba(255,255,255,0.65); border-radius: 9px; backdrop-filter: blur(4px); }
.topic-tags { margin-top: 13px; display: flex; flex-wrap: wrap; gap: 6px; }
.topic-tag { display: inline-flex; align-items: center; border-radius: 12px; padding: 5px 14px; font-size: 0.9rem; font-weight: 700; }
.modal-divider { height: 1px; background: rgba(0,0,0,0.07); margin: 0 28px; }
.modal-body { padding: 20px 28px 32px; }
.section-title { font-size: 0.9rem; font-weight: 800; color: #888; margin: 22px 0 10px; letter-spacing: 0.08em; text-transform: uppercase; display: flex; align-items: center; gap: 8px; }
.section-title::after { content: ""; flex: 1; height: 1px; background: #eee; }
.summary-3lines { display: flex; flex-direction: column; gap: 7px; }
.summary-line { display: flex; align-items: flex-start; gap: 12px; padding: 12px 16px; background: #f8fafc; border-radius: 9px; font-size: 1rem; line-height: 1.7; color: #444; }
.summary-num { min-width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.78rem; font-weight: 800; color: #fff; flex-shrink: 0; margin-top: 2px; }
.qa-item { background: #f8fafc; border: 1px solid #eaecf0; border-radius: 10px; padding: 16px 18px; margin-bottom: 8px; }
.qa-q { font-weight: 700; color: #1a1a2e; margin-bottom: 8px; font-size: 0.97rem; line-height: 1.55; }
.qa-a-summary { color: #555; line-height: 1.75; font-size: 0.93rem; }
.qa-toggle { font-size: 0.85rem; color: #1565c0; cursor: pointer; margin-top: 10px; display: inline-flex; align-items: center; gap: 3px; user-select: none; }
.qa-toggle:hover { text-decoration: underline; }
.qa-original { display: none; background: #f1f3f5; border-radius: 7px; padding: 12px 14px; margin-top: 8px; font-size: 0.88rem; color: #555; line-height: 1.68; white-space: pre-wrap; }
.highlight-item { border-radius: 8px; padding: 11px 15px; margin-bottom: 7px; font-size: 0.95rem; line-height: 1.68; }
.code-item { background: #f1f8e9; border-left: 3px solid #66bb6a; }
.insight-item { background: #fff8e1; border-left: 3px solid #ffca28; }
.loading { text-align: center; color: #bbb; padding: 50px 0; font-size: 1rem; }
</style>
</head>
<body>
<div class="page">
  <div class="page-header">
    <div>
      <div class="page-title">📓 Chaeyeon\'s Research Note</div>
      <div class="page-subtitle">의공학 연구 일지</div>
    </div>
    <div class="cal-nav-row">
      <button class="cal-nav" id="prev-month">&#8592;</button>
      <div class="month-label" id="month-label"></div>
      <button class="cal-nav" id="next-month">&#8594;</button>
    </div>
  </div>
  <div class="theme-legend">
    <span class="legend-label">Theme</span>
    <div class="legend-item"><div class="legend-dot" style="background:#e91e63"></div>호흡 조절</div>
    <div class="legend-item"><div class="legend-dot" style="background:#8bc34a"></div>초음파·혈압</div>
    <div class="legend-item"><div class="legend-dot" style="background:#9c27b0"></div>수면·초저주파</div>
    <div class="legend-item"><div class="legend-dot" style="background:#9e9e9e"></div>기타</div>
  </div>
  <div class="cal-grid" id="cal-grid"></div>
</div>

<div class="modal-backdrop" id="modal-backdrop" onclick="closeModal(event)">
  <div class="modal" id="modal">
    <button class="modal-close" onclick="closeModalDirect()">&#10005;</button>
    <div id="modal-content"><div class="loading">로딩 중…</div></div>
  </div>
</div>

<script>
const NOTES_META = __NOTES_META__;
const WEEKLY_GOALS = __WEEKLY_GOALS__;
const THEME_COLORS = __THEME_COLORS__;
const NOTES_DATA = __NOTES_DATA__;

let currentYear, currentMonth;

function getISOWeekKey(year, month, day) {
  const d = new Date(Date.UTC(year, month, day));
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
  const ys = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const w = Math.ceil((((d - ys) / 86400000) + 1) / 7);
  return `${d.getUTCFullYear()}-W${String(w).padStart(2, '0')}`;
}

function init() {
  const t = new Date();
  currentYear = t.getFullYear();
  currentMonth = t.getMonth();
  renderCalendar();
}

function renderCalendar() {
  document.getElementById('month-label').textContent = `${currentYear}년 ${currentMonth + 1}월`;
  const grid = document.getElementById('cal-grid');
  grid.innerHTML = '';

  const wh = document.createElement('div');
  wh.className = 'cal-week-header';
  wh.textContent = '주간 목표';
  grid.appendChild(wh);

  ['일','월','화','수','목','금','토'].forEach(d => {
    const h = document.createElement('div');
    h.className = 'cal-day-name';
    h.textContent = d;
    grid.appendChild(h);
  });

  const firstDay = new Date(currentYear, currentMonth, 1).getDay();
  const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
  const today = new Date();

  const slots = [];
  for (let i = 0; i < firstDay; i++) slots.push(null);
  for (let d = 1; d <= daysInMonth; d++) slots.push(d);
  while (slots.length % 7 !== 0) slots.push(null);

  for (let w = 0; w < slots.length; w += 7) {
    const week = slots.slice(w, w + 7);
    const firstNonNull = week.find(d => d !== null);
    const monday = week[1]; // ISO weeks start Monday; use Monday for weekKey alignment
    const keyDay = (monday !== null && monday !== undefined) ? monday : firstNonNull;
    const weekKey = keyDay ? getISOWeekKey(currentYear, currentMonth, keyDay) : '';

    const goalCell = document.createElement('div');
    goalCell.className = 'cal-week-goal';
    const goalText = WEEKLY_GOALS[weekKey] || '';
    goalCell.innerHTML = goalText
      ? `<span class="week-goal-text">${escHtml(goalText)}</span>`
      : `<span class="week-goal-empty">목표 없음</span>`;
    grid.appendChild(goalCell);

    week.forEach(dayNum => {
      if (dayNum === null) {
        const blank = document.createElement('div');
        blank.className = 'cal-cell empty';
        grid.appendChild(blank);
        return;
      }
      const dateStr = `${currentYear}-${String(currentMonth+1).padStart(2,'0')}-${String(dayNum).padStart(2,'0')}`;
      const cell = document.createElement('div');
      cell.className = 'cal-cell';

      if (currentYear === today.getFullYear() && currentMonth === today.getMonth() && dayNum === today.getDate())
        cell.classList.add('today');

      const noteInfo = NOTES_META[dateStr];
      const theme = noteInfo ? (noteInfo.theme || 'misc') : null;

      if (noteInfo) {
        cell.classList.add('has-note', `theme-${theme}`);
        cell.onclick = () => openNote(dateStr);
      }

      const numEl = document.createElement('div');
      numEl.className = 'day-num';
      numEl.textContent = dayNum;
      cell.appendChild(numEl);

      if (noteInfo) {
        const kwList = document.createElement('div');
        kwList.className = 'kw-list';
        (noteInfo.topics || []).slice(0, 3).forEach(kw => {
          const tag = document.createElement('span');
          tag.className = 'kw-tag';
          tag.textContent = kw;
          kwList.appendChild(tag);
        });
        cell.appendChild(kwList);
      }

      grid.appendChild(cell);
    });
  }
}

function openNote(dateStr) {
  document.getElementById('modal-backdrop').classList.add('open');
  const note = NOTES_DATA[dateStr];
  if (note) {
    renderModalNote(note);
  } else {
    document.getElementById('modal-content').innerHTML = '<div class="loading">노트를 불러올 수 없습니다.</div>';
  }
}

function renderModalNote(note) {
  const theme = note.theme || 'misc';
  const tc = THEME_COLORS[theme] || THEME_COLORS.misc;

  const [y, m, d] = (note.date || '').split('-').map(Number);
  const dayNames = ['일','월','화','수','목','금','토'];
  const dateLabel = d ? `${y}년 ${m}월 ${d}일 (${dayNames[new Date(y, m-1, d).getDay()]})` : escHtml(note.date || '');

  const topics = (note.topics || []).map(t =>
    `<span class="topic-tag" style="background:${tc.bg};color:${tc.text}">${escHtml(t)}</span>`
  ).join('');

  const lines3 = (note.summary_3lines && note.summary_3lines.length)
    ? note.summary_3lines
    : (note.summary ? [note.summary] : []);
  const summaryHtml = lines3.map((line, i) =>
    `<div class="summary-line"><span class="summary-num" style="background:${tc.accent}">${i+1}</span>${escHtml(line)}</div>`
  ).join('');

  const qaHtml = (note.qa_highlights || []).map((qa, i) => `
    <div class="qa-item">
      <div class="qa-q">Q. ${escHtml(qa.q_original || '')}</div>
      <div class="qa-a-summary">A. ${escHtml(qa.a_summary || '')}</div>
      ${qa.a_original ? `<span class="qa-toggle" onclick="toggleOrig(${i})">▼ 원문 보기</span><div class="qa-original" id="qa-orig-${i}">${escHtml(qa.a_original)}</div>` : ''}
    </div>`).join('');

  const codes = (note.code_highlights || []).map(c =>
    `<div class="highlight-item code-item">🔧 ${escHtml(c)}</div>`).join('');

  const insights = (note.insights || []).map(ins =>
    `<div class="highlight-item insight-item">💡 ${escHtml(ins)}</div>`).join('');

  document.getElementById('modal-content').innerHTML = `
    <div class="modal-header" style="background:linear-gradient(140deg,${tc.bg} 0%,#fff 65%);border-left:4px solid ${tc.accent}">
      <span class="modal-theme-badge" style="background:${tc.accent}">${escHtml(tc.label)}</span>
      <div class="modal-date">${dateLabel}</div>
      <div class="modal-time">⏰ ${escHtml(note.start_time||'')} ~ ${escHtml(note.end_time||'')}</div>
      ${note.daily_goal ? `<div class="modal-goal">🎯 ${escHtml(note.daily_goal)}</div>` : ''}
      <div class="topic-tags">${topics}</div>
    </div>
    <div class="modal-divider"></div>
    <div class="modal-body">
      ${summaryHtml ? `<div class="section-title">📝 오늘의 요약</div><div class="summary-3lines">${summaryHtml}</div>` : ''}
      ${qaHtml ? `<div class="section-title">💬 주요 Q&A</div>${qaHtml}` : ''}
      ${codes ? `<div class="section-title">🔧 코드 / 구현</div>${codes}` : ''}
      ${insights ? `<div class="section-title">💡 핵심 인사이트</div>${insights}` : ''}
    </div>`;
}

function closeModal(e) { if (e.target === document.getElementById('modal-backdrop')) closeModalDirect(); }
function closeModalDirect() { document.getElementById('modal-backdrop').classList.remove('open'); }
function toggleOrig(i) {
  const el = document.getElementById(`qa-orig-${i}`);
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

document.getElementById('prev-month').onclick = () => {
  if (--currentMonth < 0) { currentMonth = 11; currentYear--; }
  renderCalendar();
};
document.getElementById('next-month').onclick = () => {
  if (++currentMonth > 11) { currentMonth = 0; currentYear++; }
  renderCalendar();
};
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModalDirect(); });

init();

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('sw.js'));
}
</script>
</body>
</html>'''


def rebuild_index_html(output_dir: Path) -> None:
    data_dir = output_dir / "data"
    notes_meta: dict[str, dict] = {}
    notes_data: dict[str, dict] = {}
    for p in sorted(data_dir.glob("*.json")):
        if p.name == "weekly_goals.json" or len(p.stem) != 10:
            continue
        try:
            note = json.loads(p.read_text(encoding='utf-8'))
            notes_meta[p.stem] = {
                "topics": note.get('topics', []),
                "theme": note.get('theme', 'misc'),
            }
            notes_data[p.stem] = note
        except (json.JSONDecodeError, OSError):
            notes_meta[p.stem] = {"topics": [], "theme": "misc"}

    weekly_goals: dict[str, str] = {}
    if WEEKLY_GOALS_FILE.exists():
        try:
            weekly_goals = json.loads(WEEKLY_GOALS_FILE.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pass

    def _escape(obj) -> str:
        return (
            json.dumps(obj, ensure_ascii=False)
            .replace('</', '<\\/')
            .replace('<!--', '<\\!--')
        )

    html = (HTML_TEMPLATE
        .replace('__NOTES_META__', _escape(notes_meta))
        .replace('__WEEKLY_GOALS__', _escape(weekly_goals))
        .replace('__THEME_COLORS__', _escape(THEME_COLORS))
        .replace('__NOTES_DATA__', _escape(notes_data))
    )
    (output_dir / "index.html").write_text(html, encoding='utf-8')


def main() -> None:
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

    daily_goal = state.get('daily_goal', '')
    end_time = datetime.now(tz=timezone.utc)
    start_str = start_time.astimezone().strftime('%Y%m%d %H:%M')
    end_str = end_time.astimezone().strftime('%Y%m%d %H:%M')

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
        note_data = generate_note_data(conversation, date_str, start_str, end_str, daily_goal)
    except Exception as e:
        print(f"[research_note] API call failed: {e}", file=sys.stderr)
        sys.exit(1)

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    write_note_json(note_data, NOTES_DIR)
    rebuild_index_html(NOTES_DIR)
    print(f"[research_note] Done! Chaeyeon's_Research_Note.html updated.")


if __name__ == '__main__':
    main()
