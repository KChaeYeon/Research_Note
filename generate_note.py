import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load ANTHROPIC_API_KEY from ~/.research_notes_env if not already set.
# (Kept out of ~/.bashrc to avoid conflicting with Claude Code OAuth.)
if not os.environ.get("ANTHROPIC_API_KEY"):
    _env_file = Path.home() / ".research_notes_env"
    if _env_file.is_file():
        for _line in _env_file.read_text().splitlines():
            _m = re.match(r'\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)', _line)
            if _m:
                os.environ.setdefault(_m.group(1), _m.group(2).strip().strip('"').strip("'"))

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
<title>채연이의 도비 일지 🧟 — 탈 출 기 원 !</title>
<link rel="manifest" href="manifest.json">
<meta name="theme-color" content="#0a0a14">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="도비일지">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
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
@media (max-width: 700px) {
  .page { padding: 16px 10px; }
  .page-title { font-size: 1.2rem; }
  .page-subtitle { font-size: 0.82rem; }
  .theme-legend { gap: 10px; }
  .legend-item { font-size: 0.78rem; }
}
.kw-tag { font-size: 0.78rem; border-radius: 8px; padding: 3px 7px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.topic-tags { margin-top: 13px; display: flex; flex-wrap: wrap; gap: 6px; }
.topic-tag { display: inline-flex; align-items: center; border-radius: 12px; padding: 5px 14px; font-size: 0.9rem; font-weight: 700; }
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
.theme-tabs { display: flex; gap: 0; border-bottom: 2px solid #eee; background: #fafafa; }
.theme-tab { flex: 1; padding: 11px 4px; font-size: 0.82rem; font-weight: 700; text-align: center; cursor: pointer; border: none; background: none; color: #aaa; border-bottom: 3px solid transparent; margin-bottom: -2px; transition: color 0.15s, border-color 0.15s; letter-spacing: -0.01em; white-space: nowrap; }
.theme-tab.active { color: var(--tab-color); border-bottom-color: var(--tab-color); }
.theme-panel { display: none; padding: 18px 28px 28px; }
.theme-panel.active { display: block; }
.theme-panel-empty { text-align: center; color: #ccc; padding: 40px 0; font-size: 0.95rem; }
.advisor-session { margin-top: 4px; }
.advisor-meta { font-size: 0.82rem; color: #888; margin-bottom: 10px; font-style: italic; }
.advisor-block { border: 1px solid #c5cae9; border-radius: 9px; margin-bottom: 6px; overflow: hidden; }
.advisor-btn { width: 100%; text-align: left; background: #e8eaf6; border: none; padding: 11px 14px; font-size: 0.9rem; font-weight: 700; color: #283593; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: background 0.15s; }
.advisor-btn:hover { background: #d9ddf6; }
.advisor-arrow { font-size: 0.72rem; transition: transform 0.2s; display: inline-block; }
.advisor-btn.open .advisor-arrow { transform: rotate(90deg); }
.advisor-body { display: none; padding: 13px 16px; font-size: 0.88rem; line-height: 1.8; color: #444; background: #fafbff; border-top: 1px solid #e8ecf0; white-space: pre-wrap; }
/* ═══ NEOBRUTALISM ADDITIONS ═══ */
:root{--bg-color:#fcfbf9;--card-bg:#ffffff;--border-color:#222222;--accent-color:#a8e6cf;--text-main:#333333;--text-muted:#777777;}
.journey-container{background:#f7f6f2;border:2px solid var(--border-color);border-radius:12px;padding:14px 18px;margin-bottom:20px;}
.journey-title{font-size:1rem;font-weight:700;color:var(--text-main);display:flex;justify-content:space-between;margin-bottom:8px;}
.journey-track{height:12px;background:#e0deda;border:2px solid var(--border-color);border-radius:3px;position:relative;margin:18px 0 6px 0;}
.journey-progress{height:100%;background:var(--accent-color);border-radius:2px;transition:width 0.5s;}
.journey-character{position:absolute;top:-22px;transform:translateX(-50%);font-size:24px;}
.journey-destination{position:absolute;right:0;top:-22px;font-size:24px;transform:translateX(50%);}
.journey-pct{font-size:0.85rem;color:var(--text-muted);text-align:right;margin-top:4px;}
.main-layout{display:grid;grid-template-columns:280px 1fr;gap:22px;align-items:start;}
.sidebar-left{background:var(--card-bg);border:3px solid var(--border-color);box-shadow:4px 4px 0 var(--border-color);padding:22px;display:flex;flex-direction:column;gap:14px;}
.sidebar-section-title{font-size:1.1rem;font-weight:800;border-bottom:2px solid var(--border-color);padding-bottom:8px;margin-bottom:4px;}
.goal-list{list-style:none;display:flex;flex-direction:column;gap:12px;}
.goal-item{display:flex;align-items:flex-start;gap:10px;font-size:0.97rem;line-height:1.5;}
.goal-item input[type="checkbox"]{margin-top:3px;width:16px;height:16px;accent-color:#2d6a4f;cursor:pointer;flex-shrink:0;}
.goal-item label{cursor:pointer;color:var(--text-main);word-break:keep-all;overflow-wrap:anywhere;}
.calendar-section{background:var(--card-bg);border:3px solid var(--border-color);box-shadow:4px 4px 0 var(--border-color);padding:22px;}
.calendar-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;}
.calendar-nav-btn{background:var(--card-bg);border:2px solid var(--border-color);padding:6px 14px;cursor:pointer;font-size:0.97rem;box-shadow:2px 2px 0 var(--border-color);font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;}
.calendar-nav-btn:active{box-shadow:none;transform:translate(2px,2px);}
.calendar-month-label{font-size:1.2rem;font-weight:800;color:var(--border-color);}
.calendar-grid-7{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;}
.cal-weekday{text-align:center;font-size:1rem;font-weight:800;padding:8px 0;border-bottom:2px solid var(--border-color);}
.cal-weekday.sun{color:#e57373;}
.cal-weekday.sat{color:#1e88e5;}
.cal-day{min-height:100px;border:1.5px solid #ddd;padding:7px;background:#fff;display:flex;flex-direction:column;justify-content:space-between;transition:all 0.15s;cursor:default;}
.cal-day.has-note{cursor:pointer;border-color:#bbb;}
.cal-day.has-note:hover{background:#fdfaf3;border-color:var(--border-color);box-shadow:2px 2px 0 rgba(0,0,0,0.07);}
.cal-day.today{border:2.5px solid #1565c0;background:#f0f5ff;}
.cal-day.today .cal-day-num{color:#1565c0;font-weight:800;}
.cal-day.empty{background:transparent;border-color:transparent;}
.cal-day-num{font-size:1rem;font-weight:700;color:#666;}
.cal-day-zombie{display:block;text-align:center;font-size:1.3rem;margin-top:3px;}
.day-summary-dot{font-size:0.78rem;background:#eef7f4;border:1px solid var(--border-color);padding:4px 5px;border-radius:3px;margin-top:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.cal-day.theme-rsp .day-summary-dot{background:#fce4ec;border-color:#f48fb1;}
.cal-day.theme-sleep .day-summary-dot{background:#f3e5f5;border-color:#ce93d8;}
.cal-day.theme-ultrasound .day-summary-dot{background:#f1f8e9;border-color:#aed581;}
.cal-day.theme-misc .day-summary-dot{background:#f5f5f5;border-color:#bdbdbd;}
.detail-panel{position:fixed;top:0;right:-480px;width:450px;height:100vh;background:var(--card-bg);border-left:4px solid var(--border-color);box-shadow:-8px 0 0 rgba(0,0,0,0.04);padding:28px 24px;transition:right 0.3s cubic-bezier(0.25,1,0.5,1);z-index:1000;display:flex;flex-direction:column;gap:18px;overflow-y:auto;font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;}
.detail-panel.open{right:0;}
.panel-close-btn{align-self:flex-end;background:none;border:2px solid var(--border-color);padding:5px 12px;cursor:pointer;font-size:0.95rem;font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;}
.panel-close-btn:hover{background:#f0f0f0;}
.panel-date{font-size:1.4rem;font-weight:800;border-bottom:2px solid var(--border-color);padding-bottom:8px;}
.panel-section{background:#faf9f6;border:2px solid var(--border-color);padding:15px;box-shadow:3px 3px 0 #e0deda;}
.panel-section-title{font-size:1rem;font-weight:800;margin-bottom:10px;color:var(--border-color);}
.keyword-tag{display:inline-block;background:#fff;border:1.5px solid var(--border-color);padding:4px 10px;font-size:0.88rem;margin:3px 4px 3px 0;border-radius:3px;}
.panel-summary-text{font-size:0.97rem;line-height:1.7;color:#444;}
.panel-tabs-header{display:flex;gap:4px;border-bottom:2px solid var(--border-color);overflow-x:auto;}
.panel-tab-btn{background:#f0f0f0;border:2px solid var(--border-color);border-bottom:none;padding:8px 13px;cursor:pointer;font-size:0.92rem;border-radius:5px 5px 0 0;white-space:nowrap;font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;}
.panel-tab-btn.active{background:#fff;font-weight:800;border-bottom:2px solid #fff;margin-bottom:-2px;position:relative;z-index:1;}
.panel-tab-content{display:none;padding:14px;border:2px solid var(--border-color);background:#fff;border-radius:0 5px 5px 5px;min-height:120px;}
.panel-tab-content.active{display:block;}
.panel-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.15);z-index:999;}
.panel-overlay.open{display:block;}
@media(max-width:900px){.main-layout{grid-template-columns:1fr;}.sidebar-left{box-shadow:2px 2px 0 var(--border-color);}.detail-panel{width:100%;right:-100%;}.cal-day{min-height:64px;}}
/* ═══ WHITE THEME ═══ */
body{font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;background:#fcfbf9;color:#333;min-height:100vh;}
.page-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;padding:22px 28px 20px;background:linear-gradient(135deg,#12122a 0%,#1a1a3e 60%,#0d1b2a 100%);border-radius:20px;position:relative;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.06);}
.page-header::after{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 85% 50%,rgba(100,150,255,.09) 0%,transparent 60%);pointer-events:none;}
.page-title{font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;font-size:1.9rem;font-weight:800;color:#fff;text-shadow:0 2px 24px rgba(120,180,255,.22);letter-spacing:-.01em;line-height:1.3;}
.page-subtitle{font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;font-size:1rem;color:rgba(255,255,255,.52);opacity:1;margin-top:5px;letter-spacing:0;animation:none;}
.header-badges{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px;position:relative;z-index:1;}
.header-badge{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.13);border-radius:20px;padding:4px 12px;font-size:.78rem;color:rgba(255,255,255,.72);display:inline-flex;align-items:center;gap:4px;white-space:nowrap;transition:background .15s;}
.header-badge:hover{background:rgba(255,255,255,.16);color:rgba(255,255,255,.92);}
.month-label{font-family:'Malgun Gothic','맑은 고딕',system-ui,sans-serif;font-size:1.2rem;font-weight:700;color:rgba(255,255,255,.9);min-width:110px;text-align:center;}
.cal-nav{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:8px;cursor:pointer;padding:5px 16px;font-size:1rem;color:rgba(255,255,255,.8);transition:all .15s;}
.cal-nav:hover{background:rgba(255,255,255,.2);border-color:rgba(255,255,255,.35);color:#fff;}























.theme-legend{display:flex;align-items:center;gap:16px;margin-bottom:14px;flex-wrap:wrap;padding:0;background:transparent;border:none;}
.legend-label{font-size:.85rem;font-weight:700;color:#999;letter-spacing:.06em;text-transform:uppercase;margin-right:4px;}
.legend-item{display:flex;align-items:center;gap:5px;font-size:.88rem;color:#666;}
.legend-dot{width:10px;height:10px;border-radius:50%;}

.today-zombie{display:block;text-align:center;font-size:1.2rem;line-height:1;margin-top:4px;width:100%;}

@media(max-width:700px){.page{padding:10px 6px;}.page-header{padding:14px 16px 14px;border-radius:14px;}.page-title{font-size:1.2rem;}.page-subtitle{font-size:.8rem;}.header-badge{font-size:.66rem;padding:3px 8px;}.cal-nav-row{flex-direction:column;align-items:center;gap:4px;}.month-label{font-size:1rem;}.today-zombie{font-size:1.8rem;margin-top:2px;}}
</style>
</head>
<body>
<div class="page">
  <div class="page-header">
    <div>
      <div class="page-title">🧟 채연이의 도비 일지</div>
      <div class="page-subtitle">탈출기원 연구 일지</div>
      <div class="header-badges">
        <span class="header-badge" id="badge-survival">🎓 5년차 생존자</span>
        <span class="header-badge">🧠 glymphatic 연구 중</span>
        <span class="header-badge">☕ 카페인으로 버팀</span>
        <span class="header-badge">✍️ 논문 쓰는 좀비</span>
      </div>
    </div>
    <div class="cal-nav-row">
      <button class="cal-nav" id="prev-month">&#8592;</button>
      <div class="month-label" id="month-label"></div>
      <button class="cal-nav" id="next-month">&#8594;</button>
    </div>
  </div>
  <!-- Journey Progress Bar -->
  <div class="journey-container">
    <div class="journey-title">
      <span>🧠 현재 단계: 수면·호흡 연구 논문 작성 중</span>
      <span id="journey-pct-label"></span>
    </div>
    <div class="journey-track">
      <div class="journey-progress" id="journey-progress"></div>
      <div class="journey-character" id="journey-char">🏃‍♀️</div>
      <div class="journey-destination">🎓</div>
    </div>
    <div class="journey-pct" id="journey-days"></div>
  </div>

  <!-- Main 2-column layout -->
  <div class="main-layout">

    <!-- Sidebar: Weekly Goals -->
    <aside class="sidebar-left">
      <div class="sidebar-section-title">🎯 이번 주 목표</div>
      <ul class="goal-list" id="weekly-goal-list">
        <li class="goal-item" style="color:#aaa;font-size:0.9rem;">목표를 불러오는 중...</li>
      </ul>
    </aside>

    <!-- Calendar -->
    <main class="calendar-section">
      <div class="calendar-header">
        <button class="calendar-nav-btn" id="cal-prev">◀ 이전 달</button>
        <div class="calendar-month-label" id="cal-month-label"></div>
        <button class="calendar-nav-btn" id="cal-next">다음 달 ▶</button>
      </div>
      <div class="calendar-grid-7" id="cal-grid-7"></div>
    </main>

  </div>
</div>

<!-- Right slide panel overlay -->
<div class="panel-overlay" id="panel-overlay" onclick="closePanel()"></div>

<!-- Right slide panel -->
<div class="detail-panel" id="detail-panel">
  <button class="panel-close-btn" onclick="closePanel()">닫기 ✕</button>
  <div class="panel-date" id="panel-date"></div>
  <div class="panel-section">
    <div class="panel-section-title">🔑 핵심 키워드</div>
    <div id="panel-keywords"></div>
  </div>
  <div class="panel-section">
    <div class="panel-section-title">📌 오늘의 요약</div>
    <div id="panel-summary"></div>
  </div>
  <div class="panel-section" style="flex:1">
    <div class="panel-section-title">🔬 상세 기록</div>
    <div id="panel-tabs-header" class="panel-tabs-header"></div>
    <div id="panel-tabs-content"></div>
  </div>
</div>

<script>
const NOTES_META = __NOTES_META__;
const WEEKLY_GOALS = __WEEKLY_GOALS__;
const THEME_COLORS = __THEME_COLORS__;
const NOTES_DATA = __NOTES_DATA__;

let currentYear, currentMonth;

/* ─ Helpers ─ */
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function getISOWeekKey(year, month, day) {
  const d = new Date(Date.UTC(year, month, day));
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
  const ys = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const w = Math.ceil((((d - ys) / 86400000) + 1) / 7);
  return `${d.getUTCFullYear()}-W${String(w).padStart(2,'0')}`;
}

/* ─ Journey Bar ─ */
function renderJourney() {
  const start = new Date('2022-03-02');
  const end   = new Date('2027-08-31'); // estimated graduation
  const now   = new Date();
  const total = end - start;
  const elapsed = now - start;
  const pct = Math.min(100, Math.max(0, Math.round(elapsed / total * 100)));
  const days = Math.floor(elapsed / 86400000);

  const prog = document.getElementById('journey-progress');
  const char = document.getElementById('journey-char');
  const pctLabel = document.getElementById('journey-pct-label');
  const daysEl = document.getElementById('journey-days');

  if (prog) prog.style.width = pct + '%';
  if (char) { char.style.left = pct + '%'; char.textContent = '🏃‍♀️'; }
  if (pctLabel) pctLabel.textContent = `졸업까지 ${pct}% 완료!`;
  if (daysEl) daysEl.textContent = `입학 ${days}일째 생존 중 🧟`;
}

/* ─ Sidebar: Weekly Goals ─ */
function renderSidebar() {
  const today = new Date();
  const weekKey = getISOWeekKey(today.getFullYear(), today.getMonth(), today.getDate());
  const goalText = WEEKLY_GOALS[weekKey] || '';
  const list = document.getElementById('weekly-goal-list');
  if (!list) return;

  if (!goalText) {
    list.innerHTML = '<li class="goal-item" style="color:#aaa;font-size:0.9rem;">이번 주 목표 없음</li>';
    return;
  }

  const goals = goalText.split('\\n').filter(g => g.trim());
  list.innerHTML = goals.map((g, i) => `
    <li class="goal-item">
      <input type="checkbox" id="sg${i}">
      <label for="sg${i}">${escHtml(g.trim())}</label>
    </li>`).join('');
}

/* ─ Calendar ─ */
function renderCalendar() {
  const label = document.getElementById('cal-month-label');
  if (label) label.textContent = `${currentYear}년 ${currentMonth + 1}월`;

  const grid = document.getElementById('cal-grid-7');
  if (!grid) return;
  grid.innerHTML = '';

  const THEME_EMOJI = { rsp:'🫁', sleep:'💤', ultrasound:'🔊', misc:'📋' };

  // Weekday headers
  ['일','월','화','수','목','금','토'].forEach((d, i) => {
    const h = document.createElement('div');
    h.className = 'cal-weekday' + (i===0?' sun':i===6?' sat':'');
    h.textContent = d;
    grid.appendChild(h);
  });

  const firstDay = new Date(currentYear, currentMonth, 1).getDay();
  const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
  const today = new Date();

  // Empty cells before first day
  for (let i = 0; i < firstDay; i++) {
    const blank = document.createElement('div');
    blank.className = 'cal-day empty';
    grid.appendChild(blank);
  }

  // Day cells
  for (let dayNum = 1; dayNum <= daysInMonth; dayNum++) {
    const dateStr = `${currentYear}-${String(currentMonth+1).padStart(2,'0')}-${String(dayNum).padStart(2,'0')}`;
    const cell = document.createElement('div');
    cell.className = 'cal-day';

    const isToday = (currentYear === today.getFullYear() && currentMonth === today.getMonth() && dayNum === today.getDate());
    if (isToday) cell.classList.add('today');

    const noteInfo = NOTES_META[dateStr];
    if (noteInfo) {
      const theme = noteInfo.theme || 'misc';
      cell.classList.add('has-note', `theme-${theme}`);
      cell.addEventListener('click', () => openPanel(dateStr));
    }

    // Day number
    const numEl = document.createElement('div');
    numEl.className = 'cal-day-num';
    numEl.textContent = dayNum;
    cell.appendChild(numEl);

    // Today zombie
    if (isToday) {
      const zEl = document.createElement('div');
      zEl.className = 'cal-day-zombie';
      zEl.textContent = '🧟';
      cell.appendChild(zEl);
    }

    // Summary dot for notes
    if (noteInfo && noteInfo.topics && noteInfo.topics.length > 0) {
      const dot = document.createElement('div');
      dot.className = 'day-summary-dot';
      const theme = noteInfo.theme || 'misc';
      const emoji = THEME_EMOJI[theme] || '📋';
      dot.textContent = emoji + ' ' + (noteInfo.topics[0] || '').substring(0, 14);
      cell.appendChild(dot);
    }

    grid.appendChild(cell);
  }

  // Trailing empty cells to complete last row
  const totalCells = firstDay + daysInMonth;
  const remainder = totalCells % 7;
  if (remainder > 0) {
    for (let i = 0; i < 7 - remainder; i++) {
      const blank = document.createElement('div');
      blank.className = 'cal-day empty';
      grid.appendChild(blank);
    }
  }
}

/* ─ Slide Panel ─ */
function openPanel(dateStr) {
  const note = NOTES_DATA[dateStr];
  if (!note) return;

  const [y, m, d] = dateStr.split('-').map(Number);
  const dayNames = ['일','월','화','수','목','금','토'];
  const dow = dayNames[new Date(y, m-1, d).getDay()];
  document.getElementById('panel-date').textContent = `${y}년 ${m}월 ${d}일 (${dow})`;

  // Keywords (topics)
  const kwContainer = document.getElementById('panel-keywords');
  kwContainer.innerHTML = (note.topics || []).map(t =>
    `<span class="keyword-tag">${escHtml(t)}</span>`).join('') || '<span style="color:#aaa">키워드 없음</span>';

  // Summary
  const summaryContainer = document.getElementById('panel-summary');
  const lines = note.summary_3lines || [];
  const nonEmpty = lines.filter(l => l && l.trim());
  summaryContainer.innerHTML = nonEmpty.map(l =>
    `<p class="panel-summary-text" style="margin-bottom:6px">• ${escHtml(l)}</p>`).join('') ||
    `<p class="panel-summary-text" style="color:#aaa">요약 없음</p>`;

  // Tabs: Q&A / Code / Insights
  const tabsHeaderEl = document.getElementById('panel-tabs-header');
  const tabsContentEl = document.getElementById('panel-tabs-content');

  const tabs = [
    { id:'qa', label:'💬 Q&A', content: buildQAHtml(note) },
    { id:'code', label:'🔧 코드', content: buildListHtml(note.code_highlights || [], 'code-item') },
    { id:'insight', label:'💡 인사이트', content: buildListHtml(note.insights || [], 'insight-item') },
  ];

  tabsHeaderEl.innerHTML = tabs.map((t, i) =>
    `<button class="panel-tab-btn${i===0?' active':''}" data-tab="${t.id}" onclick="switchPanelTab('${t.id}')">${t.label}</button>`).join('');

  tabsContentEl.innerHTML = tabs.map((t, i) =>
    `<div class="panel-tab-content${i===0?' active':''}" id="ptab-${t.id}">${t.content}</div>`).join('');

  document.getElementById('detail-panel').classList.add('open');
  document.getElementById('panel-overlay').classList.add('open');
}

function buildQAHtml(note) {
  const qas = note.qa_highlights || [];
  if (!qas.length) return '<p style="color:#aaa;padding:10px">Q&A 기록 없음</p>';
  return qas.map((qa, i) => {
    if (typeof qa === 'string') {
      return `<div class="qa-item"><div class="qa-a-summary">${escHtml(qa)}</div></div>`;
    }
    return `<div class="qa-item">
      <div class="qa-q">Q. ${escHtml(qa.q_original || '')}</div>
      <div class="qa-a-summary">A. ${escHtml(qa.a_summary || '')}</div>
      ${qa.a_original ? `<span class="qa-toggle" onclick="toggleOrig(${i})">▼ 원문</span><div class="qa-original" id="qa-orig-${i}">${escHtml(qa.a_original)}</div>` : ''}
    </div>`;
  }).join('');
}

function buildListHtml(items, cls) {
  if (!items.length) return '<p style="color:#aaa;padding:10px">기록 없음</p>';
  return items.map(item => `<div class="highlight-item ${cls}">${escHtml(item)}</div>`).join('');
}

function closePanel() {
  document.getElementById('detail-panel').classList.remove('open');
  document.getElementById('panel-overlay').classList.remove('open');
}

function switchPanelTab(tabId) {
  document.querySelectorAll('.panel-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.panel-tab-content').forEach(c => c.classList.remove('active'));
  const btn = document.querySelector(`.panel-tab-btn[data-tab="${tabId}"]`);
  if (btn) btn.classList.add('active');
  const content = document.getElementById('ptab-' + tabId);
  if (content) content.classList.add('active');
}

function toggleOrig(i) {
  const el = document.getElementById(`qa-orig-${i}`);
  if (el) el.style.display = el.style.display === 'block' ? 'none' : 'block';
}

function toggleAdvisor(idx) {
  const body = document.getElementById('adv-body-' + idx);
  const btn = document.getElementById('adv-btn-' + idx);
  if (!body || !btn) return;
  const isOpen = body.style.display === 'block';
  body.style.display = isOpen ? 'none' : 'block';
  btn.classList.toggle('open', !isOpen);
}

/* ─ Init ─ */
function init() {
  const t = new Date();
  currentYear = t.getFullYear();
  currentMonth = t.getMonth();
  renderJourney();
  renderSidebar();
  renderCalendar();
}

document.getElementById('cal-prev').addEventListener('click', () => {
  if (--currentMonth < 0) { currentMonth = 11; currentYear--; }
  renderCalendar();
});
document.getElementById('cal-next').addEventListener('click', () => {
  if (++currentMonth > 11) { currentMonth = 0; currentYear++; }
  renderCalendar();
});

document.addEventListener('keydown', e => { if (e.key === 'Escape') closePanel(); });

// Survival counter in header badge
(function(){
  var start = new Date('2022-03-02');
  var now = new Date();
  var days = Math.floor((now - start) / 86400000);
  var years = Math.max(1, Math.floor(days / 365) + 1);
  var el = document.getElementById('badge-survival');
  if(el) el.textContent = '🎓 ' + years + '년차 생존자 +' + days + '일';
})();

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('sw.js'));
}

init();
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
