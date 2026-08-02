"""
Pi Scanner Dashboard - web interface served from the Pi (SQLite version).

Access from any device on the network: http://pi3:8080
Shows all transmissions with search by time, name, text, frequency.
Also shows live scanner status (current channel, queue depth).

Uses scanner_db (SQLite) instead of JSONL for all data access.

Run:  python3 dashboard_updated.py
Or:   systemd service (pi-dashboard.service)
"""

import os
import json
import re
import urllib.parse
from pathlib import Path

import requests
from flask import Flask, render_template_string, request, jsonify, send_file

import scanner_db

app = Flask(__name__)

# Text decoders for inline decoding when GPU posts results
try:
    from codes import decode_for
    from phonetic import decode_plates
    from phone import detect_phones
    _DECODERS_AVAILABLE = True
except ImportError:
    _DECODERS_AVAILABLE = False


def _run_text_decoders(text, channel_name):
    """Run text-based decoders on a transcript. Returns a dict of findings."""
    if not _DECODERS_AVAILABLE or not text:
        return {}
    results = {}
    try:
        plates = decode_plates(text)
        if plates:
            results["plates"] = [p["plate"] for p in plates]
    except Exception:
        pass
    try:
        phones = detect_phones(text)
        if phones:
            results["phones"] = [p["phone"] for p in phones]
    except Exception:
        pass
    try:
        profile, codes = decode_for(text, channel_name)
        if codes:
            results["codes"] = [{"code": c["code"], "meaning": c["meaning"]} for c in codes]
            if profile:
                results["code_profile"] = profile.name
    except Exception:
        pass
    return results

# Config
STATUS_FILE = Path("/home/pi/scanner/status.json")
GPU_SERVER_URL = os.environ.get("GPU_SERVER_URL", "")


# --- Helpers ----------------------------------------------------------------

def _load_status():
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"state": "unknown", "channel": "", "queue": 0, "screen": []}


def _build_decoded_display(record):
    """Build HTML badges for the decoded column from a record's decoded fields."""
    parts = []
    # Audio-based decoders (stored in 'decoded' field)
    d = record.get("decoded") or {}
    if d.get("dtmf"):
        parts.append(f'<span class="badge b-dtmf" data-dtype="dtmf">DTMF:{d["dtmf"]}</span>')
    if d.get("morse"):
        parts.append(f'<span class="badge b-morse" data-dtype="morse">CW:{d["morse"]}</span>')
    if d.get("fsk"):
        parts.append(f'<span class="badge b-fsk" data-dtype="fsk">{d["fsk"][:30]}</span>')
    if d.get("tones") and isinstance(d["tones"], list):
        shown = ", ".join(f"{f:.0f}Hz/{dur:.1f}s" for f, dur in d["tones"][:3])
        parts.append(f'<span class="badge b-fsk" data-dtype="tones">{shown}</span>')
    # Text-based decoders (stored in 'decoded_text' field)
    dt_field = record.get("decoded_text") or {}
    if dt_field.get("codes"):
        for c in dt_field["codes"][:3]:
            parts.append(f'<span class="badge b-code" data-dtype="codes">{c["code"]}={c["meaning"][:20]}</span>')
    if dt_field.get("plates"):
        for p in dt_field["plates"][:2]:
            parts.append(f'<span class="badge b-plate" data-dtype="plates">{p}</span>')
    if dt_field.get("phones"):
        for p in dt_field["phones"][:2]:
            parts.append(f'<span class="badge b-phone" data-dtype="phones">{p}</span>')
    return " ".join(parts)


def _enrich_results(results, query=""):
    """Add text_hl (highlighted text) and decoded_display to each record."""
    # Assign queue position to Transcribing... items (1 = next to be transcribed)
    queue_pos = 1
    for r in results:
        text = r.get("text", "")
        if text == "Transcribing...":
            r["queue_pos"] = queue_pos
            queue_pos += 1
        if query and text:
            pat = re.compile(re.escape(query), re.IGNORECASE)
            text = pat.sub(lambda m: f'<span class="hl">{m.group()}</span>', text)
        r["text_hl"] = text
        r["decoded_display"] = _build_decoded_display(r)
    return results


# --- HTML Template ----------------------------------------------------------

HTML = """<!DOCTYPE html>
<html>
<head>
<title>Pi Scanner</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#111;color:#ddd;padding:10px}
h1{color:#4fc3f7;font-size:1.4em;margin-bottom:8px}
.stats-bar{background:#1a1a2e;padding:7px 12px;border-radius:6px;margin-bottom:10px;font-size:12px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.stats-bar span{white-space:nowrap}
.stats-bar b{color:#4fc3f7}
.status{background:#1a2744;padding:10px;border-radius:6px;margin-bottom:10px;font-size:13px;display:flex;gap:15px;flex-wrap:wrap;align-items:flex-start}
.scanner-screen{background:#001a00;border:2px solid #444;border-radius:6px;padding:8px 12px;font-family:'Courier New',monospace;font-size:12px;color:#33ff33;width:290px;line-height:1.3}
.screen-title{color:#888;font-size:10px;text-align:center;margin-bottom:4px;font-family:system-ui;border-bottom:1px solid #333;padding-bottom:3px}
.screen-header{color:#ffaa00;font-size:10px;margin-bottom:4px;padding-bottom:3px;border-bottom:1px solid #1a3a1a;height:2.6em;overflow:hidden;line-height:1.3}
.screen-section{padding:3px 0;border-bottom:1px solid #1a3a1a;height:3.9em;overflow:hidden;line-height:1.3}
.screen-section:last-child{border-bottom:none}
.screen-footer{color:#888;font-size:10px;margin-top:3px;padding-top:3px;border-top:1px solid #1a3a1a;height:1.3em;overflow:hidden}
.screen-line{white-space:pre;overflow:hidden;text-overflow:ellipsis}
.controls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center}
.controls input,.controls select{padding:5px 8px;border:1px solid #333;border-radius:4px;background:#1a1a2e;color:#ddd;font-size:12px}
.controls select{max-width:160px}
.controls input[type=text]{min-width:100px;flex:1}
.controls button{padding:5px 12px;background:#4fc3f7;color:#111;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:12px}
.controls label{font-size:12px;display:flex;align-items:center;gap:4px;cursor:pointer;color:#aaa}
.controls label input[type=checkbox]{accent-color:#4fc3f7}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;font-size:12px;width:100%;table-layout:auto}
th{background:#1a2744;padding:6px 8px;text-align:left;position:sticky;top:0;white-space:nowrap;user-select:none;cursor:pointer;position:relative;overflow:hidden;z-index:2}
th:hover{background:#223a5e}
th .sort-arrow{font-size:9px;margin-left:3px;opacity:0.4}
th.sort-asc .sort-arrow::after{content:'\\25B2'}
th.sort-desc .sort-arrow::after{content:'\\25BC'}
th .resize-handle{position:absolute;right:0;top:0;bottom:0;width:5px;cursor:col-resize;background:transparent;z-index:3}
th .resize-handle:hover,th .resize-handle.active{background:#4fc3f7}
td{padding:5px 8px;border-bottom:1px solid #222;vertical-align:top;white-space:nowrap}
tr:hover{background:#1a2744}
.col-time{color:#aaa;max-width:80px;flex-shrink:0}
.col-dur{color:#aaa;text-align:right;max-width:50px;flex-shrink:0}
.col-sys{color:#ce93d8;max-width:100px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis}
.col-grp{color:#a5d6a7;max-width:100px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis}
.col-ch{color:#81d4fa;max-width:120px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis}
.col-freq{color:#888;max-width:60px;flex-shrink:0;text-align:right}
.col-text{white-space:normal;word-break:break-word;flex:1;min-width:300px}
.col-decoded{font-size:11px;white-space:nowrap;color:#a5d6a7;max-width:150px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis}
.col-decoded .badge{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;margin:1px}
.col-decoded .b-code{background:#1565c0}.col-decoded .b-plate{background:#2e7d32}
.col-decoded .b-phone{background:#6a1b9a}.col-decoded .b-dtmf{background:#ef6c00}
.col-decoded .b-morse{background:#7e57c2}.col-decoded .b-fsk{background:#00695c}
.decoded-filters{display:flex;gap:6px;flex-wrap:wrap;align-items:center;padding:5px 0;margin-bottom:6px}
.decoded-filters span.df-label{font-size:10px;color:#888;margin-right:2px;white-space:nowrap}
.decoded-filters .df-btn{padding:2px 7px;background:transparent;border:1px solid #555;color:#888;border-radius:3px;font-size:10px;cursor:pointer;transition:all .15s}
.decoded-filters .df-btn:hover{border-color:#4fc3f7;color:#4fc3f7}
.decoded-filters .df-btn.active{background:#4fc3f7;color:#111;border-color:#4fc3f7}
.play-btn{background:#2e7d32;color:#fff;border:none;border-radius:3px;padding:2px 5px;margin-right:4px;cursor:pointer;font-size:10px;display:inline-flex;align-items:center;gap:2px;vertical-align:middle}
.play-btn:hover{background:#43a047}
.play-btn.playing{background:#c62828}
.play-btn.playing:hover{background:#e53935}
.hl{background:#ffeb3b33;color:#ffeb3b;padding:1px 2px;border-radius:2px}
.empty{text-align:center;padding:30px;color:#666}
.blank{color:#555;font-style:italic}
.transcribing{color:#ffb74d;font-style:italic}
.pagination-nav{display:flex;gap:8px;align-items:center;font-size:13px;margin-left:10px}
.pagination-nav a.prev-btn,.pagination-nav a.next-btn{color:#4fc3f7;text-decoration:none;padding:2px 8px;border:1px solid #4fc3f7;border-radius:4px;font-size:11px;transition:all 0.2s}
.pagination-nav a.prev-btn:hover,.pagination-nav a.next-btn:hover{background:#4fc3f7;color:#111}
.pagination-nav .page-indicator{color:#888;font-size:12px}
@media(max-width:900px){
.col-sys,.col-grp{display:none}
.controls{flex-direction:column}
table{font-size:11px}
.pagination-nav{display:none}
}
</style>
</head>
<body>
<h1>&#x1F4E1; Pi Scanner</h1>
<div class="stats-bar">
<span>CPU <b id="sys-cpu">--</b>%</span>
<span>RAM <b id="sys-ram">--</b>%</span>
<span><b id="sys-temp">--</b>&deg;C</span>
<span>Queue: <b id="status-queue">{{ pending }}</b></span>
<span>Records: <b>{{ total }}</b></span>
{% if total > end_rec - start_rec + 1 or total_pages > 1 %}<span>Showing {{ start_rec }}–{{ end_rec }}</span>{% endif %}
{% if total_pages > 1 %}
<span class="pagination-nav">
{% if page > 1 %}<a href="?q={{ query }}{{ '&ch=' + channel if channel }}{{ '&sys=' + system if system }}{{ '&grp=' + group if group }}{{ '&freq=' + freq_filter if freq_filter }}{{ '&h=' + hours|string if hours != 24 }}{{ '&nb=1' if hide_blank }}{{ '&ps=' + page_size|string if page_size != 1000 }}{{ '&page=' + (page - 1)|string }}" class="prev-btn">&lt; Prev</a>{% endif %}
<span class="page-indicator">Page {{ page }}/{{ total_pages }}</span>
{% if page < total_pages %}<a href="?q={{ query }}{{ '&ch=' + channel if channel }}{{ '&sys=' + system if system }}{{ '&grp=' + group if group }}{{ '&freq=' + freq_filter if freq_filter }}{{ '&h=' + hours|string if hours != 24 }}{{ '&nb=1' if hide_blank }}{{ '&ps=' + page_size|string if page_size != 1000 }}{{ '&page=' + (page + 1)|string }}" class="next-btn">Next &gt;</a>{% endif %}
</span>
{% endif %}
</div>
<div class="status">
<div class="scanner-screen">
<div class="screen-title">BCD436HP &mdash; <span id="status-state">{{ status.state }}</span></div>
<div id="screen-content">
{% if status.screen and status.screen.sections is defined %}
<div class="screen-header">
{% if status.screen.header is iterable and status.screen.header is not string %}
{% for h in status.screen.header %}{{ h }}<br>{% endfor %}
{% else %}{{ status.screen.header }}{% endif %}
</div>
{% for section in status.screen.sections %}
<div class="screen-section">{% for line in section %}{{ line }}<br>{% endfor %}</div>
{% endfor %}
{% if status.screen.footer %}<div class="screen-footer">{{ status.screen.footer }}</div>{% endif %}
{% else %}
{% for line in status.screen %}<div class="screen-line">{{ line }}</div>{% endfor %}
{% endif %}
</div>
</div>
</div>"""

HTML += """<form class="controls" method="GET" id="filter-form">
<input type="text" name="q" placeholder="Search text..." value="{{ query }}" id="flt-q">
<select name="sys" id="flt-sys"><option value="">All Systems</option>
{% for s in systems %}<option value="{{ s.value }}" {{ 'selected' if s.value==system }}>{{ s.value }} [{{ s.count }}]</option>{% endfor %}
</select>
<select name="grp" id="flt-grp"><option value="">All Groups</option>
{% for g in groups %}<option value="{{ g.value }}" {{ 'selected' if g.value==group }}>{{ g.value }} [{{ g.count }}]</option>{% endfor %}
</select>
<select name="ch" id="flt-ch"><option value="">All Channels</option>
{% for c in channels %}<option value="{{ c.value }}" {{ 'selected' if c.value==channel }}>{{ c.value }} [{{ c.count }}]</option>{% endfor %}
</select>
<select name="freq" id="flt-freq"><option value="">All Freqs</option>
{% for f in freqs %}<option value="{{ f.value }}" {{ 'selected' if f.value==freq_filter }}>{{ f.value }} [{{ f.count }}]</option>{% endfor %}
</select>
<select name="h" id="flt-hours">
<option value="1" {{'selected' if hours==1}}>1h</option>
<option value="4" {{'selected' if hours==4}}>4h</option>
<option value="12" {{'selected' if hours==12}}>12h</option>
<option value="24" {{'selected' if hours==24}}>24h</option>
<option value="48" {{'selected' if hours==48}}>48h</option>
<option value="168" {{'selected' if hours==168}}>7d</option>
<option value="9999" {{'selected' if hours==9999}}>All</option>
</select>
<label><input type="checkbox" name="nb" value="1" {{ 'checked' if hide_blank }}> Hide blank</label>
<select name="ps" id="flt-ps">
<option value="100" {{'selected' if page_size==100}}>100/pg</option>
<option value="500" {{'selected' if page_size==500}}>500/pg</option>
<option value="1000" {{'selected' if page_size==1000}}>1000/pg</option>
<option value="2000" {{'selected' if page_size==2000}}>2000/pg</option>
<option value="5000" {{'selected' if page_size==5000}}>5000/pg</option>
</select>
</form>
<div class="decoded-filters" id="decoded-filters">
<span class="df-label">Decoded:</span>
<button type="button" class="df-btn active" data-dtype="dtmf">DTMF</button>
<button type="button" class="df-btn active" data-dtype="morse">CW</button>
<button type="button" class="df-btn active" data-dtype="fsk">FSK</button>
<button type="button" class="df-btn active" data-dtype="tones">Tones</button>
<button type="button" class="df-btn active" data-dtype="codes">Codes</button>
<button type="button" class="df-btn active" data-dtype="plates">Plates</button>
<button type="button" class="df-btn active" data-dtype="phones">Phones</button>
</div>
{% if results %}
<div class="table-wrap">
<table id="scan-table">
<thead><tr>
<th data-col="time" data-field="time">Time<span class="sort-arrow"></span><span class="resize-handle"></span></th>
<th data-col="text" data-field="text">Text<span class="sort-arrow"></span><span class="resize-handle"></span></th>
<th data-col="decoded" data-field="decoded">Decoded<span class="sort-arrow"></span><span class="resize-handle"></span></th>
<th data-col="dur" data-field="duration_sec">Dur<span class="sort-arrow"></span><span class="resize-handle"></span></th>
<th data-col="sys" data-field="system">System<span class="sort-arrow"></span><span class="resize-handle"></span></th>
<th data-col="grp" data-field="group">Group<span class="sort-arrow"></span><span class="resize-handle"></span></th>
<th data-col="ch" data-field="channel">Channel<span class="sort-arrow"></span><span class="resize-handle"></span></th>
<th data-col="freq" data-field="frequency">Freq<span class="sort-arrow"></span><span class="resize-handle"></span></th>
</tr></thead>
<tbody style="visibility:hidden">
{% for r in results|reverse %}
<tr>
<td class="col-time" data-col="time">{{ r.time[5:19] }}</td>
<td class="col-text" data-col="text">
{% if r.clip %}<button class="play-btn" data-audio="{{ r.clip }}" title="Play"><span class="play-icon">&#9654;</span></button>{% endif %}
{% if r.text == 'Transcribing...' %}<span class="transcribing">Transcribing... [{{ r.queue_pos }}]</span>{% elif r.text == 'Transcribing now' %}<span class="transcribing" style="color:#4fc3f7">Transcribing now</span>{% elif r.text and r.text not in ('[BLANK_AUDIO]', '(no speech)') %}{{ r.text_hl|safe }}{% elif r.text == '[BLANK_AUDIO]' %}<span class="blank">blank</span>{% else %}<span class="blank">-</span>{% endif %}
</td>
<td class="col-decoded" data-col="decoded">{{ r.decoded_display|safe }}</td>
<td class="col-dur" data-col="dur">{{ r.duration_sec }}s</td>
<td class="col-sys" data-col="sys" title="{{ r.system }}">{{ r.system }}</td>
<td class="col-grp" data-col="grp" title="{{ r.group }}">{{ r.group }}</td>
<td class="col-ch" data-col="ch" title="{{ r.channel }}">{{ r.channel }}</td>
<td class="col-freq" data-col="freq">{{ r.frequency }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>
{% else %}
<div class="empty">No transmissions found.</div>
{% endif %}"""

HTML += """<script>
// ========== Column management: sort, resize, drag, persist ==========
(function() {
    const STORAGE_KEY = 'scanner_dashboard_cols';
    const table = document.getElementById('scan-table');
    if (!table) return;
    const thead = table.querySelector('thead tr');
    const ths = () => Array.from(thead.querySelectorAll('th'));

    function loadConfig() {
        try { const r = localStorage.getItem(STORAGE_KEY); if (r) return JSON.parse(r); } catch(e) {}
        return null;
    }
    function saveConfig() {
        const config = ths().map(th => ({ col: th.dataset.col, width: th.style.width ? parseInt(th.style.width) : 0 }));
        localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    }
    function applyConfig(config) {
        if (!config || !config.length) return;
        // Only apply saved widths (no column reorder — breaks AJAX refresh)
        const thMap = {}; ths().forEach(th => { thMap[th.dataset.col] = th; });
        config.forEach(c => { const th = thMap[c.col]; if (th && c.width > 0) { th.style.width = c.width+'px'; th.style.minWidth = c.width+'px'; }});
    }
    applyConfig(loadConfig());

    // ===== Sorting =====
    let sortField = null, sortDir = 'desc';
    let didDrag = false;
    window._activeSortCol = undefined;

    function _applySortToBody() {
        const table = document.getElementById('scan-table');
        if (!table || sortField === null) return;
        const allThs = ths();
        const th = allThs.find(h => h.dataset.field === sortField);
        if (!th) return;
        const colIdx = allThs.indexOf(th);
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
            let va = (a.children[colIdx] ? a.children[colIdx].textContent : '').trim();
            let vb = (b.children[colIdx] ? b.children[colIdx].textContent : '').trim();
            const na = parseFloat(va), nb = parseFloat(vb);
            if (!isNaN(na) && !isNaN(nb)) return sortDir === 'asc' ? na - nb : nb - na;
            return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        });
        rows.forEach(r => tbody.appendChild(r));
    }
    window._applySortToBody = _applySortToBody;

    thead.addEventListener('click', function(e) {
        if (didDrag) { didDrag = false; return; }
        const th = e.target.closest('th');
        if (!th || e.target.closest('.resize-handle')) return;
        const field = th.dataset.field; if (!field) return;
        if (sortField === field) { sortDir = sortDir === 'asc' ? 'desc' : 'asc'; }
        else { sortField = field; sortDir = 'desc'; }
        window._activeSortCol = sortField;
        ths().forEach(h => h.classList.remove('sort-asc','sort-desc'));
        th.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
        _applySortToBody();
    });

    // ===== Resizing =====
    let resizing = null;
    thead.addEventListener('mousedown', function(e) {
        const handle = e.target.closest('.resize-handle');
        if (!handle) return;
        e.preventDefault();
        const th = handle.parentElement;
        const startX = e.clientX, startW = th.offsetWidth;
        handle.classList.add('active');
        resizing = {th, startX, startW, handle};
        function onMove(ev) { if (!resizing) return; const w = Math.max(30, resizing.startW + ev.clientX - resizing.startX); resizing.th.style.width = w+'px'; resizing.th.style.minWidth = w+'px'; }
        function onUp() { if (resizing) resizing.handle.classList.remove('active'); resizing = null; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); saveConfig(); }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    // ===== Drag to reorder (disabled — incompatible with AJAX refresh) =====

    // Default sort: time descending (newest first)
    sortField = 'time'; sortDir = 'desc';
    window._activeSortCol = sortField;
    const timeTh = ths().find(h => h.dataset.field === 'time');
    if (timeTh) timeTh.classList.add('sort-desc');
    _applySortToBody();
})();
</script>"""

HTML += """<script>
// ========== Decoded type filters ==========
(function() {
    const DKEY = 'scanner_decoded_visible';
    const ALL_TYPES = ['dtmf','morse','fsk','tones','codes','plates','phones'];
    let visible = new Set(ALL_TYPES);

    function load() {
        try { const s = localStorage.getItem(DKEY); if (s) visible = new Set(JSON.parse(s)); } catch(e) {}
    }
    function save() { localStorage.setItem(DKEY, JSON.stringify(Array.from(visible))); }

    function apply() {
        document.querySelectorAll('[data-dtype]').forEach(el => {
            if (el.classList.contains('df-btn')) return;
            el.style.display = visible.has(el.dataset.dtype) ? '' : 'none';
        });
    }
    window._applyDecodedFilters = apply;

    load();
    const bar = document.getElementById('decoded-filters');
    if (!bar) return;
    bar.querySelectorAll('.df-btn').forEach(btn => {
        const t = btn.dataset.dtype;
        if (!visible.has(t)) btn.classList.remove('active');
        btn.addEventListener('click', () => {
            if (visible.has(t)) { visible.delete(t); btn.classList.remove('active'); }
            else { visible.add(t); btn.classList.add('active'); }
            save(); apply();
        });
    });
    apply();
})();
</script>"""

HTML += """<script>
// ========== Scanner screen, audio, sysinfo ==========
function updateScreen() {
    fetch('/api/status')
        .then(r => r.json())
        .then(d => {
            let html = '';
            let scr = d.screen;
            if (scr && scr.sections) {
                if (scr.header) { html += '<div class="screen-header">'; if (Array.isArray(scr.header)) { scr.header.forEach(h => { html += h + '<br>'; }); } else { html += scr.header; } html += '</div>'; }
                scr.sections.forEach(sec => { html += '<div class="screen-section">'; sec.forEach(line => { html += line + '<br>'; }); html += '</div>'; });
                if (scr.footer) html += '<div class="screen-footer">' + scr.footer + '</div>';
            } else if (scr && Array.isArray(scr)) {
                scr.forEach(l => { html += '<div class="screen-line">' + l + '</div>'; });
            } else { html = '<div class="screen-line" style="color:#555">No data</div>'; }
            document.getElementById('screen-content').innerHTML = html;
            document.getElementById('status-state').textContent = d.state || 'unknown';
            document.getElementById('status-queue').textContent = d.queue || 0;
        }).catch(() => {});
}
let currentAudio = null, currentButton = null;
function stopAudio() { if (currentAudio) { currentAudio.pause(); currentAudio = null; } if (currentButton) { currentButton.classList.remove('playing'); currentButton.querySelector('.play-icon').textContent = '\u25b6'; currentButton = null; } }
document.addEventListener('click', function(e) {
    const btn = e.target.closest('.play-btn');
    if (btn) { e.stopPropagation(); if (currentButton === btn) { stopAudio(); return; } stopAudio(); btn.classList.add('playing'); btn.querySelector('.play-icon').textContent = '\u23f8'; currentButton = btn; currentAudio = new Audio('/audio/' + encodeURIComponent(btn.dataset.audio)); currentAudio.addEventListener('ended', stopAudio); currentAudio.play(); }
});
function updateSysinfo() { fetch('/api/sysinfo').then(r => r.json()).then(d => { document.getElementById('sys-cpu').textContent = d.cpu; document.getElementById('sys-ram').textContent = d.ram; document.getElementById('sys-temp').textContent = d.temp; }).catch(() => {}); }
updateSysinfo();
setInterval(updateSysinfo, 3000);
setInterval(updateScreen, 2000);

// ========== AJAX table refresh (no full page reload) ==========
function refreshTable() {
    const params = new URLSearchParams(window.location.search);
    params.set('_t', Date.now());
    fetch('/api/table?' + params.toString())
        .then(r => r.json())
        .then(d => {
            const tbody = document.querySelector('#scan-table tbody');
            if (tbody && d.html) {
                tbody.innerHTML = d.html;
                tbody.style.visibility = 'visible';
                // Re-apply active sort after refresh
                if (window._activeSortCol !== undefined) {
                    _applySortToBody();
                }
                // Re-apply decoded type filters after refresh
                if (window._applyDecodedFilters) {
                    window._applyDecodedFilters();
                }
            }
            const spans = document.querySelectorAll('.stats-bar span b');
            spans.forEach(b => {
                const parent = b.parentElement;
                if (parent && parent.textContent.includes('Records')) b.textContent = d.total;
                if (parent && parent.textContent.includes('Showing')) b.textContent = d.showing;
            });
        }).catch(() => {});
}
setInterval(refreshTable, 10000);
refreshTable();
</script>"""

HTML += """<script>
// ========== Cascading filter dropdowns ==========
(function() {
    const FILTER_KEY = 'scanner_dashboard_filters';
    const sysSel = document.getElementById('flt-sys');
    const grpSel = document.getElementById('flt-grp');
    const chSel = document.getElementById('flt-ch');
    const freqSel = document.getElementById('flt-freq');
    const hoursSel = document.getElementById('flt-hours');
    if (!sysSel) return;

    // Hierarchy: System > Group > Channel > Freq
    const hierarchy = [sysSel, grpSel, chSel, freqSel];

    function saveFilters() {
        const state = { sys: sysSel.value, grp: grpSel.value, ch: chSel.value, freq: freqSel.value };
        localStorage.setItem(FILTER_KEY, JSON.stringify(state));
    }
    function loadFilters() {
        try { const r = localStorage.getItem(FILTER_KEY); if (r) return JSON.parse(r); } catch(e) {}
        return null;
    }

    // On page load, if URL has filter params, save them to localStorage
    const urlParams = new URLSearchParams(window.location.search);
    const hasUrlFilters = urlParams.has('sys') || urlParams.has('grp') || urlParams.has('ch') || urlParams.has('freq');
    if (hasUrlFilters) {
        saveFilters();
    }

    function repopulate(select, options, label) {
        const current = select.value;
        select.innerHTML = '<option value="">All ' + label + '</option>';
        options.forEach(item => {
            const v = item.value || item;
            const cnt = item.count;
            const opt = document.createElement('option');
            opt.value = v;
            opt.textContent = cnt !== undefined ? v + ' [' + cnt + ']' : v;
            if (v === current) opt.selected = true;
            select.appendChild(opt);
        });
        if (current && !options.some(item => (item.value || item) === current)) {
            select.value = '';
        }
        // Auto-select if only one option available
        if (!select.value && options.length === 1) {
            select.value = options[0].value || options[0];
        }
    }

    function clearDownstream(sel) {
        // Clear all dropdowns below sel in the hierarchy
        const idx = hierarchy.indexOf(sel);
        for (let i = idx + 1; i < hierarchy.length; i++) {
            hierarchy[i].value = '';
        }
    }

    function fetchOptionsAndSubmit() {
        saveFilters();
        const params = new URLSearchParams();
        if (sysSel.value) params.set('sys', sysSel.value);
        if (grpSel.value) params.set('grp', grpSel.value);
        if (chSel.value) params.set('ch', chSel.value);
        if (freqSel.value) params.set('freq', freqSel.value);
        params.set('h', hoursSel.value);
        params.set('_t', Date.now());

        fetch('/api/filter_options?' + params.toString())
            .then(r => r.json())
            .then(d => {
                if (!d || !d.systems) { submitForm(); return; }
                repopulate(sysSel, d.systems || [], 'Systems');
                repopulate(grpSel, d.groups || [], 'Groups');
                repopulate(chSel, d.channels || [], 'Channels');
                repopulate(freqSel, d.freqs || [], 'Freqs');
                saveFilters();
                submitForm();
            }).catch(() => { submitForm(); });
    }

    function onDropdownChange(e) {
        // If user selected "All" (empty value), clear all downstream
        if (!e.target.value) {
            clearDownstream(e.target);
        }
        fetchOptionsAndSubmit();
    }

    sysSel.addEventListener('change', onDropdownChange);
    grpSel.addEventListener('change', onDropdownChange);
    chSel.addEventListener('change', onDropdownChange);
    freqSel.addEventListener('change', onDropdownChange);
    hoursSel.addEventListener('change', function() { fetchOptionsAndSubmit(); });
    var psSel = document.getElementById('flt-ps');
    if (psSel) psSel.addEventListener('change', function() { submitForm(); });

    // Initial options sync on page load (populate dropdowns only, no auto-select)
    (function() {
        const params = new URLSearchParams();
        if (sysSel.value) params.set('sys', sysSel.value);
        if (grpSel.value) params.set('grp', grpSel.value);
        if (chSel.value) params.set('ch', chSel.value);
        if (freqSel.value) params.set('freq', freqSel.value);
        params.set('h', hoursSel.value);
        params.set('_t', Date.now());
        fetch('/api/filter_options?' + params.toString())
            .then(r => r.json())
            .then(d => {
                if (!d || !d.systems) return;
                // Populate without auto-select (pass false flag via temp override)
                var saved = select => select.value;
                [sysSel, grpSel, chSel, freqSel].forEach(sel => {
                    var cur = sel.value;
                    var label = sel === sysSel ? 'Systems' : sel === grpSel ? 'Groups' : sel === chSel ? 'Channels' : 'Freqs';
                    var opts = sel === sysSel ? d.systems : sel === grpSel ? d.groups : sel === chSel ? d.channels : d.freqs;
                    sel.innerHTML = '<option value="">All ' + label + '</option>';
                    (opts||[]).forEach(item => {
                        var v = item.value || item;
                        var cnt = item.count;
                        var opt = document.createElement('option');
                        opt.value = v;
                        opt.textContent = cnt !== undefined ? v + ' [' + cnt + ']' : v;
                        if (v === cur) opt.selected = true;
                        sel.appendChild(opt);
                    });
                });
            }).catch(() => {});
    })();

    // --- Auto-submit: form controls ---
    const form = document.getElementById('filter-form');
    const qInput = document.getElementById('flt-q');
    const nbCheckbox = form.querySelector('input[name="nb"]');

    function submitForm() { form.submit(); }

    if (nbCheckbox) nbCheckbox.addEventListener('change', submitForm);

    // Text input: submit on Enter, or after 1s debounce
    let debounceTimer = null;
    qInput.addEventListener('keydown', function(e) { if (e.key === 'Enter') { e.preventDefault(); submitForm(); } });
    qInput.addEventListener('input', function() { clearTimeout(debounceTimer); debounceTimer = setTimeout(submitForm, 1000); });
})();
</script>
</body>
</html>"""


# --- Row Template (for AJAX table refresh) ----------------------------------

ROW_TEMPLATE = """{% for r in results %}
<tr>
<td class="col-time" data-col="time">{{ r.time[5:19] }}</td>
<td class="col-text" data-col="text">
{% if r.clip %}<button class="play-btn" data-audio="{{ r.clip }}" title="Play"><span class="play-icon">&#9654;</span></button>{% endif %}
{% if r.text == 'Transcribing...' %}<span class="transcribing">Transcribing... [{{ r.queue_pos }}]</span>{% elif r.text == 'Transcribing now' %}<span class="transcribing" style="color:#4fc3f7">Transcribing now</span>{% elif r.text and r.text not in ('[BLANK_AUDIO]', '(no speech)') %}{{ r.text_hl|safe }}{% elif r.text == '[BLANK_AUDIO]' %}<span class="blank">blank</span>{% else %}<span class="blank">-</span>{% endif %}
</td>
<td class="col-decoded" data-col="decoded">{{ r.decoded_display|safe }}</td>
<td class="col-dur" data-col="dur">{{ r.duration_sec }}s</td>
<td class="col-sys" data-col="sys" title="{{ r.system }}">{{ r.system }}</td>
<td class="col-grp" data-col="grp" title="{{ r.group }}">{{ r.group }}</td>
<td class="col-ch" data-col="ch" title="{{ r.channel }}">{{ r.channel }}</td>
<td class="col-freq" data-col="freq">{{ r.frequency }}</td>
</tr>
{% endfor %}"""


# --- Flask Routes -----------------------------------------------------------

@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    channel = request.args.get("ch", "").strip()
    system = request.args.get("sys", "").strip()
    group = request.args.get("grp", "").strip()
    freq_filter = request.args.get("freq", "").strip()
    hours = int(request.args.get("h") or 24)
    hide_blank = request.args.get("nb", "") == "1"
    
    # Pagination
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(5000, max(10, int(request.args.get("ps", 1000))))
    
    results = scanner_db.get_transmissions(
        hours=hours, system=system, group=group, channel=channel,
        freq=freq_filter, query=query, hide_blank=hide_blank, limit=page_size,
        offset=(page - 1) * page_size
    )
    total = scanner_db.get_filtered_count(
        hours=hours, system=system, group=group, channel=channel,
        freq=freq_filter, query=query, hide_blank=hide_blank
    )
    status = _load_status()
    pending = scanner_db.get_pending_count()
    
    # Calculate pagination
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    start_rec = (page - 1) * page_size + 1 if total > 0 else 0
    end_rec = total if page >= total_pages else page * page_size

    # Get cascading filter options
    opts = scanner_db.get_filter_options(
        hours=hours, system=system, group=group, channel=channel, freq=freq_filter
    )

    _enrich_results(results, query)

    from flask import make_response
    resp = make_response(render_template_string(
        HTML, results=results, total=total,
        query=query, channel=channel, system=system,
        group=group, freq_filter=freq_filter,
        hours=hours, hide_blank=hide_blank,
        status=status,
        systems=opts["systems"], groups=opts["groups"],
        channels=opts["channels"], freqs=opts["freqs"],
        page=page, total_pages=total_pages, start_rec=start_rec, end_rec=end_rec,
        pending=pending, page_size=page_size
    ))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route("/api/table")
def api_table():
    """Return table HTML + stats for AJAX refresh (no full page reload)."""
    query = request.args.get("q", "").strip()
    channel = request.args.get("ch", "").strip()
    system = request.args.get("sys", "").strip()
    group = request.args.get("grp", "").strip()
    freq_filter = request.args.get("freq", "").strip()
    hours = int(request.args.get("h") or 24)
    hide_blank = request.args.get("nb", "") == "1"
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(5000, max(10, int(request.args.get("ps", 1000))))

    results = scanner_db.get_transmissions(
        hours=hours, system=system, group=group, channel=channel,
        freq=freq_filter, query=query, hide_blank=hide_blank, limit=page_size,
        offset=(page - 1) * page_size
    )
    total = scanner_db.get_filtered_count(
        hours=hours, system=system, group=group, channel=channel,
        freq=freq_filter, query=query, hide_blank=hide_blank
    )

    _enrich_results(results, query)

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    html = render_template_string(ROW_TEMPLATE, results=results)
    return jsonify({
        "html": html,
        "total": total,
        "showing": len(results),
        "page": page,
        "total_pages": total_pages,
        "start_rec": (page - 1) * page_size + 1 if total > 0 else 0,
        "end_rec": total if page >= total_pages else page * page_size,
    })


@app.route("/api/filter_options")
def api_filter_options():
    """Return valid dropdown options given current filter selections (cascading)."""
    system = request.args.get("sys", "").strip()
    group = request.args.get("grp", "").strip()
    channel = request.args.get("ch", "").strip()
    freq = request.args.get("freq", "").strip()
    hours = int(request.args.get("h") or 24)

    opts = scanner_db.get_filter_options(
        hours=hours, system=system, group=group, channel=channel, freq=freq
    )
    return jsonify(opts)


@app.route("/api/status")
def api_status():
    data = _load_status()
    data["queue"] = scanner_db.get_pending_count()
    # Strip control characters from screen data that can break browser rendering
    def _sanitize_screen(obj):
        if isinstance(obj, str):
            return ''.join(c for c in obj if c >= ' ' or c in '\n\r\t')
        if isinstance(obj, list):
            return [_sanitize_screen(item) for item in obj]
        if isinstance(obj, dict):
            return {k: _sanitize_screen(v) for k, v in obj.items()}
        return obj
    if "screen" in data:
        data["screen"] = _sanitize_screen(data["screen"])
    return jsonify(data)


@app.route("/api/sysinfo")
def api_sysinfo():
    """Return CPU%, RAM%, and CPU temperature."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["bash", "-c",
             "read c1 i1 < <(head -1 /proc/stat | awk '{print $2+$3+$4+$6+$7+$8, $5}'); "
             "sleep 1; "
             "read c2 i2 < <(head -1 /proc/stat | awk '{print $2+$3+$4+$6+$7+$8, $5}'); "
             "echo $(( (c2-c1)*100 / (c2-c1+i2-i1) ))"],
            timeout=3, text=True
        ).strip()
        cpu = int(out)
    except Exception:
        try:
            with open('/proc/loadavg') as f:
                load1 = float(f.read().split()[0])
            import multiprocessing
            cores = multiprocessing.cpu_count()
            cpu = min(100, int(load1 / cores * 100))
        except Exception:
            cpu = 0
    try:
        with open('/proc/meminfo') as f:
            mem = {}
            for line in f:
                parts = line.split()
                if parts[0] in ('MemTotal:', 'MemAvailable:'):
                    mem[parts[0]] = int(parts[1])
        total = mem.get('MemTotal:', 1)
        avail = mem.get('MemAvailable:', 0)
        ram = int((total - avail) / total * 100)
    except Exception:
        ram = 0
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = round(int(f.read().strip()) / 1000, 1)
    except Exception:
        temp = 0
    return jsonify({"cpu": cpu, "ram": ram, "temp": temp})


@app.route("/audio/<path:audio_path>")
def serve_audio(audio_path):
    """Serve audio files."""
    try:
        decoded_path = urllib.parse.unquote(audio_path)
        if decoded_path.startswith('/'):
            audio_file = Path(decoded_path)
        else:
            audio_file = Path('/' + decoded_path)
        if not audio_file.exists():
            # Try .mp3 version if .wav is missing (converted but DB not updated)
            if audio_file.suffix == '.wav':
                mp3_file = audio_file.with_suffix('.mp3')
                if mp3_file.exists():
                    audio_file = mp3_file
                else:
                    return "Audio file not found", 404
            else:
                return "Audio file not found", 404
        mimetype = 'audio/mpeg' if audio_file.suffix == '.mp3' else 'audio/wav'
        return send_file(
            audio_file,
            mimetype=mimetype,
            as_attachment=False,
            download_name=audio_file.name
        )
    except Exception as e:
        return f"Error serving audio: {str(e)}", 500


@app.route("/api/untranscribed")
def api_untranscribed():
    """Return untranscribed records for the GPU server to pick up."""
    limit = int(request.args.get("limit", 10))
    records = scanner_db.get_untranscribed(limit=limit)
    return jsonify({"count": len(records), "records": records})


@app.route("/api/pi_transcribed")
def api_pi_transcribed():
    """Return records transcribed by Pi but not yet by GPU (for re-transcription)."""
    limit = int(request.args.get("limit", 100))
    records = scanner_db.get_pi_transcribed(limit=limit)
    return jsonify({"count": len(records), "records": records})


@app.route("/api/transcribe_result", methods=["POST"])
def api_transcribe_result():
    """GPU server posts transcription results back."""
    data = request.get_json(force=True)
    record_id = data.get("id")
    text = data.get("text", "")
    transcribed_by = data.get("transcribed_by", "gpu")

    if not record_id:
        return jsonify({"error": "missing id"}), 400

    updates = {
        "text": text,
        "transcribed": True,
        "transcribed_by": transcribed_by,
    }
    # Run text decoders inline (codes, plates, phones)
    if text and text not in ('(no speech)', '[BLANK_AUDIO]', ''):
        with scanner_db.get_db() as conn:
            row = conn.execute("SELECT channel FROM transmissions WHERE id = ?", (record_id,)).fetchone()
            channel_name = row["channel"] if row else ""
        decoded_text = _run_text_decoders(text, channel_name)
        if decoded_text:
            updates["decoded_text"] = decoded_text
    success = scanner_db.update_transmission(record_id, updates)
    if success:
        return jsonify({"status": "ok", "id": record_id})
    else:
        return jsonify({"error": "record not found or no update"}), 404


@app.route("/api/batch_transcribe_result", methods=["POST"])
def api_batch_transcribe_result():
    """GPU server posts multiple transcription results in one batch."""
    data = request.get_json(force=True)
    results = data.get("results", [])
    
    if not results:
        return jsonify({"error": "missing results array"}), 400
    
    batch_updates = []
    for item in results:
        record_id = item.get("id")
        text = item.get("text", "")
        transcribed_by = item.get("transcribed_by", "gpu")
        
        if not record_id:
            continue
        
        batch_updates.append({
            "id": record_id,
            "text": text,
            "transcribed": True,
            "transcribed_by": transcribed_by,
        })
    
    if not batch_updates:
        return jsonify({"error": "no valid records in batch"}), 400
    
    success_count = 0
    for update in batch_updates:
        updates = {
            "text": update["text"],
            "transcribed": True,
            "transcribed_by": update["transcribed_by"],
        }
        # Run text decoders inline (codes, plates, phones)
        text = update["text"]
        if text and text not in ('(no speech)', '[BLANK_AUDIO]', ''):
            with scanner_db.get_db() as conn:
                row = conn.execute("SELECT channel FROM transmissions WHERE id = ?", (update["id"],)).fetchone()
                channel_name = row["channel"] if row else ""
            decoded_text = _run_text_decoders(text, channel_name)
            if decoded_text:
                updates["decoded_text"] = decoded_text
        if scanner_db.update_transmission(update["id"], updates):
            success_count += 1
    
    return jsonify({
        "status": "ok",
        "processed": len(batch_updates),
        "success": success_count,
        "failed": len(batch_updates) - success_count,
    })


@app.route("/api/gpu_status")
def api_gpu_status():
    """Proxy to GPU server status endpoint."""
    try:
        resp = requests.get(f"{GPU_SERVER_URL}/status", timeout=5)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e), "gpu_server": GPU_SERVER_URL}), 502


# --- Main -------------------------------------------------------------------

if __name__ == "__main__":
    scanner_db.init_db()
    print("Pi Scanner Dashboard (SQLite)")
    print("Access from any device: http://pi3:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
