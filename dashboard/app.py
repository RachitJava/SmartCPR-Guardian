"""
SmartCPR Guardian — Real-Time Dashboard
Flask + SocketIO web dashboard for live monitoring.
Shows: ECG stream, vitals, FOLD-RM alerts, counterfactuals.
"""

import eventlet
eventlet.monkey_patch()
print("Eventlet monkey patch applied at startup.")

import os
import sys
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.fold_rm import FoldRMCardiacClassifier
from ai_engine.feature_extractor import ECGFeatureExtractor, PatientVitalsSimulator
from alert_system.dispatcher import EmergencyDispatcher, EmergencyAlert, Location
from cloud_ai.counterfactual import MC3GCounterfactualGenerator
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "smartcpr-guardian-2025"
socketio = SocketIO(app, cors_allowed_origins="*")

# Global simulation state
current_scenario = "normal"
simulation_active = True
SIMULATION_SETTINGS = {
    "normal":         {"hr": 72, "spo2": 99, "bp": 120},
    "vfib":           {"spo2": 75, "bp": 40},
    "asystole":       {"spo2": 60, "bp": 30},
    "bradycardia":    {"hr": 18, "spo2": 87, "bp": 58},
    "pre_arrest":     {"hr": 18, "spo2": 87, "bp": 58},
    "heart_failure":  {"hr": 105, "spo2": 89, "bp": 74},
    "tachycardia":    {"hr": 190, "spo2": 94, "bp": 105}
}

print("Initializing AI components...")
classifier = FoldRMCardiacClassifier()
print("Classifier initialized.")
extractor = ECGFeatureExtractor()
print("Extractor initialized.")
dispatcher = EmergencyDispatcher()
print("Dispatcher initialized.")
cf_generator = MC3GCounterfactualGenerator()
print("CF Generator initialized.")

background_thread = None
thread_lock = threading.Lock()

@socketio.on('connect')
def handle_connect():
    global background_thread
    with thread_lock:
        if background_thread is None:
            print("🚀 Starting background monitoring task...")
            background_thread = socketio.start_background_task(target=monitoring_loop)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartCPR Guardian — Live Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0f1e;
    --card: #111827;
    --border: #1f2937;
    --green: #10b981;
    --red: #ef4444;
    --amber: #f59e0b;
    --blue: #3b82f6;
    --purple: #8b5cf6;
    --text: #f9fafb;
    --muted: #6b7280;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* Header */
  .header {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .logo { display:flex; align-items:center; gap:12px; }
  .logo-icon {
    width:42px; height:42px; background: linear-gradient(135deg,#ef4444,#dc2626);
    border-radius:10px; display:flex; align-items:center; justify-content:center;
    font-size:22px; animation: heartbeat 1.5s ease-in-out infinite;
  }
  @keyframes heartbeat {
    0%,100%{transform:scale(1)} 50%{transform:scale(1.12)}
  }
  .logo-text h1 { font-size:1.3rem; font-weight:700; color:var(--text); }
  .logo-text p  { font-size:0.75rem; color:var(--muted); }
  .status-badge {
    padding:6px 16px; border-radius:999px; font-size:0.8rem; font-weight:600;
    display:flex; align-items:center; gap:6px;
  }
  .status-badge.active {
    background:rgba(16,185,129,0.15); color:var(--green); border:1px solid rgba(16,185,129,0.3);
  }
  .status-badge.alert {
    background:rgba(239,68,68,0.15); color:var(--red); border:1px solid rgba(239,68,68,0.4);
    animation: blink 0.6s ease-in-out infinite;
  }
  @keyframes blink { 50%{opacity:0.4} }
  .dot { width:8px; height:8px; border-radius:50%; background:currentColor; }

  /* Layout */
  .main { padding:24px 32px; display:grid; gap:20px; }
  .top-row { display:grid; grid-template-columns: repeat(5, 1fr); gap:16px; }
  .middle-row { display:grid; grid-template-columns: 2fr 1fr; gap:20px; }
  .bottom-row { display:grid; grid-template-columns: 1fr 1fr; gap:20px; }

  /* Vital Cards */
  .vital-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius:14px; padding:18px;
    transition: border-color 0.3s, box-shadow 0.3s;
  }
  .vital-card:hover { border-color:#374151; box-shadow:0 4px 24px rgba(0,0,0,0.4); }
  .vital-card.danger {
    border-color: var(--red);
    box-shadow: 0 0 20px rgba(239,68,68,0.2);
    animation: pulse-danger 1s ease-in-out infinite;
  }
  @keyframes pulse-danger {
    0%,100%{box-shadow:0 0 20px rgba(239,68,68,0.2)}
    50%{box-shadow:0 0 30px rgba(239,68,68,0.4)}
  }
  .vital-label { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
  .vital-value { font-size:2rem; font-weight:700; font-family:'JetBrains Mono', monospace; }
  .vital-value.green  { color: var(--green); }
  .vital-value.red    { color: var(--red); }
  .vital-value.amber  { color: var(--amber); }
  .vital-unit { font-size:0.7rem; color:var(--muted); margin-top:2px; }
  .vital-bar { height:3px; background:var(--border); border-radius:2px; margin-top:10px; }
  .vital-bar-fill { height:100%; border-radius:2px; transition: width 0.5s; }

  /* ECG Chart */
  .chart-card {
    background: var(--card); border:1px solid var(--border); border-radius:14px; overflow:hidden;
  }
  .chart-header {
    padding:14px 18px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between;
  }
  .chart-title { font-size:0.85rem; font-weight:600; display:flex; align-items:center; gap:8px; }
  .live-dot {
    width:8px; height:8px; border-radius:50%; background:var(--green);
    animation: blink 1s ease-in-out infinite;
  }

  /* AI Panel */
  .ai-card {
    background: var(--card); border:1px solid var(--border); border-radius:14px; overflow:hidden;
  }
  .ai-header {
    padding:14px 18px; border-bottom:1px solid var(--border);
    background: linear-gradient(135deg, rgba(139,92,246,0.1), transparent);
  }
  .ai-body { padding:16px; max-height:320px; overflow-y:auto; }
  .ai-body::-webkit-scrollbar { width:4px; }
  .ai-body::-webkit-scrollbar-track { background:var(--border); }
  .ai-body::-webkit-scrollbar-thumb { background:var(--purple); border-radius:2px; }

  .rule-chip {
    background: rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3);
    border-radius:8px; padding:8px 12px; margin-bottom:8px;
    font-size:0.75rem; font-family:'JetBrains Mono', monospace; color:#c4b5fd;
    line-height:1.5;
  }

  /* Counterfactual Panel */
  .cf-card {
    background: var(--card); border:1px solid var(--border); border-radius:14px; overflow:hidden;
  }
  .cf-body { padding:16px; max-height:280px; overflow-y:auto; }
  .cf-item {
    border-left:3px solid var(--blue); padding:10px 12px; margin-bottom:10px;
    background:rgba(59,130,246,0.05); border-radius:0 8px 8px 0;
  }
  .cf-item.urgent { border-color:var(--red); background:rgba(239,68,68,0.05); }
  .cf-feature { font-size:0.8rem; font-weight:600; color:var(--blue); margin-bottom:4px; }
  .cf-feature.urgent { color:var(--red); }
  .cf-action { font-size:0.75rem; color:var(--text); line-height:1.5; }
  .cf-timeline { font-size:0.65rem; color:var(--muted); margin-top:4px; }

  /* Alert Log */
  .log-card {
    background:var(--card); border:1px solid var(--border); border-radius:14px; overflow:hidden;
  }
  .log-body { padding:12px; max-height:240px; overflow-y:auto; }
  .log-entry {
    font-size:0.72rem; font-family:'JetBrains Mono', monospace;
    padding:6px 8px; border-radius:6px; margin-bottom:4px; border-left:3px solid var(--border);
  }
  .log-entry.error   { border-color:var(--red);   background:rgba(239,68,68,0.08); color:#fca5a5; }
  .log-entry.warning { border-color:var(--amber); background:rgba(245,158,11,0.08); color:#fcd34d; }
  .log-entry.success { border-color:var(--green); background:rgba(16,185,129,0.08); color:#6ee7b7; }
  .log-entry.info    { border-color:var(--blue);  background:rgba(59,130,246,0.08); color:#93c5fd; }

  /* Controls */
  .controls {
    display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:20px;
  }
  .btn {
    padding:9px 18px; border-radius:9px; font-size:0.82rem; font-weight:600;
    cursor:pointer; border:none; transition:all 0.2s;
  }
  .btn-green { background:var(--green); color:#fff; }
  .btn-green:hover { background:#059669; transform:translateY(-1px); }
  .btn-red   { background:var(--red);   color:#fff; }
  .btn-red:hover   { background:#dc2626; transform:translateY(-1px); }
  .btn-amber { background:var(--amber); color:#000; }
  .btn-amber:hover { background:#d97706; }
  .btn-blue  { background:var(--blue);  color:#fff; }
  .btn-blue:hover  { background:#2563eb; }
  .btn-purple{ background:var(--purple);color:#fff; }
  .btn-purple:hover{ background:#7c3aed; }

  .scenario-label { font-size:0.75rem; color:var(--muted); margin-right:4px; }

  /* CPR Indicator */
  .cpr-active {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border: 2px solid var(--red);
    border-radius:14px; padding:16px; text-align:center;
    animation: pulse-danger 0.8s ease-in-out infinite;
  }
  .cpr-text { font-size:1.1rem; font-weight:700; color:var(--red); }
  .cpr-rate  { font-size:0.75rem; color:var(--muted); margin-top:4px; }
  .cpr-hidden { display:none; }

  .section-title { font-size:0.78rem; font-weight:600; color:var(--muted);
    text-transform:uppercase; letter-spacing:0.1em; }
    
  /* Modal */
  .modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:1000; align-items:center; justify-content:center; }
  .modal-content { background:#111d2b; border:1px solid #2d3244; border-radius:16px; padding:28px; width:90%; max-height:85vh; display:flex; flex-direction:column; box-shadow:0 25px 50px -12px rgba(0,0,0,0.5); }
  
  /* Scrollbar override */
  #settingsForm::-webkit-scrollbar { width:8px; }
  #settingsForm::-webkit-scrollbar-track { background:#0a0c14; border-radius:10px; }
  #settingsForm::-webkit-scrollbar-thumb { background:#3b82f6; border-radius:10px; }
  #settingsForm::-webkit-scrollbar-thumb:hover { background:#60a5fa; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">
    <div class="logo-icon">🫀</div>
    <div class="logo-text">
      <h1>SmartCPR Guardian</h1>
      <p>AI-Driven Cardiac Arrest Detection & Response System</p>
    </div>
  </div>
  <div id="statusBadge" class="status-badge active">
    <div class="dot"></div>
    <span id="statusText">Monitoring Active</span>
  </div>
</div>

<div class="main">

  <!-- Controls -->
  <div class="controls">
    <span class="scenario-label">Simulate:</span>
    <button class="btn btn-green" onclick="setScenario('normal')">✅ Normal</button>
    <button class="btn btn-red"   onclick="setScenario('vfib')">⚡ VFib Arrest</button>
    <button class="btn btn-amber" onclick="setScenario('asystole')">📉 Asystole</button>
    <button class="btn btn-amber" onclick="setScenario('bradycardia')">🐌 Bradycardia</button>
    <button class="btn btn-blue"  onclick="setScenario('pre_arrest')">⚠️ Pre-Arrest</button>
    <button class="btn btn-blue"  onclick="setScenario('heart_failure')">🫀 Heart Failure</button>
    <button class="btn btn-purple" onclick="setScenario('tachycardia')">🏃 Tachycardia</button>
    <div style="margin-left:auto">
      <button class="btn btn-blue" onclick="showSettings()">⚙️ Settings</button>
      <button class="btn btn-purple" onclick="showRules()">📋 View FOLD-RM Rules</button>
    </div>
  </div>

  <!-- CPR Active Banner (hidden by default) -->
  <div id="cprBanner" class="cpr-hidden cpr-active">
    <div class="cpr-text">🤖 AUTO-CPR ACTIVE — 30 compressions/min</div>
    <div class="cpr-rate">AHA Standard | Defibrillation check every 2 min</div>
  </div>

  <!-- Vital Signs Row -->
  <div class="top-row">
    <div class="vital-card" id="card-hr">
      <div class="vital-label">❤️ Heart Rate</div>
      <div class="vital-value green" id="vital-hr">--</div>
      <div class="vital-unit">bpm</div>
      <div class="vital-bar"><div class="vital-bar-fill" id="bar-hr" style="width:70%;background:#10b981"></div></div>
    </div>
    <div class="vital-card" id="card-spo2">
      <div class="vital-label">🫁 SpO₂</div>
      <div class="vital-value green" id="vital-spo2">--</div>
      <div class="vital-unit">%</div>
      <div class="vital-bar"><div class="vital-bar-fill" id="bar-spo2" style="width:99%;background:#10b981"></div></div>
    </div>
    <div class="vital-card" id="card-bp">
      <div class="vital-label">💉 Systolic BP</div>
      <div class="vital-value green" id="vital-bp">--</div>
      <div class="vital-unit">mmHg</div>
      <div class="vital-bar"><div class="vital-bar-fill" id="bar-bp" style="width:60%;background:#10b981"></div></div>
    </div>
    <div class="vital-card" id="card-conf">
      <div class="vital-label">🧠 AI Confidence</div>
      <div class="vital-value green" id="vital-conf">--</div>
      <div class="vital-unit">arrest probability</div>
      <div class="vital-bar"><div class="vital-bar-fill" id="bar-conf" style="width:2%;background:#10b981"></div></div>
    </div>
    <div class="vital-card" id="card-label">
      <div class="vital-label">📊 FOLD-RM Label</div>
      <div class="vital-value green" id="vital-label" style="font-size:1rem">NORMAL</div>
      <div class="vital-unit">classification result</div>
    </div>
  </div>

  <!-- ECG + AI Rules -->
  <div class="middle-row">
    <div class="chart-card">
      <div class="chart-header">
        <div class="chart-title"><div class="live-dot"></div>Live ECG Waveform</div>
        <span style="font-size:0.7rem;color:var(--muted)">Sampling: 360 Hz | Window: 10s</span>
      </div>
      <div id="ecgChart" style="width:100%;height:240px;"></div>
    </div>
    <div class="ai-card">
      <div class="ai-header">
        <div class="section-title">🔬 FOLD-RM Rules Fired</div>
      </div>
      <div class="ai-body" id="rulesBody">
        <div style="color:var(--muted);font-size:0.8rem;padding:8px">
          Monitoring... No arrest rules active.
        </div>
      </div>
    </div>
  </div>

  <!-- Counterfactuals + Log -->
  <div class="bottom-row">
    <div class="cf-card">
      <div class="ai-header">
        <div class="section-title">🔄 MC3G Counterfactual Interventions</div>
      </div>
      <div class="cf-body" id="cfBody">
        <div style="color:var(--muted);font-size:0.8rem;padding:8px">
          No interventions needed — patient stable.
        </div>
      </div>
    </div>
    <div class="log-card">
      <div class="ai-header">
        <div class="section-title">📋 System Event Log</div>
      </div>
      <div class="log-body" id="logBody"></div>
    </div>
  </div>

</div>

<!-- FOLD-RM Rules Modal -->
<div id="rulesModal" class="modal">
  <div class="modal-content" style="max-width:700px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
      <h3 style="color:#f9fafb">FOLD-RM Cardiac Arrest Rules (AHA 2022)</h3>
      <button onclick="document.getElementById('rulesModal').style.display='none'" style="background:none;border:none;color:#6b7280;font-size:1.5rem;cursor:pointer">✕</button>
    </div>
    <div id="rulesModalContent" style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#c4b5fd;line-height:2"></div>
  </div>
</div>

<!-- Settings Modal -->
<div id="settingsModal" class="modal">
  <div class="modal-content" style="max-width:800px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <h3>⚙️ Simulation Settings</h3>
      <span class="close" onclick="closeSettings()" style="cursor:pointer;font-size:1.5rem">&times;</span>
    </div>
    <div id="settingsForm" style="display:grid;grid-template-columns:repeat(auto-fill, minmax(320px, 1fr));gap:20px;max-height:65vh;overflow-y:auto;padding:15px;margin-bottom:10px">
      <!-- Generated by JS -->
    </div>
    <div style="margin-top:20px;text-align:right">
      <button class="btn btn-blue" onclick="saveSettings()">💾 Save & Apply</button>
    </div>
  </div>
</div>

<script>
const socket = io();
let ecgBuffer = Array(360).fill(0);
let scenariosConfig = {};

socket.on('connect', () => addLog('info', '🔗 Connected to SmartCPR Guardian server'));

socket.on('vitals_update', (data) => {
  updateVitals(data);
  updateECG(data.ecg_sample);
  updateRules(data.fired_rules);
  updateCounterfactuals(data.counterfactuals);
  updateStatus(data.label, data.confidence);
});

socket.on('emergency_alert', (data) => {
  addLog('error', `🚨 CARDIAC ARREST DETECTED — Conf: ${(data.confidence*100).toFixed(0)}%`);
  addLog('error', `🚑 Ambulance dispatched | ETA: ${data.eta?.toFixed(0) ?? '?'} min`);
  addLog('success', '🤖 Auto-CPR initiated — 30 compressions/min');
  addLog('info', `🏥 Pre-arrival data sent to nearest cardiac ER`);
  document.getElementById('cprBanner').style.display = 'block';
  document.getElementById('cprBanner').className = 'cpr-active';
});

function updateVitals(data) {
  const hr   = data.vitals?.heart_rate ?? '--';
  const spo2 = data.vitals?.spo2 ?? '--';
  const bp   = data.vitals?.systolic_bp ?? '--';
  const conf = data.confidence ?? 0;
  const label = data.label ?? 'normal';

  setText('vital-hr',   typeof hr   === 'number' ? hr.toFixed(0)   : hr);
  setText('vital-spo2', typeof spo2 === 'number' ? spo2.toFixed(0) : spo2);
  setText('vital-bp',   typeof bp   === 'number' ? bp.toFixed(0)   : bp);
  setText('vital-conf', (conf * 100).toFixed(0) + '%');
  setText('vital-label', label.replace('_',' ').toUpperCase());

  // Color coding
  const isArrest = label !== 'normal';
  colorVital('vital-hr',   hr   < 40 || hr  > 150 ? 'red' : 'green');
  colorVital('vital-spo2', spo2 < 90 ? 'red' : spo2 < 95 ? 'amber' : 'green');
  colorVital('vital-bp',   bp   < 70 ? 'red' : bp < 90 ? 'amber' : 'green');
  colorVital('vital-conf', conf > 0.8 ? 'red' : conf > 0.5 ? 'amber' : 'green');
  colorVital('vital-label', isArrest ? 'red' : 'green');

  // Danger cards
  ['card-hr','card-spo2','card-bp','card-conf','card-label'].forEach(id => {
    document.getElementById(id).className = 'vital-card' + (isArrest ? ' danger' : '');
  });

  // Bars
  setBar('bar-hr',   Math.min(100, (hr/200)*100),   hr<40||hr>150?'#ef4444':'#10b981');
  setBar('bar-spo2', spo2,                           spo2<90?'#ef4444':'#10b981');
  setBar('bar-bp',   Math.min(100,(bp/180)*100),     bp<70?'#ef4444':'#10b981');
  setBar('bar-conf', conf*100,                       conf>0.8?'#ef4444':'#10b981');
}

function updateECG(sample) {
  if (!sample) return;
  ecgBuffer.push(...sample);
  ecgBuffer = ecgBuffer.slice(-1080); // 3s at 360Hz
  const x = Array.from({length: ecgBuffer.length}, (_,i) => i/360);
  const color = document.getElementById('vital-label').textContent.includes('ARREST')
    ? '#ef4444' : '#10b981';
  Plotly.react('ecgChart', [{
    x, y: ecgBuffer, type:'scatter', mode:'lines',
    line:{color, width:1.5, shape:'linear'},
  }], {
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    margin:{t:10,b:30,l:40,r:10},
    xaxis:{title:'Time (s)', color:'#6b7280', gridcolor:'#1f2937', showgrid:true},
    yaxis:{title:'mV', color:'#6b7280', gridcolor:'#1f2937', showgrid:true},
    showlegend:false,
    font:{family:'JetBrains Mono', color:'#6b7280', size:10},
  }, {responsive:true, displayModeBar:false});
}

function updateRules(rules) {
  const el = document.getElementById('rulesBody');
  if (!rules || rules.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);font-size:0.8rem;padding:8px">✅ No arrest rules fired — patient stable.</div>';
    return;
  }
  el.innerHTML = rules.map(r =>
    `<div class="rule-chip">⚡ ${r}</div>`
  ).join('');
}

function updateCounterfactuals(cfs) {
  const el = document.getElementById('cfBody');
  if (!cfs || cfs.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);font-size:0.8rem;padding:8px">✅ No interventions needed.</div>';
    document.getElementById('cprBanner').style.display='none';
    return;
  }
  el.innerHTML = cfs.map(cf => {
    const urgent = cf.feasibility.includes('immediate') || cf.feasibility.includes('SmartCPR');
    const arrow = cf.direction === 'increase' ? '↑' : '↓';
    return `<div class="cf-item ${urgent?'urgent':''}">
      <div class="cf-feature ${urgent?'urgent':''}">${arrow} ${cf.feature.replace(/_/g,' ').toUpperCase()}: ${cf.current_value} → ${cf.target_value}</div>
      <div class="cf-action">${cf.clinical_action}</div>
      <div class="cf-timeline">⏱ ${cf.feasibility}</div>
    </div>`;
  }).join('');
}

function updateStatus(label, confidence) {
  const badge = document.getElementById('statusBadge');
  const text  = document.getElementById('statusText');
  const conf = (confidence*100).toFixed(0);
  
  if (label === 'cardiac_arrest') {
    badge.className = 'status-badge alert';
    text.textContent = '🚨 CARDIAC ARREST — ' + conf + '%';
  } else if (label === 'heart_failure') {
    badge.className = 'status-badge warning';
    text.textContent = '🫀 HEART FAILURE — ' + conf + '%';
  } else if (label === 'tachycardia') {
    badge.className = 'status-badge warning';
    text.textContent = '🏃 TACHY — ' + conf + '%';
  } else if (label === 'pre_arrest_alert') {
    badge.className = 'status-badge alert';
    text.textContent = '⚠️ PRE-ARREST — ' + conf + '%';
  } else {
    badge.className = 'status-badge active';
    text.textContent = 'Monitoring Active';
    document.getElementById('cprBanner').style.display='none';
  }
}

function setScenario(s) {
  fetch('/api/scenario', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({scenario: s})
  });
  const labels = {
    normal:'✅ Normal sinus rhythm',vfib:'⚡ VFib cardiac arrest',
    asystole:'📉 Asystole (flatline)',bradycardia:'🐌 Severe bradycardia',
    pre_arrest:'⚠️ Pre-arrest state', heart_failure:'🫀 Congestive Heart Failure',
    tachycardia:'🏃 Supraventricular Tachycardia'
  };
  addLog('info', 'Scenario changed: ' + (labels[s]||s));
}

async function showSettings() {
  const resp = await fetch('/api/settings');
  scenariosConfig = await resp.json();
  const form = document.getElementById('settingsForm');
  
  // Sort scenarios to keep Normal at top, others alphabetical
  const keys = Object.keys(scenariosConfig).sort((a,b) => {
    if (a === 'normal') return -1;
    if (b === 'normal') return 1;
    return a.localeCompare(b);
  });

  form.innerHTML = keys.map(s => {
    const cfg = scenariosConfig[s];
    const scenarioTitle = s.replace(/_/g, ' ').toUpperCase();
    return `<div style="background:#1a1d29;padding:18px;border-radius:14px;border:1px solid #2d3244;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1)">
      <h4 style="margin-bottom:12px;color:var(--accent-purple);font-size:0.9rem">${scenarioTitle}</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        ${Object.keys(cfg).map(k => `
          <div>
            <label style="display:block;font-size:0.7rem;color:var(--muted);margin-bottom:4px">${k.toUpperCase()}</label>
            <input type="number" id="set_${s}_${k}" value="${cfg[k]}" step="0.5"
              style="width:100%;background:#0a0c14;border:1px solid #334155;color:white;padding:8px;border-radius:6px;font-size:0.85rem">
          </div>
        `).join('')}
      </div>
    </div>`;
  }).join('');
  document.getElementById('settingsModal').style.display='flex';
}

function closeSettings() { document.getElementById('settingsModal').style.display='none'; }

async function saveSettings() {
  for (let s in scenariosConfig) {
    for (let k in scenariosConfig[s]) {
      scenariosConfig[s][k] = parseFloat(document.getElementById(`set_${s}_${k}`).value);
    }
  }
  await fetch('/api/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(scenariosConfig)
  });
  addLog('info', 'Simulation settings updated successfully');
  closeSettings();
}

function showRules() {
  fetch('/fold_rules').then(r=>r.json()).then(data => {
    document.getElementById('rulesModalContent').textContent = data.rules.join('\\n\\n');
    document.getElementById('rulesModal').style.display='flex';
  });
}

function addLog(type, msg) {
  const el = document.getElementById('logBody');
  const ts = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = 'log-entry ' + type;
  div.textContent = `[${ts}] ${msg}`;
  el.insertBefore(div, el.firstChild);
  if (el.children.length > 50) el.removeChild(el.lastChild);
}

function setText(id, val) { document.getElementById(id).textContent = val; }
function colorVital(id, color) {
  const el = document.getElementById(id);
  el.className = 'vital-value ' + color;
}
function setBar(id, pct, color) {
  const el = document.getElementById(id);
  el.style.width = Math.max(1, Math.min(100, pct)) + '%';
  el.style.background = color;
}

// Init chart
Plotly.newPlot('ecgChart', [{x:[],y:[],type:'scatter',mode:'lines',
  line:{color:'#10b981',width:1.5}}],
  {paper_bgcolor:'transparent',plot_bgcolor:'transparent',
   margin:{t:10,b:30,l:40,r:10},showlegend:false,
   xaxis:{color:'#6b7280',gridcolor:'#1f2937'},
   yaxis:{color:'#6b7280',gridcolor:'#1f2937'}},
  {responsive:true,displayModeBar:false});

addLog('success', '🫀 SmartCPR Guardian initialized');
addLog('info', 'FOLD-RM classifier loaded — 5 cardiac arrest rules active');
addLog('info', 'MC3G counterfactual engine ready');
addLog('info', 'Emergency dispatch system connected');
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global SIMULATION_SETTINGS
    if request.method == 'POST':
        SIMULATION_SETTINGS = request.json
        return jsonify({"status": "success"})
    return jsonify(SIMULATION_SETTINGS)


@app.route('/api/scenario', methods=['POST'])
def set_scenario():
    global current_scenario
    data = request.json
    current_scenario = data.get("scenario", "normal")
    return jsonify({"status": "ok", "scenario": current_scenario})


@app.route("/fold_rules")
def fold_rules():
    rules = [str(r) for r in classifier.rules]
    asp = [r.as_asp() for r in classifier.rules]
    return jsonify({"rules": asp})


def monitoring_loop():
    """Background loop: simulate vitals, run AI, emit to dashboard."""
    global current_scenario

    while True:
        try:
            # Generate vitals based on scenario
            sim = PatientVitalsSimulator
            # Get targets from settings
            st = SIMULATION_SETTINGS.get(current_scenario, {})
            
            scenarios = {
                "normal":       lambda: sim.normal(**st),
                "vfib":         lambda: sim.cardiac_arrest_vfib(**st),
                "asystole":     lambda: sim.cardiac_arrest_asystole(**st),
                "bradycardia":  lambda: sim.pre_arrest_bradycardia(**st),
                "pre_arrest":   lambda: sim.pre_arrest_bradycardia(**st),
                "heart_failure": lambda: sim.heart_failure(**st),
                "tachycardia":  lambda: sim.tachycardia(**st),
            }
            ecg, sensor_data = scenarios.get(current_scenario, scenarios["normal"])()

            # Extract features
            features = extractor.extract(
                ecg,
                spo2=sensor_data["spo2"],
                systolic_bp=sensor_data["systolic_bp"],
                motion_accel=sensor_data["motion_accel"]
            )

            # Run FOLD-RM classifier
            result = classifier.predict(features)

            # Generate counterfactuals if arrest detected
            cfs = []
            if result["label"] != "normal":
                cfs_obj = cf_generator.generate(
                    features, result["fired_rules"], result["label"])
                cfs = [
                    {
                        "feature": c.feature,
                        "current_value": c.current_value,
                        "target_value": c.target_value,
                        "direction": c.direction,
                        "clinical_action": c.clinical_action,
                        "feasibility": c.feasibility,
                    }
                    for c in cfs_obj
                ]

            # Emit to dashboard
            payload = {
                "vitals": {k: round(float(v), 2) for k, v in features.items()},
                "label": result["label"],
                "confidence": result["confidence"],
                "fired_rules": result["fired_rules"],
                "counterfactuals": cfs,
                "ecg_sample": ecg[:72].tolist(),  # 0.2s chunk @ 360Hz
            }
            socketio.emit("vitals_update", payload)

            # If arrest: emit emergency alert
            if result["label"] in ("cardiac_arrest",) and result["confidence"] > 0.85:
                socketio.emit("emergency_alert", {
                    "confidence": result["confidence"],
                    "eta": 7.5,
                    "hospital": "UT Southwestern Medical Center"
                })

        except Exception as e:
            print(f"Monitoring error: {e}")

        time.sleep(0.2)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting SmartCPR Guardian on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
