function fmtSeconds(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s}s`;
}

document.addEventListener('DOMContentLoaded', () => {
  const raw = sessionStorage.getItem('nf_lastSession');
  let session;
  try { session = raw ? JSON.parse(raw) : null; } catch {}

  const summary = document.getElementById('summary');
  const fMeter = document.getElementById('focusMeter');
  const sMeter = document.getElementById('stressMeter');
  const fText = document.getElementById('focusText');
  const sText = document.getElementById('stressText');
  const aiBox = document.getElementById('aiRecommendations');
  const chatBox = document.getElementById('chatMessages');
  const chatInput = document.getElementById('chatQuestion');
  const chatSend = document.getElementById('chatSend');

  const API_BASE = (window.NF && NF.loadApiBase()) || (window.NF_API_BASE || (location.protocol === 'file:' ? 'http://localhost:8000' : ''));
  const WS_BASE = (window.NF && NF.wsBase()) || (API_BASE ? API_BASE.replace(/^http/, 'ws') : '');

  function setMeters(focusPct, stressPct) {
    const f = Math.max(0, Math.min(100, Math.round(focusPct)));
    const s = Math.max(0, Math.min(100, Math.round(stressPct)));
    fMeter.style.width = `${f}%`;
    sMeter.style.width = `${s}%`;
    fText.textContent = f >= 70 ? 'Strong sustained focus' : f >= 50 ? 'Moderate focus' : 'Low focus';
    sText.textContent = s >= 70 ? 'High stress indicators' : s >= 45 ? 'Moderate stress indicators' : 'Low stress indicators';
  }

  function asPct(val, fallback) {
    if (val == null || isNaN(val)) return fallback;
    return val <= 1 ? val * 100 : val; // normalize 0..1 to % if needed
  }

  if (!session) {
    summary.textContent = 'No recent session found. Start a new session to see results.';
    setMeters(0, 0);
    // Try to fetch state from backend as a fallback
    try {
      fetch(`${API_BASE}/api/state`)
        .then((r) => r.ok ? r.json() : null)
        .then((st) => {
          if (!st) return;
          const f = asPct(st.focus ?? 0.5, 50);
          const s = asPct(st.stress ?? 0.0, 20);
          setMeters(f, s);
        })
        .catch(() => {});
    } catch {}
  } else {
    summary.textContent = `Duration: ${fmtSeconds(session.durationSec)} — Captured via on-device webcam`;
    const focus = asPct(session.focus ?? 62, 62);
    const stress = asPct(session.stress ?? 41, 41);
    setMeters(focus, stress);
  }

  // Fetch recommendations from backend (RAG + optional HF LLM)
  if (aiBox) {
    if (!API_BASE && location.protocol !== 'file:') {
      aiBox.innerHTML = '<div class="hint">Configure backend URL (top-right) to enable recommendations.</div>';
      return;
    }
    aiBox.innerHTML = '<div class="skeleton block" style="width:70%"></div><div class="skeleton block" style="width:95%"></div><div class="skeleton block" style="width:88%"></div>';
    try {
      fetch(`${API_BASE}/api/recommendations`)
        .then((r) => r.ok ? r.json() : Promise.reject(new Error('failed')))
        .then((data) => {
          const recs = data && data.recommendations ? data.recommendations : [];
          if (!recs.length) {
            aiBox.innerHTML = '<div class="hint">No recommendations available right now.</div>';
            return;
          }
          const top = recs[0];
          const txt = (top.text || '');
          // typewriter reveal
          const wrap = document.createElement('div');
          wrap.className = 'ai-reco';
          aiBox.innerHTML = '';
          aiBox.appendChild(wrap);
          let i = 0;
          const safe = txt.replace(/</g, '&lt;').replace(/>/g, '&gt;');
          function tick(){
            wrap.innerHTML = safe.slice(0, i).replace(/\n/g, '<br/>');
            i += Math.max(1, Math.floor(Math.random()*3));
            if (i <= safe.length) requestAnimationFrame(tick);
          }
          requestAnimationFrame(tick);
        })
        .catch(() => {
          aiBox.innerHTML = '<div class="hint">Unable to load recommendations. Please try again later.</div>';
        });
    } catch {
      aiBox.innerHTML = '<div class="hint">Unable to load recommendations. Please try again later.</div>';
    }
  }

  // Chat wiring
  function appendMsg(sender, text) {
    if (!chatBox) return;
    const item = document.createElement('div');
    item.className = 'chat-msg';
    const who = sender === 'user' ? 'You' : 'Coach';
    const safe = String(text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    item.innerHTML = `<strong>${who}:</strong> ${safe}`;
    chatBox.appendChild(item);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  let typingEl = null;
  function showTyping() {
    if (!chatBox) return;
    typingEl = document.createElement('div');
    typingEl.className = 'chat-msg';
    typingEl.innerHTML = `<strong>Coach:</strong> <span class="typing-dots"><span></span><span></span><span></span></span>`;
    chatBox.appendChild(typingEl);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  function hideTyping() {
    if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
    typingEl = null;
  }

  async function sendQuestion() {
    if (!chatInput) return;
    const q = chatInput.value.trim();
    if (!q) return;
    appendMsg('user', q);
    chatInput.value = '';
    try {
      if (chatSend) chatSend.disabled = true;
      showTyping();
      const resp = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      if (!resp.ok) throw new Error('request failed');
      const data = await resp.json();
      appendMsg('bot', data.answer || 'Sorry, no answer available.');
    } catch (e) {
      appendMsg('bot', 'Sorry, I could not answer right now.');
    } finally {
      hideTyping();
      if (chatSend) chatSend.disabled = false;
    }
  }

  if (chatSend) {
    chatSend.addEventListener('click', sendQuestion);
  }
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        sendQuestion();
      }
    });
  }

  // Live metrics via WebSocket (if backend reachable)
  try {
    const ws = new WebSocket(`${WS_BASE}/ws/neuroflow`);
    ws.onopen = () => {
      // Kick a keep-alive ping occasionally
      setInterval(() => { try { ws.send(JSON.stringify({ type: 'ping' })); } catch {} }, 25000);
      if (window.NF) NF.setConnStatus('Online', 'ok');
    };
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'analysis' && msg.data) {
          const prev = parseInt((fMeter.style.width || '0').replace('%',''));
          const f = Math.round(((msg.data.focus ?? 0.5) <= 1 ? (msg.data.focus || 0.5) * 100 : msg.data.focus));
          const s = Math.round(((msg.data.stress ?? 0.0) <= 1 ? (msg.data.stress || 0.0) * 100 : msg.data.stress));
          fMeter.style.width = `${f}%`;
          sMeter.style.width = `${s}%`;
          fText.textContent = f >= 70 ? 'Strong sustained focus' : f >= 50 ? 'Moderate focus' : 'Low focus';
          sText.textContent = s >= 70 ? 'High stress indicators' : s >= 45 ? 'Moderate stress indicators' : 'Low stress indicators';
          if (!Number.isNaN(prev) && f - prev >= 10) triggerConfetti();
        }
      } catch {}
    };
    ws.onerror = () => { if (window.NF) NF.setConnStatus('Error', 'err'); };
    ws.onclose = () => { if (window.NF) NF.setConnStatus('Offline', 'err'); };
  } catch {}
  function triggerConfetti() {
    const canvas = document.createElement('canvas');
    canvas.className = 'confetti-canvas';
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = window.innerWidth);
    const h = (canvas.height = window.innerHeight);
    const N = 120;
    const pieces = Array.from({ length: N }, () => ({
      x: Math.random() * w,
      y: -20 - Math.random() * 60,
      vx: -2 + Math.random() * 4,
      vy: 2 + Math.random() * 4,
      size: 4 + Math.random() * 6,
      rot: Math.random() * Math.PI,
      vr: -0.2 + Math.random() * 0.4,
      color: Math.random() < 0.5 ? '#6c7bff' : (Math.random() < 0.5 ? '#6cf5ff' : '#3ee2a8')
    }));
    let t = 0;
    function step() {
      t += 16;
      ctx.clearRect(0, 0, w, h);
      pieces.forEach(p => {
        p.x += p.vx; p.y += p.vy; p.vy += 0.02; p.rot += p.vr;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size/2, -p.size/2, p.size, p.size);
        ctx.restore();
      });
      if (t < 1200) requestAnimationFrame(step); else document.body.removeChild(canvas);
    }
    requestAnimationFrame(step);
  }
});
