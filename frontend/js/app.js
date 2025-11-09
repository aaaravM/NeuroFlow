// Global UI utilities and settings for NeuroFlow
// - Settings modal for Backend URL
// - Connection status pill helpers
// - Lightweight animated background particles

window.NF = (function () {
  const state = {
    apiBase: undefined,
  };

  function loadApiBase() {
    if (state.apiBase) return state.apiBase;
    try {
      const saved = localStorage.getItem('NF_API_BASE');
      if (saved) state.apiBase = saved;
    } catch {}
    if (!state.apiBase && location.protocol === 'file:') {
      state.apiBase = 'http://localhost:8000';
    }
    return state.apiBase || '';
  }

  function setApiBase(val) {
    state.apiBase = val || '';
    try { localStorage.setItem('NF_API_BASE', state.apiBase); } catch {}
  }

  function wsBase() {
    const base = loadApiBase();
    if (!base) return '';
    return base.replace(/^http/, 'ws');
  }

  function buildSettingsModal() {
    const modal = document.createElement('div');
    modal.className = 'modal hidden';
    modal.innerHTML = `
      <div class="backdrop" data-close></div>
      <div class="modal-card">
        <h3>Settings</h3>
        <label>
          <span>Backend API Base URL</span>
          <input type="text" id="nfApiBase" placeholder="https://your-backend.example.com" />
        </label>
        <div class="modal-actions">
          <button class="btn btn-ghost" id="nfCancel">Cancel</button>
          <button class="btn btn-primary" id="nfSave">Save</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    function close() { modal.classList.add('hidden'); }
    function open() {
      const input = modal.querySelector('#nfApiBase');
      input.value = loadApiBase();
      modal.classList.remove('hidden');
      setTimeout(() => input.focus(), 50);
    }

    modal.querySelector('[data-close]').addEventListener('click', close);
    modal.querySelector('#nfCancel').addEventListener('click', close);
    modal.querySelector('#nfSave').addEventListener('click', () => {
      const val = modal.querySelector('#nfApiBase').value.trim();
      setApiBase(val);
      close();
      location.reload();
    });

    const openBtn = document.getElementById('openSettings');
    if (openBtn) openBtn.addEventListener('click', open);
  }

  function setConnStatus(text, status) {
    const ids = ['connStatus', 'connStatusLocal', 'connStatusResults'];
    ids.forEach((id) => {
      const pill = document.getElementById(id);
      if (!pill) return;
      pill.textContent = text;
      pill.classList.remove('ok', 'warn', 'err');
      if (status) pill.classList.add(status);
    });
  }

  function bgParticles() {
    const canvas = document.createElement('canvas');
    canvas.id = 'bgCanvas';
    canvas.className = 'bg-canvas';
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    let w, h, particles;
    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
      particles = Array.from({ length: Math.max(60, Math.min(140, Math.floor(w * h / 18000))) }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: 0.6 + Math.random() * 1.8,
      }));
    }
    window.addEventListener('resize', resize);
    resize();
    function step() {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(108,123,255,0.2)';
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = w; if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h; if (p.y > h) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      });
      requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  document.addEventListener('DOMContentLoaded', () => {
    // Attach gear if present and build modal
    buildSettingsModal();
    // Add background particles once per page
    bgParticles();

    // Auto-discover backend if none configured
    (async function autoDiscover() {
      if (state.apiBase) return;
      try { const saved = localStorage.getItem('NF_API_BASE'); if (saved) { state.apiBase = saved; return; } } catch {}

      const candidates = [];
      if (location.protocol === 'http:' || location.protocol === 'https:') {
        candidates.push(location.origin);
      }
      // Common local dev ports
      candidates.push('http://localhost:8000');
      candidates.push('https://localhost:8000');
      candidates.push('http://127.0.0.1:8000');

      const ok = await (async function tryList() {
        for (const base of candidates) {
          try {
            const ctl = new AbortController();
            const id = setTimeout(() => ctl.abort(), 2000);
            const r = await fetch(base + '/', { signal: ctl.signal });
            clearTimeout(id);
            if (!r.ok) continue;
            const j = await r.json().catch(() => null);
            if (j && String(j.service || '').includes('NeuroFlow')) {
              setApiBase(base);
              setConnStatus('Online', 'ok');
              // Force a reload so all pages pick up the new API base
              setTimeout(() => location.reload(), 300);
              return true;
            }
          } catch { /* ignore */ }
        }
        return false;
      })();

      if (!ok) {
        setConnStatus('Not set', 'warn');
      }
    })();
  });

  function confetti(durationMs = 1200) {
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
      if (t < durationMs) requestAnimationFrame(step); else document.body.removeChild(canvas);
    }
    requestAnimationFrame(step);
  }

  // Subtle parallax for hero blob
  document.addEventListener('mousemove', (e) => {
    const blob = document.querySelector('.hero .blob');
    if (!blob) return;
    const x = (e.clientX / window.innerWidth - 0.5) * 8;
    const y = (e.clientY / window.innerHeight - 0.5) * 8;
    blob.style.transform = `translate(${x}px, ${y}px)`;
  });

  return { loadApiBase, setApiBase, wsBase, setConnStatus, confetti };
})();
