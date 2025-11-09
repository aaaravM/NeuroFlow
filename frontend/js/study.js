let mediaStream;
let recorder;
let chunks = [];
let startedAt = null;
let nextRequested = false;
let ws;
let frameTimer;
let lastAnalysis;

const API_BASE = (window.NF && NF.loadApiBase()) || (window.NF_API_BASE || (location.protocol === 'file:' ? 'http://localhost:8000' : ''));
const WS_BASE = (window.NF && NF.wsBase()) || (API_BASE ? API_BASE.replace(/^http/, 'ws') : '');

async function setupCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const video = document.getElementById('preview');
    video.srcObject = mediaStream;
    if (!API_BASE && location.protocol !== 'file:') {
      console.warn('No backend configured. Use the Backend button to set NF_API_BASE.');
    }
  } catch (err) {
    alert('Camera access is required to start a session.');
    console.error(err);
  }
}

function startRecording() {
  chunks = [];
  if (!mediaStream) return;
  recorder = new MediaRecorder(mediaStream, { mimeType: 'video/webm;codecs=vp9' });
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  recorder.onstop = () => {
    const durationSec = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    // Use last live analysis if available; fallback to conservative defaults
    const focus = Math.min(100, Math.max(0, Math.round((lastAnalysis?.focus ?? 0.62) * 100)));
    const stress = Math.min(100, Math.max(0, Math.round((lastAnalysis?.stress ?? 0.41) * 100)));
    const session = { ts: Date.now(), durationSec, focus, stress };
    sessionStorage.setItem('nf_lastSession', JSON.stringify(session));

    // Save recorded video as data URL for editor page
    try {
      const blob = new Blob(chunks, { type: 'video/webm' });
      const reader = new FileReader();
      reader.onloadend = () => {
        try {
          sessionStorage.setItem('nf_videoData', reader.result);
        } catch (e) {
          console.warn('Could not store video in sessionStorage (size limit).');
        }
        finalizeStop();
      };
      reader.readAsDataURL(blob);
    } catch (e) {
      console.error('Failed to prepare video blob', e);
      finalizeStop();
    }
  };
  recorder.start();
  startedAt = Date.now();

  // Start analysis WS and frame loop
  openWebSocket();
  startFrameLoop();
}

function stopRecording() {
  if (recorder && recorder.state !== 'inactive') {
    recorder.stop();
  }
  stopFrameLoop();
  closeWebSocket();
}

function finalizeStop() {
  // Free tracks to stop camera light
  try { mediaStream.getTracks().forEach((t) => t.stop()); } catch {}
  const btn = document.getElementById('recordBtn');
  const nextBtn = document.getElementById('nextBtn');
  if (btn) { btn.textContent = 'Recording Stopped'; btn.disabled = true; }
  if (nextBtn) { nextBtn.disabled = false; }
  try { if (window.NF) window.NF.confetti(1500); } catch {}
  if (nextRequested) {
    location.href = 'editor.html';
  }
}

function openWebSocket() {
  try {
    const url = `${WS_BASE}/ws/neuroflow`;
    ws = new WebSocket(url);
    ws.onopen = () => {
      if (window.NF) NF.setConnStatus('Online', 'ok');
    };
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'analysis' && msg.data) {
          lastAnalysis = {
            focus: msg.data.focus ?? lastAnalysis?.focus,
            stress: msg.data.stress ?? lastAnalysis?.stress,
            timestamp: Date.now()
          };
        }
      } catch {}
    };
    ws.onerror = () => { if (window.NF) NF.setConnStatus('Error', 'err'); };
    ws.onclose = () => { if (window.NF) NF.setConnStatus('Offline', 'err'); };
  } catch (e) {
    console.warn('WebSocket unavailable; live analysis disabled.', e);
    if (window.NF) NF.setConnStatus('Unavailable', 'warn');
  }
}

function closeWebSocket() {
  try { if (ws && ws.readyState <= 1) ws.close(); } catch {}
  ws = null;
}

function startFrameLoop() {
  if (frameTimer) return;
  const video = document.getElementById('preview');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const FPS = 1; // 1 fps to keep things light
  frameTimer = setInterval(() => {
    try {
      if (!ws || ws.readyState !== 1) return;
      if (!video || video.readyState < 2) return;
      // Slightly higher resolution + quality to improve face detection
      canvas.width = 640; canvas.height = 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      ws.send(JSON.stringify({ type: 'frame', image: dataUrl, keystroke: false, cursor: [0, 0] }));
    } catch {}
  }, 1000 / FPS);
}

function stopFrameLoop() {
  if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
}

document.addEventListener('DOMContentLoaded', async () => {
  await setupCamera();
  const btn = document.getElementById('recordBtn');
  const nextBtn = document.getElementById('nextBtn');
  let recording = false;

  btn.addEventListener('click', () => {
    if (!recording) {
      recording = true;
      btn.textContent = 'Stop recording';
      startRecording();
    } else {
      btn.disabled = true;
      btn.textContent = 'Finishing…';
      stopRecording();
    }
  });

  nextBtn.addEventListener('click', () => {
    const hasVideo = !!sessionStorage.getItem('nf_videoData');
    if (recorder && recorder.state !== 'inactive') {
      nextRequested = true;
      btn.disabled = true;
      btn.textContent = 'Finishing…';
      stopRecording();
      return;
    }
    if (!hasVideo) {
      alert('Please stop recording before proceeding to the editor.');
      return;
    }
    location.href = 'editor.html';
  });
});
