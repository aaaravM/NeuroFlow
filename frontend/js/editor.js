document.addEventListener('DOMContentLoaded', () => {
  const videoEl = document.getElementById('recording');
  const dataUrl = sessionStorage.getItem('nf_videoData');

  if (!dataUrl) {
    videoEl.replaceWith(Object.assign(document.createElement('div'), { textContent: 'No recording available. Please re‑record your session.', className: 'hint' }));
  } else {
    videoEl.src = dataUrl;
  }

  const goToRecord = () => {
    sessionStorage.removeItem('nf_videoData');
    location.href = 'study.html';
  };

  document.getElementById('discardBtn').addEventListener('click', () => {
    sessionStorage.removeItem('nf_lastSession');
    goToRecord();
  });

  document.getElementById('rerecordBtn').addEventListener('click', () => {
    goToRecord();
  });

  document.getElementById('analyzeBtn').addEventListener('click', () => {
    // Proceed to results page
    location.href = 'results.html';
  });
});
