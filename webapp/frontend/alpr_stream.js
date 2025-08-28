document.addEventListener('DOMContentLoaded', async () => {
  const cameraSelect = document.getElementById('camera-select');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const imgEl = document.getElementById('alpr-stream');
  let currentSrc = '';

  try {
    const res = await fetch('/api/rtsp_urls');
    const cams = await res.json();
    Object.entries(cams).forEach(([name, url]) => {
      const opt = document.createElement('option');
      opt.value = url;
      opt.textContent = name;
      cameraSelect.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to load camera list', err);
  }

  startBtn.addEventListener('click', () => {
    const url = cameraSelect.value;
    if (!url) return;
    currentSrc = `/api/alpr_stream?url=${encodeURIComponent(url)}`;
    imgEl.src = currentSrc;
    pauseBtn.textContent = 'Pause';
  });

  pauseBtn.addEventListener('click', () => {
    if (!currentSrc) return;
    if (pauseBtn.textContent === 'Pause') {
      imgEl.src = '';
      pauseBtn.textContent = 'Resume';
    } else {
      imgEl.src = currentSrc;
      pauseBtn.textContent = 'Pause';
    }
  });

  stopBtn.addEventListener('click', () => {
    imgEl.src = '';
    currentSrc = '';
    pauseBtn.textContent = 'Pause';
  });
});
