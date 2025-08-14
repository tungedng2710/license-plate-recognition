document.addEventListener('DOMContentLoaded', async () => {
  const cameraSelect = document.getElementById('camera-select');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const streamImg = document.getElementById('alpr-stream');
  let currentSrc = '';

  // load camera list
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
    currentSrc = `/api/alpr/stream?url=${encodeURIComponent(url)}`;
    streamImg.src = currentSrc;
    pauseBtn.textContent = 'Pause';
  });

  pauseBtn.addEventListener('click', () => {
    if (!streamImg.src) return;
    if (pauseBtn.textContent === 'Pause') {
      streamImg.removeAttribute('src');
      pauseBtn.textContent = 'Resume';
    } else {
      streamImg.src = currentSrc;
      pauseBtn.textContent = 'Pause';
    }
  });

  stopBtn.addEventListener('click', () => {
    streamImg.removeAttribute('src');
    currentSrc = '';
    pauseBtn.textContent = 'Pause';
  });
});
