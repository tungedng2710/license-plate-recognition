document.addEventListener('DOMContentLoaded', async () => {
  const cameraSelect = document.getElementById('camera-select');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const streamVideo = document.getElementById('alpr-stream');

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
    streamVideo.src = `/api/alpr/stream?url=${encodeURIComponent(url)}`;
    streamVideo.play();
    pauseBtn.textContent = 'Pause';
  });

  pauseBtn.addEventListener('click', () => {
    if (!streamVideo.src) return;
    if (streamVideo.paused) {
      streamVideo.play();
      pauseBtn.textContent = 'Pause';
    } else {
      streamVideo.pause();
      pauseBtn.textContent = 'Resume';
    }
  });

  stopBtn.addEventListener('click', () => {
    streamVideo.pause();
    streamVideo.removeAttribute('src');
    streamVideo.load();
    pauseBtn.textContent = 'Pause';
  });
});
