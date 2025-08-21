document.addEventListener('DOMContentLoaded', async () => {
  const cameraSelect = document.getElementById('camera-select');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const videoEl = document.getElementById('alpr-stream');
  let pc = null;

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

  startBtn.addEventListener('click', async () => {
    const url = cameraSelect.value;
    if (!url) return;
    if (pc) {
      pc.close();
      pc = null;
    }
    pc = new RTCPeerConnection();
    pc.ontrack = (e) => {
      if (e.streams && e.streams[0]) {
        videoEl.srcObject = e.streams[0];
      } else {
        const stream = new MediaStream();
        stream.addTrack(e.track);
        videoEl.srcObject = stream;
      }
    };
    pc.addTransceiver('video', { direction: 'recvonly' });
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const res = await fetch(`/api/alpr/offer?url=${encodeURIComponent(url)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
    });
    const answer = await res.json();
    await pc.setRemoteDescription(answer);
    pauseBtn.textContent = 'Pause';
  });

  pauseBtn.addEventListener('click', () => {
    if (!videoEl.srcObject) return;
    if (pauseBtn.textContent === 'Pause') {
      videoEl.pause();
      pauseBtn.textContent = 'Resume';
    } else {
      videoEl.play();
      pauseBtn.textContent = 'Pause';
    }
  });

  stopBtn.addEventListener('click', () => {
    if (pc) {
      pc.close();
      pc = null;
    }
    videoEl.srcObject = null;
    pauseBtn.textContent = 'Pause';
  });
});

