document.addEventListener('DOMContentLoaded', async () => {
  let cameras = {};
  try {
    const res = await fetch('/api/rtsp_urls');
    cameras = await res.json();
  } catch (err) {
    console.error('Failed to load camera list', err);
  }

  const slots = document.querySelectorAll('.stream-slot');
  slots.forEach(slot => {
    const select = slot.querySelector('.camera-select');
    const customInput = slot.querySelector('.custom-url');
    const startBtn = slot.querySelector('.start-stream');
    const stopBtn = slot.querySelector('.stop-stream');
    const videoEl = slot.querySelector('.video-stream');
    let pc = null;

    Object.entries(cameras).forEach(([name, url]) => {
      const opt = document.createElement('option');
      opt.value = url;
      opt.textContent = name;
      select.appendChild(opt);
    });
    const customOpt = document.createElement('option');
    customOpt.value = 'custom';
    customOpt.textContent = 'Custom URL';
    select.appendChild(customOpt);

    select.addEventListener('change', () => {
      if (select.value === 'custom') {
        customInput.classList.remove('hidden');
      } else {
        customInput.classList.add('hidden');
      }
    });

    startBtn.addEventListener('click', async () => {
      let url = select.value;
      if (url === 'custom') {
        url = customInput.value.trim();
      }
      if (!url) return;
      if (pc) {
        pc.close();
        pc = null;
      }
      pc = new RTCPeerConnection();
      pc.ontrack = (e) => {
        videoEl.srcObject = e.streams[0];
      };
      pc.addTransceiver('video', { direction: 'recvonly' });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const res = await fetch(`/api/video/offer?url=${encodeURIComponent(url)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
      });
      const answer = await res.json();
      await pc.setRemoteDescription(answer);
    });

    stopBtn.addEventListener('click', () => {
      if (pc) {
        pc.close();
        pc = null;
      }
      videoEl.srcObject = null;
    });
  });
});
