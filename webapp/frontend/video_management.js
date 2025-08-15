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
    const img = slot.querySelector('.video-stream');

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

    startBtn.addEventListener('click', () => {
      let url = select.value;
      if (url === 'custom') {
        url = customInput.value.trim();
      }
      if (!url) return;
      img.src = `/api/video/stream?url=${encodeURIComponent(url)}`;
    });

    stopBtn.addEventListener('click', () => {
      img.removeAttribute('src');
    });
  });
});
