document.addEventListener('DOMContentLoaded', async () => {
  const cameraSelect = document.getElementById('camera-select');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const vehicleModel = document.getElementById('vehicle-model');
  const plateModel = document.getElementById('plate-model');
  const useCustom = document.getElementById('use-custom');
  const customUrl = document.getElementById('custom-url');
  const vconf = document.getElementById('vconf');
  const pconf = document.getElementById('pconf');
  const vconfVal = document.getElementById('vconf-val');
  const pconfVal = document.getElementById('pconf-val');
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

  // Toggle custom URL input visibility
  if (useCustom && customUrl) {
    useCustom.addEventListener('change', () => {
      if (useCustom.checked) {
        customUrl.classList.remove('hidden');
      } else {
        customUrl.classList.add('hidden');
      }
    });
  }

  // Live update conf labels
  if (vconf && vconfVal) {
    vconf.addEventListener('input', () => {
      vconfVal.textContent = Number(vconf.value).toFixed(2);
    });
  }
  if (pconf && pconfVal) {
    pconf.addEventListener('input', () => {
      pconfVal.textContent = Number(pconf.value).toFixed(2);
    });
  }

  startBtn.addEventListener('click', () => {
    const url = (useCustom && useCustom.checked && customUrl) ? customUrl.value.trim() : cameraSelect.value;
    if (!url) return;
    const v = vehicleModel ? vehicleModel.value : '';
    const p = plateModel ? plateModel.value : '';
    const qs = new URLSearchParams({ url });
    if (v) qs.set('vehicle_model', v);
    if (p) qs.set('plate_model', p);
    if (vconf) qs.set('vconf', String(vconf.value));
    if (pconf) qs.set('pconf', String(pconf.value));
    currentSrc = `/api/alpr_stream?${qs.toString()}`;
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
