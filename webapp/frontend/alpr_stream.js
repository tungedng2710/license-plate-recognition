document.addEventListener('DOMContentLoaded', () => {
  const streamInput = document.getElementById('stream-url');
  const cameraSelect = document.getElementById('camera-select');
  const modeSelect = document.getElementById('stream-mode');
  const readPlateToggle = document.getElementById('read-plate');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const vconf = document.getElementById('vconf');
  const pconf = document.getElementById('pconf');
  const vconfVal = document.getElementById('vconf-val');
  const pconfVal = document.getElementById('pconf-val');
  const streamImg = document.getElementById('alpr-stream');
  const controlPanel = document.querySelector('.control-panel');
  const placeholder = document.getElementById('stream-placeholder');
  const placeholderText = placeholder ? placeholder.querySelector('p') : null;
  const CUSTOM_OPTION_VALUE = '__custom';

  if (!streamInput || !cameraSelect || !startBtn || !pauseBtn || !stopBtn || !streamImg || !readPlateToggle || !modeSelect) {
    return;
  }

  let currentSrc = '';
  let paused = false;

  function getSelectedMode() {
    return modeSelect.value || 'alpr';
  }

  function applyModeState() {
    const preview = getSelectedMode() === 'preview';
    if (controlPanel) {
      controlPanel.classList.toggle('preview-mode', preview);
    }
    if (vconf) {
      vconf.disabled = preview;
    }
    if (pconf) {
      pconf.disabled = preview;
    }
    if (readPlateToggle) {
      readPlateToggle.disabled = preview;
    }
  }

  function stopStream(message = 'No stream running') {
    if (streamImg) {
      streamImg.removeAttribute('src');
    }
    currentSrc = '';
    paused = false;
    pauseBtn.textContent = 'Pause';
    showPlaceholder(message);
  }

  function showPlaceholder(message) {
    if (!placeholder) {
      return;
    }
    if (placeholderText && message) {
      placeholderText.textContent = message;
    }
    placeholder.style.display = 'flex';
    streamImg.style.display = 'none';
  }

  function showStream(src) {
    currentSrc = src;
    streamImg.src = src;
    streamImg.style.display = 'block';
    if (placeholder) {
      placeholder.style.display = 'none';
    }
  }

  if (vconf) {
    vconf.addEventListener('input', () => {
      if (vconfVal) {
        vconfVal.textContent = Number(vconf.value).toFixed(2);
      }
    });
  }

  if (pconf) {
    pconf.addEventListener('input', () => {
      if (pconfVal) {
        pconfVal.textContent = Number(pconf.value).toFixed(2);
      }
    });
  }

  if (streamInput) {
    streamInput.addEventListener('input', () => {
      streamInput.classList.remove('input-error');
      if (cameraSelect.value !== CUSTOM_OPTION_VALUE) {
        cameraSelect.value = CUSTOM_OPTION_VALUE;
      }
    });
  }

  cameraSelect.addEventListener('change', () => {
    if (cameraSelect.value === CUSTOM_OPTION_VALUE) {
      return;
    }
    streamInput.value = cameraSelect.value;
    streamInput.classList.remove('input-error');
  });

  startBtn.addEventListener('click', () => {
    const url = streamInput.value.trim();
    if (!url) {
      streamInput.classList.add('input-error');
      streamInput.focus();
      showPlaceholder('Provide a stream URL to begin');
      return;
    }

    const mode = getSelectedMode();
    const preview = mode === 'preview';
    const params = new URLSearchParams({ url });
    if (!preview && vconf) {
      params.set('vconf', String(vconf.value));
    }
    if (!preview && pconf) {
      params.set('pconf', String(pconf.value));
    }
    if (!preview && readPlateToggle) {
      params.set('read_plate', String(readPlateToggle.checked));
    }

    const endpoint = preview ? '/api/video' : '/api/alpr_stream';
    const src = `${endpoint}?${params.toString()}`;
    paused = false;
    pauseBtn.textContent = 'Pause';
    showStream(src);
  });

  pauseBtn.addEventListener('click', () => {
    if (!currentSrc) {
      return;
    }
    if (!paused) {
      streamImg.removeAttribute('src');
      showPlaceholder('Stream paused');
      pauseBtn.textContent = 'Resume';
      paused = true;
    } else {
      showStream(currentSrc);
      pauseBtn.textContent = 'Pause';
      paused = false;
    }
  });

  stopBtn.addEventListener('click', () => {
    const preview = getSelectedMode() === 'preview';
    const message = preview
      ? 'Camera preview stopped. Start to view the camera feed.'
      : 'No stream running';
    stopStream(message);
  });

  streamImg.addEventListener('error', () => {
    if (!currentSrc) {
      return;
    }
    stopStream('Unable to load stream. Check the URL and try again.');
  });

  if (modeSelect) {
    modeSelect.addEventListener('change', () => {
      applyModeState();
      const preview = getSelectedMode() === 'preview';
      const message = preview
        ? 'Preview mode selected. Start to view the camera feed.'
        : 'ALPR mode selected. Start to begin detection.';
      stopStream(message);
    });
  }

  applyModeState();
  showPlaceholder('No stream running');

  async function loadCameraPresets() {
    cameraSelect.disabled = true;
    try {
      const response = await fetch('/api/cameras');
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const payload = await response.json();
      const presets = Array.isArray(payload?.presets) ? payload.presets : [];
      if (!presets.length) {
        return;
      }

      const fragment = document.createDocumentFragment();
      presets.forEach((preset) => {
        if (!preset || typeof preset.url !== 'string' || typeof preset.label !== 'string') {
          return;
        }
        const option = document.createElement('option');
        option.value = preset.url;
        option.textContent = preset.label;
        fragment.appendChild(option);
      });

      cameraSelect.appendChild(fragment);
    } catch (error) {
      console.error('Failed to load camera presets', error); // eslint-disable-line no-console
    } finally {
      cameraSelect.disabled = false;
    }
  }

  loadCameraPresets();
});
