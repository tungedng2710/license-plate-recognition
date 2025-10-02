document.addEventListener('DOMContentLoaded', () => {
  const streamInput = document.getElementById('stream-url');
  const cameraSelect = document.getElementById('camera-select');
  const readPlateToggle = document.getElementById('read-plate');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const vconf = document.getElementById('vconf');
  const pconf = document.getElementById('pconf');
  const vconfVal = document.getElementById('vconf-val');
  const pconfVal = document.getElementById('pconf-val');
  const streamImg = document.getElementById('alpr-stream');
  const placeholder = document.getElementById('stream-placeholder');
  const placeholderText = placeholder ? placeholder.querySelector('p') : null;
  const CUSTOM_OPTION_VALUE = '__custom';

  if (!streamInput || !cameraSelect || !startBtn || !pauseBtn || !stopBtn || !streamImg || !readPlateToggle) {
    return;
  }

  let currentSrc = '';
  let paused = false;

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

    const params = new URLSearchParams({ url });
    if (vconf) {
      params.set('vconf', String(vconf.value));
    }
    if (pconf) {
      params.set('pconf', String(pconf.value));
    }
    if (readPlateToggle) {
      params.set('read_plate', String(readPlateToggle.checked));
    }

    const src = `/api/alpr_stream?${params.toString()}`;
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
    if (!currentSrc) {
      showPlaceholder('No stream running');
      return;
    }
    streamImg.removeAttribute('src');
    currentSrc = '';
    paused = false;
    pauseBtn.textContent = 'Pause';
    showPlaceholder('No stream running');
  });

  streamImg.addEventListener('error', () => {
    if (!currentSrc) {
      return;
    }
    showPlaceholder('Unable to load stream. Check the URL and try again.');
    streamImg.removeAttribute('src');
    currentSrc = '';
    paused = false;
    pauseBtn.textContent = 'Pause';
  });

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
