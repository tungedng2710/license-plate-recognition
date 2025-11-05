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
  const webrtcVideo = document.getElementById('webrtc-stream');
  const transportSelect = document.getElementById('stream-transport');
  const vehicleModelSelect = document.getElementById('vehicle-model');
  const controlPanel = document.querySelector('.control-panel');
  const placeholder = document.getElementById('stream-placeholder');
  const placeholderText = placeholder ? placeholder.querySelector('p') : null;
  const CUSTOM_OPTION_VALUE = '__custom';
  const TRANSPORT_MJPEG = 'mjpeg';
  const TRANSPORT_WEBRTC = 'webrtc';

  if (
    !streamInput ||
    !cameraSelect ||
    !startBtn ||
    !pauseBtn ||
    !stopBtn ||
    !streamImg ||
    !webrtcVideo ||
    !transportSelect ||
    !readPlateToggle ||
    !modeSelect ||
    !vehicleModelSelect
  ) {
    return;
  }

  let currentSrc = '';
  let paused = false;
  let currentTransport = TRANSPORT_MJPEG;
  let peerConnection = null;
  let currentStreamNonce = 0;
  let lastStartConfig = null;
  let currentVehicleModel = null;

  function getSelectedTransport() {
    return transportSelect.value || TRANSPORT_MJPEG;
  }

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

  function hideAllStreams() {
    if (streamImg) {
      streamImg.style.display = 'none';
      streamImg.removeAttribute('src');
    }
    if (webrtcVideo) {
      webrtcVideo.pause();
      webrtcVideo.removeAttribute('src');
      webrtcVideo.srcObject = null;
      webrtcVideo.style.display = 'none';
    }
  }

  function teardownWebRTC() {
    if (peerConnection) {
      try {
        peerConnection.ontrack = null;
        peerConnection.onconnectionstatechange = null;
        peerConnection.close();
      } catch (error) {
        // ignore teardown errors
      }
    }
    peerConnection = null;
    if (webrtcVideo) {
      webrtcVideo.pause();
      webrtcVideo.removeAttribute('src');
      webrtcVideo.srcObject = null;
      webrtcVideo.style.display = 'none';
    }
  }

  function stopStream(message = 'No stream running') {
    teardownWebRTC();
    hideAllStreams();
    currentSrc = '';
    currentTransport = getSelectedTransport();
    paused = false;
    pauseBtn.textContent = 'Pause';
    lastStartConfig = null;
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
    hideAllStreams();
  }

  function showMjpegStream(src) {
    currentTransport = TRANSPORT_MJPEG;
    currentSrc = src;
    streamImg.src = src;
    streamImg.style.display = 'block';
    if (webrtcVideo) {
      webrtcVideo.style.display = 'none';
      webrtcVideo.srcObject = null;
    }
    if (placeholder) {
      placeholder.style.display = 'none';
    }
  }

  async function applyVehicleModel(weightId) {
    if (!weightId) {
      return;
    }
    const previousValue = currentVehicleModel;
    vehicleModelSelect.disabled = true;
    try {
      const response = await fetch('/api/vehicle_models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weight: weightId }),
      });
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      const payload = await response.json();
      const selected = typeof payload?.selected === 'string' && payload.selected
        ? payload.selected
        : weightId;
      currentVehicleModel = selected;
      vehicleModelSelect.value = selected;
    } catch (error) {
      console.error('Failed to switch vehicle detector', error); // eslint-disable-line no-console
      if (previousValue) {
        vehicleModelSelect.value = previousValue;
      }
    } finally {
      vehicleModelSelect.disabled = false;
    }
  }

  async function loadVehicleModels() {
    vehicleModelSelect.disabled = true;
    vehicleModelSelect.innerHTML = '';

    const loadingOption = document.createElement('option');
    loadingOption.textContent = 'Loading...';
    loadingOption.disabled = true;
    loadingOption.selected = true;
    vehicleModelSelect.appendChild(loadingOption);

    try {
      const response = await fetch('/api/vehicle_models');
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const payload = await response.json();
      const models = Array.isArray(payload?.models) ? payload.models : [];
      const selected = typeof payload?.selected === 'string' ? payload.selected : null;

      vehicleModelSelect.innerHTML = '';
      if (!models.length) {
        const emptyOption = document.createElement('option');
        emptyOption.textContent = 'No models found';
        emptyOption.disabled = true;
        emptyOption.selected = true;
        vehicleModelSelect.appendChild(emptyOption);
        currentVehicleModel = null;
        return;
      }

      let foundSelected = false;
      let added = 0;
      models.forEach((model) => {
        if (!model) {
          return;
        }
        const option = document.createElement('option');
        const optionValue = typeof model.path === 'string' && model.path
          ? model.path
          : model.filename || '';
        if (!optionValue) {
          return;
        }
        option.value = optionValue;
        option.textContent = model.label || model.filename || optionValue;
        if (selected && optionValue === selected) {
          option.selected = true;
          foundSelected = true;
        }
        vehicleModelSelect.appendChild(option);
        added += 1;
      });

      if (added === 0) {
        const emptyOption = document.createElement('option');
        emptyOption.textContent = 'No models found';
        emptyOption.disabled = true;
        emptyOption.selected = true;
        vehicleModelSelect.appendChild(emptyOption);
        currentVehicleModel = null;
        return;
      }

      if (!foundSelected) {
        vehicleModelSelect.selectedIndex = 0;
        const fallbackValue = vehicleModelSelect.value;
        currentVehicleModel = fallbackValue || null;
        if (fallbackValue) {
          await applyVehicleModel(fallbackValue);
        }
        return;
      }

      const selectedValue = vehicleModelSelect.value;
      currentVehicleModel = selectedValue || null;
    } catch (error) {
      console.error('Failed to load vehicle models', error); // eslint-disable-line no-console
      vehicleModelSelect.innerHTML = '';
      const errorOption = document.createElement('option');
      errorOption.textContent = 'Unable to load models';
      errorOption.disabled = true;
      errorOption.selected = true;
      vehicleModelSelect.appendChild(errorOption);
      currentVehicleModel = null;
    } finally {
      vehicleModelSelect.disabled = false;
    }
  }

  function buildStreamConfig(url, mode) {
    const config = {
      url,
      mode,
      transport: getSelectedTransport(),
      vconf: vconf ? vconf.value : undefined,
      pconf: pconf ? pconf.value : undefined,
      readPlate: readPlateToggle ? readPlateToggle.checked : undefined,
    };
    return config;
  }

  function startMjpegStream(config) {
    teardownWebRTC();
    const params = new URLSearchParams({ url: config.url });
    const preview = config.mode === 'preview';
    if (!preview && typeof config.vconf !== 'undefined') {
      params.set('vconf', String(config.vconf));
    }
    if (!preview && typeof config.pconf !== 'undefined') {
      params.set('pconf', String(config.pconf));
    }
    if (!preview && typeof config.readPlate !== 'undefined') {
      params.set('read_plate', String(config.readPlate));
    }
    const endpoint = preview ? '/api/video' : '/api/alpr_stream';
    const src = `${endpoint}?${params.toString()}`;
    paused = false;
    pauseBtn.textContent = 'Pause';
    showMjpegStream(src);
  }

  async function startWebRTCStream(config, nonce) {
    teardownWebRTC();
    currentTransport = TRANSPORT_WEBRTC;
    hideAllStreams();
    if (placeholder) {
      if (placeholderText) {
        placeholderText.textContent = 'Connecting via WebRTC...';
      }
      placeholder.style.display = 'flex';
    }

    const pc = new RTCPeerConnection();
    peerConnection = pc;

    pc.addTransceiver('video', { direction: 'recvonly' });

    pc.ontrack = (event) => {
      if (nonce !== currentStreamNonce) {
        return;
      }
      const [stream] = event.streams || [];
      if (!stream || !webrtcVideo) {
        return;
      }
      webrtcVideo.srcObject = stream;
      webrtcVideo.style.display = 'block';
      if (placeholder) {
        placeholder.style.display = 'none';
      }
      const playPromise = webrtcVideo.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => {});
      }
    };

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed') {
        showPlaceholder('WebRTC connection failed');
        teardownWebRTC();
      } else if (pc.connectionState === 'disconnected') {
        showPlaceholder('WebRTC connection lost');
      }
    };

    const payload = {
      sdp: undefined,
      type: undefined,
      url: config.url,
      mode: config.mode,
    };

    const preview = config.mode === 'preview';
    if (!preview && typeof config.vconf !== 'undefined') {
      payload.vconf = config.vconf;
    }
    if (!preview && typeof config.pconf !== 'undefined') {
      payload.pconf = config.pconf;
    }
    if (!preview && typeof config.readPlate !== 'undefined') {
      payload.read_plate = config.readPlate;
    }

    try {
      const offer = await pc.createOffer();
      if (nonce !== currentStreamNonce) {
        return;
      }
      await pc.setLocalDescription(offer);
      payload.sdp = offer.sdp;
      payload.type = offer.type;

      const response = await fetch('/api/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      if (nonce !== currentStreamNonce) {
        return;
      }
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
      if (placeholder) {
        placeholder.style.display = 'none';
      }
      paused = false;
      pauseBtn.textContent = 'Pause';
    } catch (error) {
      console.error('Failed to start WebRTC stream', error); // eslint-disable-line no-console
      showPlaceholder('Unable to start WebRTC stream. Check console for details.');
      teardownWebRTC();
    }
  }

  async function startStream(config) {
    lastStartConfig = config;
    paused = false;
    pauseBtn.textContent = 'Pause';
    currentStreamNonce += 1;
    const nonce = currentStreamNonce;
    if (config.transport === TRANSPORT_WEBRTC) {
      await startWebRTCStream(config, nonce);
    } else {
      startMjpegStream(config);
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

  vehicleModelSelect.addEventListener('change', async () => {
    const value = vehicleModelSelect.value;
    if (!value || value === currentVehicleModel) {
      return;
    }
    await applyVehicleModel(value);
  });

  startBtn.addEventListener('click', async () => {
    const url = streamInput.value.trim();
    if (!url) {
      streamInput.classList.add('input-error');
      streamInput.focus();
      showPlaceholder('Provide a stream URL to begin');
      return;
    }

    const mode = getSelectedMode();
    const config = buildStreamConfig(url, mode);
    await startStream(config);
  });

  pauseBtn.addEventListener('click', async () => {
    if (!lastStartConfig && !currentSrc) {
      return;
    }

    if (currentTransport === TRANSPORT_MJPEG) {
      if (!currentSrc) {
        return;
      }
      if (!paused) {
        streamImg.removeAttribute('src');
        showPlaceholder('Stream paused');
        pauseBtn.textContent = 'Resume';
        paused = true;
      } else {
        showMjpegStream(currentSrc);
        pauseBtn.textContent = 'Pause';
        paused = false;
      }
      return;
    }

    if (!paused) {
      teardownWebRTC();
      showPlaceholder('Stream paused');
      pauseBtn.textContent = 'Resume';
      paused = true;
    } else {
      if (!lastStartConfig) {
        return;
      }
      paused = false;
      pauseBtn.textContent = 'Pause';
      await startStream(lastStartConfig);
    }
  });

  stopBtn.addEventListener('click', () => {
    const preview = getSelectedMode() === 'preview';
    const message = preview
      ? 'Camera preview stopped. Start to view the camera feed.'
      : 'No stream running';
    stopStream(message);
  });

  transportSelect.addEventListener('change', () => {
    const value = getSelectedTransport();
    const message = value === TRANSPORT_WEBRTC
      ? 'WebRTC selected. Start to negotiate the stream.'
      : 'HTTP MJPEG selected. Start to view the camera feed.';
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
  loadVehicleModels();
});
