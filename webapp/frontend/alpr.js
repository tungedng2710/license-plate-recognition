document.addEventListener('DOMContentLoaded', () => {
  const alprForm = document.getElementById('alpr-form');
  const alprResult = document.getElementById('alpr-result');
  const alprImageInput = document.getElementById('alpr-image');
  const alprUploadedImage = document.getElementById('alpr-uploaded-image');
  const streamForm = document.getElementById('alpr-stream-form');
  const streamUrl = document.getElementById('alpr-rtsp');
  const streamImage = document.getElementById('alpr-stream');

  alprImageInput.addEventListener('change', () => {
    const file = alprImageInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        alprUploadedImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  });

  alprForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = alprImageInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/alpr', { method: 'POST', body: formData });
    const blob = await res.blob();
    alprResult.src = URL.createObjectURL(blob);
  });

  streamForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const url = streamUrl.value.trim();
    if (!url) return;
    streamImage.src = `/api/alpr/stream?url=${encodeURIComponent(url)}`;
  });
});
