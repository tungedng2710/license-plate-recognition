document.addEventListener('DOMContentLoaded', () => {
  const alprForm = document.getElementById('alpr-form');
  const alprResult = document.getElementById('alpr-result');
  const alprImageInput = document.getElementById('alpr-image');
  const alprUploadedImage = document.getElementById('alpr-uploaded-image');
  const toggleBtn = document.getElementById('toggle-sidebar');
  const sidebar = document.querySelector('.sidebar');

  toggleBtn.addEventListener('click', () => {
    const isCollapsed = sidebar.classList.toggle('collapsed');
    toggleBtn.innerHTML = isCollapsed ? '<i class="fas fa-chevron-right"></i>' : '<i class="fas fa-chevron-left"></i>';
  });

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
});
