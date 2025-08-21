document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('toggle-sidebar');
  const sidebar = document.querySelector('.sidebar');
  const userBtn = document.querySelector('.user-settings');
  const userPopup = document.getElementById('user-popup');

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const isCollapsed = sidebar.classList.toggle('collapsed');
      toggleBtn.innerHTML = isCollapsed ? '<i class="fas fa-chevron-right"></i>' : '<i class="fas fa-chevron-left"></i>';
    });
  }

  if (userBtn && userPopup) {
    userBtn.addEventListener('click', (e) => {
      e.preventDefault();
      userPopup.classList.toggle('show');
    });

    document.addEventListener('click', (e) => {
      if (!userPopup.contains(e.target) && !userBtn.contains(e.target)) {
        userPopup.classList.remove('show');
      }
    });
  }
});
