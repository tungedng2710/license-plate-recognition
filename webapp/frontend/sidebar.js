document.addEventListener('DOMContentLoaded', () => {
  const userBtn = document.querySelector('.user-settings');
  const userPopup = document.getElementById('user-popup');

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
