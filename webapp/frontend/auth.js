document.addEventListener('DOMContentLoaded', () => {
  // Delegate logout handling so it works for dynamic popups
  document.addEventListener('click', (e) => {
    const target = e.target;
    if (target && (target.id === 'logout-btn' || target.closest && target.closest('#logout-btn'))) {
      localStorage.removeItem('authenticated');
      window.location.href = 'login.html';
    }
  });

  const isLoginPage = window.location.pathname.endsWith('login.html');
  const authenticated = localStorage.getItem('authenticated');
  if (!isLoginPage && !authenticated) {
    window.location.href = 'login.html';
  }
  if (isLoginPage && authenticated) {
    window.location.href = 'index.html';
  }
});
