document.addEventListener('DOMContentLoaded', () => {
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      localStorage.removeItem('authenticated');
      window.location.href = 'login.html';
    });
  }

  const isLoginPage = window.location.pathname.endsWith('login.html');
  const authenticated = localStorage.getItem('authenticated');
  if (!isLoginPage && !authenticated) {
    window.location.href = 'login.html';
  }
  if (isLoginPage && authenticated) {
    window.location.href = 'index.html';
  }
});
