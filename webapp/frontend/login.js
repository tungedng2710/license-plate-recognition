document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('login-form');
  const errorMsg = document.getElementById('login-error');

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    if (username === 'admin' && password === 'admin') {
      localStorage.setItem('authenticated', 'true');
      // Persist basic user identity for the User page
      localStorage.setItem('userName', username);
      if (!localStorage.getItem('userEmail')) {
        localStorage.setItem('userEmail', `${username}@example.com`);
      }
      window.location.href = 'index.html';
    } else {
      errorMsg.classList.remove('hidden');
    }
  });
});
