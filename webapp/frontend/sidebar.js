document.addEventListener('DOMContentLoaded', () => {
  const userBtn = document.querySelector('.user-settings');
  const userPopup = document.getElementById('user-popup');

  if (userBtn && userPopup) {
    // Populate the popup content dynamically
    const displayName = (userBtn.querySelector('.link-text')?.textContent || 'User').trim();
    const email = (localStorage.getItem('userEmail') || `${displayName.toLowerCase()}@example.com`).trim();

    userPopup.innerHTML = `
      <div class="user-info">
        <div class="label">User</div>
        <div class="value" id="user-username">${displayName}</div>
        <div class="label">Name</div>
        <div class="value" id="user-name">${displayName}</div>
        <div class="label">Email</div>
        <div class="value" id="user-email">${email}</div>
      </div>
      <div class="user-actions">
        <button id="logout-btn" class="logout-btn">Logout</button>
      </div>
    `;

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
