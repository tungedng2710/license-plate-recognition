document.addEventListener('DOMContentLoaded', () => {
  const homeLink = document.getElementById('home-link');
  const projectsLink = document.getElementById('projects-link');
  const homePage = document.getElementById('home-page');
  const projectsPage = document.getElementById('projects-page');
  const startBtn = document.getElementById('start-btn');

  function handleRouteChange() {
    const hash = window.location.hash;
    if (hash === '#/projects') {
      homePage.style.display = 'none';
      projectsPage.style.display = 'block';
      homeLink.classList.remove('active');
      projectsLink.classList.add('active');
    } else {
      homePage.style.display = 'flex';
      projectsPage.style.display = 'none';
      homeLink.classList.add('active');
      projectsLink.classList.remove('active');
    }
  }

  window.addEventListener('hashchange', handleRouteChange);
  startBtn.addEventListener('click', () => {
    window.location.hash = '#/projects';
  });

  handleRouteChange();
});
