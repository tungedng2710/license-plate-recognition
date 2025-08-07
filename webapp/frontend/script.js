  document.addEventListener('DOMContentLoaded', () => {
    const homeLink = document.getElementById('home-link');
    const trainingLink = document.getElementById('training-link');
    const datasetsLink = document.getElementById('datasets-link');
    const projectsLink = document.getElementById('projects-link');
    const homePage = document.getElementById('home-page');
    const trainingPage = document.getElementById('training-page');
    const datasetsPage = document.getElementById('datasets-page');
    const projectsPage = document.getElementById('projects-page');
    const datasetModal = document.getElementById('dataset-modal');
  const modalClose = document.getElementById('modal-close');
  const startBtn = document.getElementById('start-btn');
  const toggleBtn = document.getElementById('toggle-sidebar');
  const sidebar = document.querySelector('.sidebar');

   VANTA.WAVES({
     el: '#home-page',
     color: 0x001f3f,
     shininess: 20,
     waveHeight: 10,
     waveSpeed: 0.5,
     zoom: 0.8
   });

    toggleBtn.addEventListener('click', () => {
      const isCollapsed = sidebar.classList.toggle('collapsed');
      toggleBtn.innerHTML = isCollapsed ? '<i class="fas fa-chevron-right"></i>' : '<i class="fas fa-chevron-left"></i>';
    });

    function handleRouteChange() {
      const hash = window.location.hash;
      if (hash === '#/training') {
        homePage.style.display = 'none';
        trainingPage.style.display = 'block';
        datasetsPage.style.display = 'none';
        projectsPage.style.display = 'none';
        homeLink.classList.remove('active');
        trainingLink.classList.add('active');
        datasetsLink.classList.remove('active');
        projectsLink.classList.remove('active');
      } else if (hash === '#/datasets') {
        homePage.style.display = 'none';
        trainingPage.style.display = 'none';
        datasetsPage.style.display = 'block';
        projectsPage.style.display = 'none';
        homeLink.classList.remove('active');
        trainingLink.classList.remove('active');
        datasetsLink.classList.add('active');
        projectsLink.classList.remove('active');
        loadDatasetCards();
      } else if (hash === '#/projects') {
        homePage.style.display = 'none';
        trainingPage.style.display = 'none';
        datasetsPage.style.display = 'none';
        projectsPage.style.display = 'block';
        homeLink.classList.remove('active');
        trainingLink.classList.remove('active');
        datasetsLink.classList.remove('active');
        projectsLink.classList.add('active');
      } else {
        homePage.style.display = 'flex';
        trainingPage.style.display = 'none';
        datasetsPage.style.display = 'none';
        projectsPage.style.display = 'none';
        homeLink.classList.add('active');
        trainingLink.classList.remove('active');
        datasetsLink.classList.remove('active');
        projectsLink.classList.remove('active');
      }
    }

  window.addEventListener('hashchange', handleRouteChange);
  // Handle initial page load
  handleRouteChange();

    startBtn.addEventListener('click', () => {
      window.location.hash = '#/training';
    });

  async function loadDatasets() {
    const res = await fetch('/api/datasets');
    const data = await res.json();
    const container = document.getElementById('datasets');
    container.innerHTML = '';

    data.datasets.forEach(ds => {
      const div = document.createElement('div');
      div.className = 'p-4 border rounded shadow hover:bg-gray-100 cursor-pointer';

      const title = document.createElement('h3');
      title.className = 'font-bold';
      title.textContent = ds;
      div.appendChild(title);

      div.onclick = () => document.getElementById('dataset').value = ds;

      container.appendChild(div);
    });
  }

  async function loadDatasetCards() {
    const res = await fetch('/api/datasets');
    const data = await res.json();
    const container = document.getElementById('dataset-cards');
    container.innerHTML = '';

    data.datasets.forEach(ds => {
      const div = document.createElement('div');
      div.className = 'dataset-card';
      div.textContent = ds;
      div.onclick = () => showDatasetStats(ds);
      container.appendChild(div);
    });
  }

  let chart;
  async function showDatasetStats(name) {
    const res = await fetch(`/api/datasets/${name}/stats`);
    const data = await res.json();
    document.getElementById('modal-title').innerText = name;
    document.getElementById('dataset-counts').innerText = `Train: ${data.train}, Val: ${data.val}, Test: ${data.test}`;
    const ctx = document.getElementById('datasetChart').getContext('2d');
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Train', 'Val', 'Test'],
        datasets: [{
          data: [data.train, data.val, data.test],
          backgroundColor: ['#3b82f6', '#10b981', '#f59e0b']
        }]
      }
    });
    datasetModal.classList.remove('hidden');
  }

  modalClose.addEventListener('click', () => {
    datasetModal.classList.add('hidden');
  });

  let progressTimer;
  async function pollProgress() {
    const res = await fetch('/api/progress');
    const data = await res.json();
    document.getElementById('progressBar').style.width = data.progress + '%';
    document.getElementById('progressText').innerText = data.progress + '%';
    if (!data.running || data.progress >= 100) {
      clearInterval(progressTimer);
    }
  }

  document.getElementById('trainForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      dataset: document.getElementById('dataset').value,
      batch: parseInt(document.getElementById('batch').value),
      img_size: parseInt(document.getElementById('img').value),
      model: document.getElementById('model').value,
      epochs: parseInt(document.getElementById('epochs').value)
    };
    await fetch('/api/train', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    document.getElementById('progressSection').classList.remove('hidden');
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').innerText = '0%';
    progressTimer = setInterval(pollProgress, 1000);
  });

  loadDatasets();
});
