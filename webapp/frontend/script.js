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
    const uploadBtn = document.getElementById('upload-btn');
    const uploadInput = document.getElementById('dataset-upload');
  const modalClose = document.getElementById('modal-close');
  const startBtn = document.getElementById('start-btn');
    const toggleBtn = document.getElementById('toggle-sidebar');
    const sidebar = document.querySelector('.sidebar');
    const body = document.body;

    toggleBtn.addEventListener('click', () => {
      const isCollapsed = sidebar.classList.toggle('collapsed');
      toggleBtn.innerHTML = isCollapsed ? '<i class="fas fa-chevron-right"></i>' : '<i class="fas fa-chevron-left"></i>';
    });

    uploadBtn.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('file', file);
      await fetch('/api/datasets/upload', {
        method: 'POST',
        body: formData
      });
      loadDatasetCards();
      loadDatasets();
    });

    function handleRouteChange() {
      const hash = window.location.hash;
      body.classList.remove('home-bg');
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
        body.classList.add('home-bg');
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

    for (const ds of data.datasets) {
      const statsRes = await fetch(`/api/datasets/${ds}/stats`);
      const stats = await statsRes.json();

      const div = document.createElement('div');
      div.className = 'dataset-card';
      div.innerHTML = `
        <img src="/api/datasets/${ds}/thumbnail" alt="${ds} thumbnail" />
        <div class="p-4">
          <h3 class="text-lg font-bold mb-2">${ds}</h3>
          <p class="text-sm mb-1">Classes: ${stats.classes}</p>
          <p class="text-sm mb-2">Train: ${stats.train}, Val: ${stats.val}, Test: ${stats.test}</p>
          <div class="tags">${(stats.tags || []).map(t => '<span>' + t + '</span>').join('')}</div>
        </div>
      `;
      div.onclick = () => showDatasetStats(ds);
      container.appendChild(div);
    }
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
