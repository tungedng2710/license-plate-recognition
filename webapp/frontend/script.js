document.addEventListener('DOMContentLoaded', () => {
  const homeLink = document.getElementById('home-link');
  const trainingLink = document.getElementById('training-link');
  const homePage = document.getElementById('home-page');
  const trainingPage = document.getElementById('training-page');
  const startBtn = document.getElementById('start-btn');

  function handleRouteChange() {
    const hash = window.location.hash;
    if (hash === '#/training') {
      homePage.style.display = 'none';
      trainingPage.style.display = 'block';
      homeLink.classList.remove('active');
      trainingLink.classList.add('active');
    } else {
      homePage.style.display = 'flex';
      trainingPage.style.display = 'none';
      homeLink.classList.add('active');
      trainingLink.classList.remove('active');
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
