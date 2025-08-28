document.addEventListener('DOMContentLoaded', () => {
  // Create floating action button (FAB)
  const fab = document.createElement('button');
  fab.className = 'chatbot-fab';
  fab.setAttribute('aria-label', 'Open chatbot');
  fab.innerHTML = '<i class="fas fa-comment"></i>';

  // Create chat panel
  const panel = document.createElement('div');
  panel.className = 'chatbot-panel glass hidden';

  panel.innerHTML = `
    <div class="chatbot-header">
      <div class="title">TonAI Chat</div>
      <button class="chatbot-close" aria-label="Close"><i class="fas fa-times"></i></button>
    </div>
    <div class="chatbot-messages" id="chatbot-messages"></div>
    <div class="chatbot-input">
      <input type="text" id="chatbot-text" placeholder="Type a message..." />
      <button id="chatbot-send"><i class="fas fa-paper-plane"></i></button>
    </div>
  `;

  document.body.appendChild(fab);
  document.body.appendChild(panel);

  const messagesEl = panel.querySelector('#chatbot-messages');
  const inputEl = panel.querySelector('#chatbot-text');
  const sendBtn = panel.querySelector('#chatbot-send');
  const closeBtn = panel.querySelector('.chatbot-close');

  function togglePanel(show) {
    if (show === true) {
      panel.classList.remove('hidden');
      fab.classList.add('hidden');
      inputEl.focus();
    } else if (show === false) {
      panel.classList.add('hidden');
      fab.classList.remove('hidden');
    } else {
      panel.classList.toggle('hidden');
      fab.classList.toggle('hidden');
    }
  }

  fab.addEventListener('click', () => togglePanel(true));
  closeBtn.addEventListener('click', () => togglePanel(false));

  function appendMessage(role, text) {
    const wrap = document.createElement('div');
    wrap.className = `chat-msg ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  // Minimal, safe Markdown rendering for bot messages
  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function mdToHtml(src) {
    // Escape HTML first
    let s = escapeHtml(src);
    // Code blocks ```...```
    s = s.replace(/```([\s\S]*?)```/g, (m, p1) => `<pre><code>${p1}</code></pre>`);
    // Inline code `code`
    s = s.replace(/`([^`]+)`/g, (m, p1) => `<code>${p1}</code>`);
    // Bold **text**
    s = s.replace(/\*\*([^*]+)\*\*/g, (m, p1) => `<strong>${p1}</strong>`);
    // Italic *text*
    s = s.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, (m, p1) => `<em>${p1}</em>`);
    // Links [text](url)
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (m, p1, p2) => `<a href="${p2}" target="_blank" rel="noopener noreferrer">${p1}</a>`);
    // Newlines to <br>
    s = s.replace(/\r?\n/g, '<br>');
    return s;
  }

  async function sendMessage() {
    const prompt = inputEl.value.trim();
    if (!prompt) return;
    inputEl.value = '';

    appendMessage('user', prompt);
    const botBubble = appendMessage('bot', '');
    let botText = '';

    try {
      const apiBase = (window.CHATBOT_API_BASE || '').replace(/\/$/, '');
      const endpoint = `${apiBase}/api/chat`;
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      if (!res.ok) {
        try {
          const txt = await res.text();
          botBubble.textContent = txt || 'Error: failed to connect to chatbot.';
        } catch (_) {
          botBubble.textContent = 'Error: failed to connect to chatbot.';
        }
        return;
      }

      if (res.body && typeof res.body.getReader === 'function') {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        while (!done) {
          const { value, done: d } = await reader.read();
          done = d;
          if (value) {
            botText += decoder.decode(value, { stream: true });
            botBubble.innerHTML = mdToHtml(botText);
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }
        }
      } else {
        // Fallback for browsers/environments without fetch streaming
        const text = await res.text();
        botText = text || 'Error: failed to receive response.';
        botBubble.innerHTML = mdToHtml(botText);
      }
    } catch (e) {
      botBubble.textContent = 'Error: ' + (e && e.message ? e.message : 'unknown');
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
});
