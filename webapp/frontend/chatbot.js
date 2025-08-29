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

  // Persist chat history while the chat is open; clear on close
  const HISTORY_KEY = 'chatbot_history_v1';
  function loadHistory() {
    try { const raw = localStorage.getItem(HISTORY_KEY); return raw ? JSON.parse(raw) : []; } catch (_) { return []; }
  }
  function saveHistory(arr) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(arr)); } catch (_) {}
  }
  function clearHistory() { try { localStorage.removeItem(HISTORY_KEY); } catch (_) {} }

  let history = loadHistory();
  function renderHistory() {
    messagesEl.innerHTML = '';
    for (const m of history) {
      appendMessage(m.role, m.text, true);
    }
  }

  function togglePanel(show) {
    if (show === true) {
      panel.classList.remove('hidden');
      fab.classList.add('hidden');
      renderHistory();
      inputEl.focus();
    } else if (show === false) {
      panel.classList.add('hidden');
      fab.classList.remove('hidden');
      // Clear history on close per requirement
      history = [];
      clearHistory();
      messagesEl.innerHTML = '';
    } else {
      panel.classList.toggle('hidden');
      fab.classList.toggle('hidden');
    }
  }

  fab.addEventListener('click', () => togglePanel(true));
  closeBtn.addEventListener('click', () => togglePanel(false));

  function appendMessage(role, text, fromHistory = false) {
    const wrap = document.createElement('div');
    wrap.className = `chat-msg ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    if (!fromHistory) {
      history.push({ role, text });
      saveHistory(history);
    }
    return { bubble, index: history.length - 1 };
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
    const botEntry = appendMessage('bot', '');
    const botBubble = botEntry.bubble;
    const botIndex = botEntry.index;
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
            // persist incremental bot text
            history[botIndex].text = botText;
            saveHistory(history);
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }
        }
      } else {
        // Fallback for browsers/environments without fetch streaming
        const text = await res.text();
        botText = text || 'Error: failed to receive response.';
        botBubble.innerHTML = mdToHtml(botText);
        history[botIndex].text = botText;
        saveHistory(history);
      }
    } catch (e) {
      botBubble.textContent = 'Error: ' + (e && e.message ? e.message : 'unknown');
    }
    // Ensure final message stored
    history[botIndex].text = botText || history[botIndex].text || '';
    saveHistory(history);
  }

  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
});
