/**
 * Chat UI: message list, input with LLM-powered autocomplete, interrupt button.
 */
import { state, sendMessage, interruptStream } from './app.js';

// ---- Static fallback suggestions (diverse domains) ----
const STATIC_SUGGESTIONS = [
  { category: '网络', queries: [
    'VPN怎么连接？', '无法访问外网怎么办？', '公司WiFi密码是多少？',
    '网络速度很慢怎么排查？', '如何配置代理？'
  ]},
  { category: '账号', queries: [
    '怎么重置密码？', '账号被锁定了怎么办？', '如何申请新员工账号？',
    '企业邮箱怎么设置？', '多因素认证怎么开启？'
  ]},
  { category: '设备', queries: [
    '新电脑怎么配置开发环境？', '打印机怎么连接？', '电脑蓝屏了怎么办？',
    'Outlook怎么配置？', '手机端如何同步企业邮件？'
  ]},
  { category: '软件', queries: [
    'Office365怎么激活？', 'VPN客户端从哪里下载？', 'Jira怎么登录？',
    'Slack无法连接怎么办？', '如何申请软件许可证？'
  ]},
];

// ---- Autocomplete State ----
let selectedSuggestionIndex = -1;
let currentSuggestions = [];
let autocompleteVisible = false;
let llmPredictTimer = null;
let isPredicting = false;

// ---- LLM Prediction ----
async function fetchPredictions(partial) {
  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ partial }),
    });
    if (!resp.ok) return [];
    const data = await resp.json();
    return (data.predictions || []).map(p => ({ category: '预测', query: p }));
  } catch (e) {
    return [];
  }
}

function matchStatic(input) {
  if (!input.trim()) return [];
  const lower = input.toLowerCase();
  const results = [];
  for (const group of STATIC_SUGGESTIONS) {
    for (const q of group.queries) {
      if (q.toLowerCase().includes(lower)) {
        results.push({ category: group.category, query: q });
      }
    }
  }
  return results.slice(0, 4);
}

function defaultStatic() {
  const defaults = [];
  const picked = new Set();
  for (const group of STATIC_SUGGESTIONS) {
    const q = group.queries[0];
    if (!picked.has(q)) {
      defaults.push({ category: group.category, query: q });
      picked.add(q);
    }
  }
  return defaults.slice(0, 4);
}

async function updateSuggestions(inputEl) {
  const val = inputEl.value.trim();

  if (!val) {
    currentSuggestions = defaultStatic();
    renderDropdown();
    return;
  }

  // Show static matches immediately
  const staticMatches = matchStatic(val);
  currentSuggestions = staticMatches;
  renderDropdown();
  selectedSuggestionIndex = -1;

  // Debounced LLM prediction (fires 500ms after last keystroke)
  if (val.length >= 2) {
    clearTimeout(llmPredictTimer);
    llmPredictTimer = setTimeout(async () => {
      if (isPredicting) return;
      isPredicting = true;

      // Show loading indicator
      const dropdown = document.getElementById('autocomplete-dropdown');
      if (dropdown && dropdown.style.display !== 'none') {
        const loader = document.createElement('li');
        loader.className = 'suggestion-item suggestion-loading';
        loader.textContent = 'AI 预测中...';
        loader.id = 'predict-loader';
        dropdown.appendChild(loader);
      }

      const predictions = await fetchPredictions(val);

      // Remove loader
      const loaderEl = document.getElementById('predict-loader');
      if (loaderEl) loaderEl.remove();

      // Merge: static first, then predictions (deduped)
      const existing = new Set(currentSuggestions.map(s => s.query));
      const newPredictions = predictions.filter(p => !existing.has(p.query));

      if (newPredictions.length > 0) {
        currentSuggestions = [...currentSuggestions, ...newPredictions].slice(0, 6);
        renderDropdown();
      }

      isPredicting = false;
    }, 500);
  }
}

function renderDropdown() {
  const dropdown = document.getElementById('autocomplete-dropdown');
  if (!dropdown) return;

  if (currentSuggestions.length === 0) {
    dropdown.innerHTML = '';
    dropdown.style.display = 'none';
    autocompleteVisible = false;
    return;
  }

  dropdown.innerHTML = currentSuggestions.map((s, i) =>
    `<li class="suggestion-item" data-index="${i}">
      <span class="suggestion-category">${s.category}</span>
      <span class="suggestion-text">${s.query}</span>
    </li>`
  ).join('');

  dropdown.style.display = 'block';
  autocompleteVisible = true;

  dropdown.querySelectorAll('.suggestion-item').forEach(item => {
    item.addEventListener('mousedown', (e) => {
      e.preventDefault();
      const idx = parseInt(item.dataset.index);
      if (currentSuggestions[idx]) {
        document.getElementById('chat-input').value = currentSuggestions[idx].query;
        hideAutocomplete();
        document.getElementById('chat-input').focus();
      }
    });
  });
}

function highlightSuggestion(index) {
  const items = document.querySelectorAll('#autocomplete-dropdown .suggestion-item');
  items.forEach(el => el.classList.remove('active'));
  if (index >= 0 && index < items.length) {
    items[index].classList.add('active');
    items[index].scrollIntoView({ block: 'nearest' });
  }
}

function hideAutocomplete() {
  const dropdown = document.getElementById('autocomplete-dropdown');
  if (dropdown) dropdown.style.display = 'none';
  autocompleteVisible = false;
  currentSuggestions = [];
  selectedSuggestionIndex = -1;
}

// ---- Message Rendering ----
function renderMessages() {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  if (state.messages.length === 0) {
    container.innerHTML = `
      <div class="welcome-state">
        <div class="welcome-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2a5 5 0 0 1 5 5v3a5 5 0 0 1-10 0V7a5 5 0 0 1 5-5z"/>
            <path d="M3 11v1a9 9 0 0 0 18 0v-1"/>
            <circle cx="9" cy="17" r="2"/>
            <circle cx="15" cy="17" r="2"/>
            <line x1="9" y1="19" x2="15" y2="19"/>
          </svg>
        </div>
        <h2>有什么可以帮助你的？</h2>
        <p>我可以回答编程、科学、技术、生活等各类问题</p>
        <div class="welcome-chips">
          <button class="welcome-chip" data-msg="VPN怎么连接？">VPN 连接</button>
          <button class="welcome-chip" data-msg="如何重置密码？">重置密码</button>
          <button class="welcome-chip" data-msg="帮我创建一个IT工单">创建工单</button>
          <button class="welcome-chip" data-msg="如何配置企业邮箱？">邮箱配置</button>
        </div>
      </div>`;
    container.querySelectorAll('.welcome-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.getElementById('chat-input').value = chip.dataset.msg;
        sendMessage(chip.dataset.msg);
      });
    });
    return;
  }

  let html = '';
  for (const msg of state.messages) {
    if (msg.role === 'user') {
      html += `<div class="message message-user">
        <div class="message-bubble user-bubble">${escapeHtml(msg.content)}</div>
      </div>`;
    } else {
      let cls = 'ai-bubble';
      if (msg.isError) cls += ' error-bubble';
      if (msg.isInfo) cls += ' info-bubble';
      html += `<div class="message message-ai">
        <div class="message-bubble ${cls}">${formatContent(msg.content)}</div>
      </div>`;
    }
  }

  if (state.isStreaming) {
    html += `<div class="message message-ai">
      <div class="message-bubble ai-bubble typing-indicator">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>
    </div>`;
  }

  container.innerHTML = html;
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatContent(text) {
  let out = escapeHtml(text);
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\n/g, '<br>');
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  return out;
}

// ---- Chat Input Setup ----
export function setupChatInput() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const interruptBtn = document.getElementById('interrupt-btn');

  if (!input || !sendBtn) return;

  input.addEventListener('input', () => {
    updateSuggestions(input);
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });

  input.addEventListener('focus', () => {
    if (!input.value.trim()) {
      currentSuggestions = defaultStatic();
      renderDropdown();
    }
  });

  input.addEventListener('blur', () => {
    clearTimeout(llmPredictTimer);
    setTimeout(hideAutocomplete, 150);
  });

  input.addEventListener('keydown', (e) => {
    if (!autocompleteVisible) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const msg = input.value;
        input.value = '';
        input.style.height = 'auto';
        sendMessage(msg);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        selectedSuggestionIndex = Math.min(selectedSuggestionIndex + 1, currentSuggestions.length - 1);
        highlightSuggestion(selectedSuggestionIndex);
        break;
      case 'ArrowUp':
        e.preventDefault();
        selectedSuggestionIndex = Math.max(selectedSuggestionIndex - 1, -1);
        highlightSuggestion(selectedSuggestionIndex);
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedSuggestionIndex >= 0 && currentSuggestions[selectedSuggestionIndex]) {
          input.value = currentSuggestions[selectedSuggestionIndex].query;
        }
        hideAutocomplete();
        const msg = input.value;
        input.value = '';
        input.style.height = 'auto';
        sendMessage(msg);
        break;
      case 'Escape':
        hideAutocomplete();
        break;
    }
  });

  sendBtn.addEventListener('click', () => {
    const msg = input.value;
    if (!msg.trim() || state.isStreaming) return;
    input.value = '';
    input.style.height = 'auto';
    hideAutocomplete();
    sendMessage(msg);
  });

  if (interruptBtn) {
    interruptBtn.addEventListener('click', () => interruptStream());
  }
}

// ---- Render ----
export function renderChat() {
  renderMessages();

  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const interruptBtn = document.getElementById('interrupt-btn');

  if (input) input.disabled = state.isStreaming;
  if (sendBtn) sendBtn.disabled = state.isStreaming;
  if (interruptBtn) {
    interruptBtn.style.display = state.isStreaming ? 'inline-flex' : 'none';
  }
}
