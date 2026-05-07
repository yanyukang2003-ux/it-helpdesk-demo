/**
 * Session management — localStorage persistence + sidebar UI.
 *
 * Keys:
 *   helpdesk_sessions       → SessionMeta[]
 *   helpdesk_sd_{id}        → { messages[], lastUserMessage }
 *   helpdesk_current_session → string (session id)
 */
import { state, resetChat } from './app.js';
import { renderChat } from './chat.js';
import { renderGraph } from './graph.js';
import { fetchCheckpoints } from './api.js';

const SESSIONS_KEY = 'helpdesk_sessions';
const CURRENT_KEY = 'helpdesk_current_session';
const SD_PREFIX = 'helpdesk_sd_';
const MAX_SESSIONS = 50;

let currentId = null;

// ---- localStorage helpers -------------------------------------------

function loadSessions() {
  try { return JSON.parse(localStorage.getItem(SESSIONS_KEY)) || []; }
  catch { return []; }
}

function saveSessions(list) {
  try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(list)); } catch {}
}

function loadSessionData(id) {
  try { return JSON.parse(localStorage.getItem(SD_PREFIX + id)); } catch { return null; }
}

function saveSessionData(id, data) {
  try { localStorage.setItem(SD_PREFIX + id, JSON.stringify(data)); } catch {}
}

function removeSessionData(id) {
  try { localStorage.removeItem(SD_PREFIX + id); } catch {}
}

function setCurrentId(id) {
  currentId = id;
  try { localStorage.setItem(CURRENT_KEY, id); } catch {}
}

function getCurrentId() {
  if (currentId) return currentId;
  try { currentId = localStorage.getItem(CURRENT_KEY); } catch {}
  return currentId;
}

// ---- Public ---------------------------------------------------------

export function saveSessionState() {
  const id = getCurrentId();
  if (!id) return;
  const list = loadSessions();
  const idx = list.findIndex(s => s.id === id);
  if (idx === -1) return;

  const meta = list[idx];
  meta.messageCount = state.messages.length;
  meta.updatedAt = Date.now();
  meta.threadId = state.threadId;

  // Derive title from first user message
  if (meta.title === '新会话') {
    const firstUser = state.messages.find(m => m.role === 'user');
    if (firstUser) {
      const t = String(firstUser.content || '').trim();
      meta.title = t.length > 40 ? t.slice(0, 40) + '…' : t;
    }
  }

  list.splice(idx, 1);
  list.unshift(meta);
  saveSessions(list);

  saveSessionData(id, {
    messages: state.messages,
    lastUserMessage: state.lastUserMessage,
  });
  renderSidebarList();
}

export function createNewSession() {
  const curId = getCurrentId();
  if (curId && state.messages.length > 0) {
    saveSessionState();
  }

  const id = crypto.randomUUID();
  const meta = {
    id,
    title: '新会话',
    threadId: null,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messageCount: 0,
  };

  const list = loadSessions();
  if (list.length >= MAX_SESSIONS) {
    // Remove oldest empty session, or oldest overall
    const empty = [...list].reverse().find(s => s.messageCount === 0);
    if (empty) {
      removeSessionData(empty.id);
      const i = list.findIndex(s => s.id === empty.id);
      if (i !== -1) list.splice(i, 1);
    } else {
      const oldest = list[list.length - 1];
      removeSessionData(oldest.id);
      list.pop();
    }
  }
  list.unshift(meta);
  saveSessions(list);
  setCurrentId(id);

  resetChat();
  renderChat();
  renderGraph();
  renderSidebarList();
}

export async function switchSession(id) {
  const curId = getCurrentId();
  if (curId && curId !== id && state.messages.length > 0) {
    saveSessionState();
  }

  const list = loadSessions();
  const meta = list.find(s => s.id === id);
  if (!meta) return;

  setCurrentId(id);
  state.threadId = meta.threadId || null;
  state.steps = [];
  state.totalSteps = 0;
  state.expandedBranches = {};
  state.checkpoints = [];
  state.isStreaming = false;
  state.streamAbortController = null;
  state.lastUserMessage = '';

  const data = loadSessionData(id);
  state.messages = data?.messages || [];
  state.lastUserMessage = data?.lastUserMessage || '';

  renderChat();
  renderGraph();
  renderSidebarList();

  if (state.threadId) {
    try {
      const d = await fetchCheckpoints(state.threadId);
      state.checkpoints = d.checkpoints || [];
    } catch {
      state.threadId = null;
    }
  }
}

export function deleteSession(id) {
  const list = loadSessions();
  const idx = list.findIndex(s => s.id === id);
  if (idx === -1) return;
  list.splice(idx, 1);
  removeSessionData(id);

  const curId = getCurrentId();
  if (curId === id) {
    if (list.length > 0) {
      const next = list[0];
      saveSessions(list);
      switchSession(next.id);
    } else {
      saveSessions(list);
      createNewSession();
    }
    return;
  }
  saveSessions(list);
  renderSidebarList();
}

// ---- Sidebar Rendering -----------------------------------------------

function relativeTime(ts) {
  const diff = Date.now() - ts;
  const min = Math.floor(diff / 60000);
  if (min < 1) return '刚刚';
  if (min < 60) return `${min}分钟前`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}小时前`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}天前`;
  return new Date(ts).toLocaleDateString('zh-CN');
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str == null ? '' : String(str);
  return d.innerHTML;
}

function renderSidebarList() {
  const el = document.getElementById('sidebar-list');
  if (!el) return;
  const list = loadSessions();
  const curId = getCurrentId();

  el.innerHTML = list.map(s => {
    const isActive = s.id === curId;
    return `
      <div class="session-item${isActive ? ' active' : ''}" data-id="${escapeHtml(s.id)}">
        <div class="session-item-info">
          <div class="session-item-title">${escapeHtml(s.title)}</div>
          <div class="session-item-meta">${s.messageCount}条 · ${relativeTime(s.updatedAt)}</div>
        </div>
        <button class="session-item-delete" data-id="${escapeHtml(s.id)}" aria-label="删除会话">×</button>
      </div>`;
  }).join('');
}

function renderSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  sidebar.innerHTML = `
    <div class="sidebar-header">
      <span class="sidebar-title">会话列表</span>
      <button class="sidebar-new-btn" id="btn-new-session" title="新建会话">+</button>
    </div>
    <div class="sidebar-list" id="sidebar-list"></div>`;
  renderSidebarList();
  attachSidebarEvents();
}

// ---- Events ----------------------------------------------------------

function attachSidebarEvents() {
  // New session button
  document.getElementById('btn-new-session')?.addEventListener('click', createNewSession);

  // Session item clicks
  document.getElementById('sidebar-list')?.addEventListener('click', (e) => {
    const item = e.target.closest('.session-item');
    if (!item) return;
    const id = item.dataset.id;

    // Delete button
    if (e.target.closest('.session-item-delete')) {
      e.stopPropagation();
      deleteSession(id);
      return;
    }

    // Switch
    if (id !== getCurrentId()) {
      switchSession(id);
      // Close mobile sidebar after switch
      const sidebar = document.getElementById('sidebar');
      if (sidebar?.classList.contains('open')) toggleSidebar();
    }
  });

  // Hamburger toggle
  document.getElementById('sidebar-toggle')?.addEventListener('click', toggleSidebar);

  // Close sidebar when clicking main content on mobile
  document.querySelector('.main-layout')?.addEventListener('click', (e) => {
    const sidebar = document.getElementById('sidebar');
    if (sidebar?.classList.contains('open') && !sidebar.contains(e.target)) {
      toggleSidebar();
    }
  });
}

export function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  sidebar.classList.toggle('open');

  let backdrop = document.getElementById('sidebar-backdrop');
  if (sidebar.classList.contains('open')) {
    if (!backdrop) {
      backdrop = document.createElement('div');
      backdrop.id = 'sidebar-backdrop';
      backdrop.className = 'sidebar-backdrop';
      backdrop.addEventListener('click', toggleSidebar);
      document.body.appendChild(backdrop);
    }
  } else {
    backdrop?.remove();
  }
}

// ---- Init ------------------------------------------------------------

export function initSessionManager() {
  const list = loadSessions();
  let curId = getCurrentId();

  // If no sessions exist, create default
  if (list.length === 0) {
    const id = crypto.randomUUID();
    const meta = {
      id,
      title: '新会话',
      threadId: null,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messageCount: 0,
    };
    list.push(meta);
    saveSessions(list);
    setCurrentId(id);
    saveSessionData(id, { messages: [], lastUserMessage: '' });
  } else if (!curId || !list.find(s => s.id === curId)) {
    curId = list[0].id;
    setCurrentId(curId);
  }

  // Restore current session
  const meta = list.find(s => s.id === getCurrentId());
  state.threadId = meta?.threadId || null;
  const data = loadSessionData(getCurrentId());
  state.messages = data?.messages || [];
  state.lastUserMessage = data?.lastUserMessage || '';

  renderSidebar();
  renderChat();
  renderGraph();

  if (state.threadId) {
    fetchCheckpoints(state.threadId)
      .then(d => { state.checkpoints = d.checkpoints || []; })
      .catch(() => { state.threadId = null; });
  }
}
