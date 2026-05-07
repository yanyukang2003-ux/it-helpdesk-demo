/**
 * Branch / re-run helpers + modification drawer + checkpoint panel.
 *
 * Adapted to the dynamic plan-and-execute backend. Branching now hits a
 * streaming endpoint and we replay the same SSE dispatcher used for the
 * initial chat.
 */
import { state, dispatchSSE } from './app.js';
import { streamBranch, fetchCheckpoints, consumeSSE } from './api.js';
import { renderChat } from './chat.js';
import { renderGraph } from './graph.js';

const KIND_LABELS = {
  analysis: '分析', retrieval: '检索', reasoning: '推断',
  summarize: '总结', answer: '回答',
};

// ---- Override-based rerun (alternatives / styles) ------------------

export async function runFromAlternative({ stepIndex, overrideState, applyLocal }) {
  if (!state.threadId) { showToast('当前会话不可用', 'error'); return; }
  const it = state.steps[stepIndex];
  if (!it?.checkpointId) { showToast('找不到该步骤的 checkpoint', 'error'); return; }

  // Optimistic local change (e.g. swap primary text right away)
  if (typeof applyLocal === 'function') applyLocal();

  // Mark downstream steps as pending so they re-render as "thinking"
  for (let i = stepIndex + 1; i < state.steps.length; i++) {
    state.steps[i].status = 'pending';
    state.steps[i].output = null;
    state.steps[i].checkpointId = null;
  }
  state.isStreaming = true;
  renderGraph();

  try {
    const response = await streamBranch({
      source_thread_id: state.threadId,
      from_checkpoint_id: it.checkpointId,
      new_human_content: null,
      override_state: overrideState,
      as_new_thread: false,
    });
    await consumeSSE(response, dispatchSSE);
  } catch (e) {
    state.isStreaming = false;
    showToast('重跑失败：' + e.message, 'error');
    renderGraph();
    throw e;
  }
}

// ---- Modification drawer (edit user message) ----------------------

export function openModificationPanel(label, stepItem, checkpointId) {
  state.activeModification = { label, stepItem, checkpointId };
  renderModificationPanel();
}

export function closeModificationPanel() {
  state.activeModification = null;
  renderModificationPanel();
}

export function renderModificationPanel() {
  const panel = document.getElementById('modification-panel');
  if (!panel) return;

  if (!state.activeModification) {
    panel.innerHTML = '';
    panel.classList.remove('active');
    return;
  }

  const { label } = state.activeModification;

  panel.innerHTML = `
    <div class="modification-overlay" id="mod-overlay"></div>
    <div class="modification-drawer">
      <div class="mod-header">
        <h3>编辑提问并重新规划流程</h3>
        <button class="mod-close-btn" id="mod-close" aria-label="关闭">
          <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M3 3l10 10M13 3L3 13"/></svg>
        </button>
      </div>
      <div class="mod-body">
        <div class="branch-field">
          <label>新的问题（留空则保留原问题不变）</label>
          <textarea id="branch-message" rows="3"
            placeholder="改写后会让 AI 重新规划步骤数与流程……"></textarea>
        </div>
        <div class="branch-field branch-checkbox">
          <input type="checkbox" id="branch-new-thread">
          <label for="branch-new-thread">复制为新会话（不影响当前对话）</label>
        </div>
        <div id="branch-error" class="branch-error" style="display:none"></div>
      </div>
      <div class="mod-footer">
        <button class="btn btn-secondary" id="mod-cancel">取消</button>
        <button class="btn btn-primary" id="mod-execute">重新规划并执行</button>
      </div>
    </div>`;

  panel.classList.add('active');

  document.getElementById('mod-overlay').addEventListener('click', closeModificationPanel);
  document.getElementById('mod-close').addEventListener('click', closeModificationPanel);
  document.getElementById('mod-cancel').addEventListener('click', closeModificationPanel);

  document.getElementById('mod-execute').addEventListener('click', async () => {
    const btn = document.getElementById('mod-execute');
    const errEl = document.getElementById('branch-error');
    btn.disabled = true; btn.textContent = '执行中…';
    errEl.style.display = 'none';

    const newMsg = document.getElementById('branch-message').value.trim() || state.lastUserMessage;
    const asNew = document.getElementById('branch-new-thread').checked;

    if (!newMsg) {
      errEl.textContent = '没有可重跑的问题';
      errEl.style.display = 'block';
      btn.disabled = false; btn.textContent = '重新规划并执行';
      return;
    }

    state.isStreaming = true;
    state.steps = [];           // will be rebuilt by plan_complete
    state.totalSteps = 0;
    state.expandedBranches = {};
    if (newMsg !== state.lastUserMessage) {
      state.messages.push({ role: 'user', content: newMsg });
      state.lastUserMessage = newMsg;
    }
    closeModificationPanel();
    renderChat();
    renderGraph();

    try {
      const response = await streamBranch({
        source_thread_id: state.threadId,
        from_checkpoint_id: '',
        new_human_content: newMsg,
        override_state: null,
        as_new_thread: asNew,
      });
      await consumeSSE(response, dispatchSSE);
    } catch (e) {
      state.isStreaming = false;
      showToast('执行失败：' + e.message, 'error');
      renderGraph();
    }
  });
}

// ---- Checkpoint panel ----------------------------------------------

let checkpointsPanelOpen = false;

export function toggleCheckpointPanel() {
  checkpointsPanelOpen = !checkpointsPanelOpen;
  renderCheckpointPanel();
}

export async function refreshCheckpoints() {
  if (!state.threadId) return;
  try {
    const data = await fetchCheckpoints(state.threadId);
    state.checkpoints = data.checkpoints || [];
  } catch (e) {
    console.warn('Failed to fetch checkpoints:', e);
  }
  renderCheckpointPanel();
}

export function renderCheckpointPanel() {
  const panel = document.getElementById('checkpoint-panel');
  if (!panel) return;

  if (!checkpointsPanelOpen) {
    panel.innerHTML = '';
    panel.classList.remove('active');
    return;
  }

  panel.classList.add('active');

  const closeBtn = `<button class="cp-close-btn" id="cp-close" aria-label="关闭">
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M3 3l10 10M13 3L3 13"/></svg>
  </button>`;

  if (!state.threadId) {
    panel.innerHTML = `
      <div class="cp-header"><h3>步骤检查点</h3>${closeBtn}</div>
      <div class="cp-empty">发送消息后可查看每步的检查点</div>`;
    document.getElementById('cp-close').addEventListener('click', toggleCheckpointPanel);
    return;
  }

  if (state.checkpoints.length === 0) {
    panel.innerHTML = `
      <div class="cp-header"><h3>步骤检查点</h3>${closeBtn}</div>
      <div class="cp-empty">暂无步骤记录</div>`;
    document.getElementById('cp-close').addEventListener('click', toggleCheckpointPanel);
    return;
  }

  let rows = '';
  for (const cp of state.checkpoints) {
    const step = cp.step || {};
    const out = cp.output || {};
    const kind = step.kind || 'reasoning';
    const kLabel = KIND_LABELS[kind] || kind;
    const primary = (out.primary || '').slice(0, 100);
    const stepNo = (cp.step_index ?? 0) + 1;

    rows += `
      <div class="cp-card">
        <div class="cp-card-header">
          <span class="cp-step">步骤 ${stepNo}</span>
          <span class="cp-source">${escapeHtml(step.title || '')}</span>
          <span class="cp-next tone-${kind}">${kLabel}</span>
        </div>
        ${primary ? `<div class="cp-messages">${escapeHtml(primary)}</div>` : '<div class="cp-messages cp-empty-inline">尚未执行</div>'}
        <div class="cp-actions">
          <code class="cp-id">${escapeHtml((cp.checkpoint_id || '').slice(-8))}</code>
        </div>
      </div>`;
  }

  panel.innerHTML = `
    <div class="cp-header">
      <h3>步骤检查点 <span class="cp-count">${state.checkpoints.length}</span></h3>
      <button class="btn btn-sm btn-outline" id="cp-refresh">刷新</button>
      ${closeBtn}
    </div>
    <div class="cp-list">${rows}</div>`;

  document.getElementById('cp-close').addEventListener('click', toggleCheckpointPanel);
  document.getElementById('cp-refresh').addEventListener('click', refreshCheckpoints);
}

// ---- Toast ---------------------------------------------------------

export function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 350);
  }, 3000);
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}
