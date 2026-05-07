/**
 * Application state + SSE wiring for plan-and-execute flow.
 *
 * Stream events:
 *   - plan_complete : { steps: [...], thread_id }
 *   - step_complete : { index, step, output, checkpoint_id }
 *   - thread_forked : { thread_id }              (for as_new_thread branches)
 *   - done          : { reply, thread_id }
 *   - error         : { message }
 */
import { streamChat, fetchCheckpoints, consumeSSE } from './api.js';
import { renderChat, setupChatInput } from './chat.js';
import { renderGraph, resetGraph } from './graph.js';
import { closeModificationPanel } from './branch.js';

// ---- State ---------------------------------------------------------

export const state = {
  threadId: null,
  isStreaming: false,
  streamAbortController: null,
  messages: [],

  // Dynamic step list. Populated when plan_complete arrives, filled in
  // by subsequent step_complete events.
  steps: [],          // each: { index, step, output, status, checkpointId, selectedAltIdx, selectedStyleKey }
  totalSteps: 0,
  expandedBranches: {},  // keyed by step index

  checkpoints: [],
  userInfo: {
    userId: localStorage.getItem('helpdesk_userId') || '',
    userName: localStorage.getItem('helpdesk_userName') || '',
    department: localStorage.getItem('helpdesk_department') || '',
  },
  activeModification: null,
  lastUserMessage: '',
};

// ---- SSE event handlers --------------------------------------------

export function onPlanComplete(event) {
  state.threadId = event.thread_id || state.threadId;
  state.totalSteps = (event.steps || []).length;
  state.steps = (event.steps || []).map((s, i) => ({
    index: i,
    step: s,
    output: null,
    status: 'pending',
    checkpointId: null,
    selectedAltIdx: 0,
    selectedStyleKey: 'default',
  }));
  state.expandedBranches = {};
  renderGraph();
}

export function onStepComplete(event) {
  const i = event.index;
  // Ensure slot exists (defensive in case plan_complete was missed)
  while (state.steps.length <= i) {
    state.steps.push({
      index: state.steps.length,
      step: null, output: null, status: 'pending',
      checkpointId: null, selectedAltIdx: 0, selectedStyleKey: 'default',
    });
  }
  const slot = state.steps[i];
  slot.step = event.step;
  slot.output = event.output;
  slot.status = 'completed';
  slot.checkpointId = event.checkpoint_id;
  // Mark all earlier as completed too
  for (let k = 0; k < i; k++) {
    if (state.steps[k] && state.steps[k].status === 'pending') {
      state.steps[k].status = 'completed';
    }
  }
  renderGraph();
}

export function onDone(event) {
  state.isStreaming = false;
  state.streamAbortController = null;
  state.threadId = event.thread_id || state.threadId;

  if (event.reply) {
    state.messages.push({ role: 'ai', content: event.reply });
  }

  renderChat();
  renderGraph();

  if (state.threadId) {
    fetchCheckpoints(state.threadId)
      .then((d) => { state.checkpoints = d.checkpoints || []; })
      .catch(() => {});
  }
}

export function onThreadForked(event) {
  if (event.thread_id) state.threadId = event.thread_id;
}

export function onError(event) {
  state.isStreaming = false;
  state.streamAbortController = null;
  state.messages.push({
    role: 'ai',
    content: `Error: ${event.message || 'Unknown error'}`,
    isError: true,
  });
  renderChat();
  renderGraph();
}

export function dispatchSSE(event) {
  switch (event.type) {
    case 'plan_complete':  onPlanComplete(event);  break;
    case 'step_complete':  onStepComplete(event);  break;
    case 'thread_forked':  onThreadForked(event);  break;
    case 'done':           onDone(event);          break;
    case 'error':          onError(event);         break;
  }
}

// ---- Public: send / interrupt / reset ------------------------------

export async function sendMessage(message) {
  if (state.isStreaming || !message.trim()) return;

  state.isStreaming = true;
  state.lastUserMessage = message.trim();
  state.messages.push({ role: 'user', content: message.trim() });

  // Reset graph for new run
  state.steps = [];
  state.totalSteps = 0;
  state.expandedBranches = {};
  closeModificationPanel();
  renderChat();
  renderGraph();

  const controller = new AbortController();
  state.streamAbortController = controller;

  try {
    const response = await streamChat({
      message: message.trim(),
      userId: state.userInfo.userId,
      userName: state.userInfo.userName,
      department: state.userInfo.department,
      threadId: state.threadId,
    }, controller.signal);
    await consumeSSE(response, dispatchSSE);
  } catch (e) {
    if (e.name === 'AbortError') {
      state.isStreaming = false;
      state.streamAbortController = null;
      state.messages.push({
        role: 'ai',
        content: '已中断执行。你可以修改上方步骤的决策后重跑。',
        isInfo: true,
      });
    } else {
      state.isStreaming = false;
      state.streamAbortController = null;
      state.messages.push({
        role: 'ai',
        content: `连接失败：${e.message}`,
        isError: true,
      });
    }
    renderChat();
    renderGraph();
  }
}

export function interruptStream() {
  if (state.streamAbortController) {
    state.streamAbortController.abort();
    state.streamAbortController = null;
  }
}

export function resetChat() {
  state.threadId = null;
  state.messages = [];
  state.checkpoints = [];
  state.steps = [];
  state.totalSteps = 0;
  state.expandedBranches = {};
  state.lastUserMessage = '';
  closeModificationPanel();
  renderChat();
  renderGraph();
}

// ---- User info -----------------------------------------------------

function setupUserInfo() {
  const fields = ['userId', 'userName', 'department'];
  for (const field of fields) {
    const el = document.getElementById(`user-${field}`);
    if (!el) continue;
    el.value = state.userInfo[field];
    el.addEventListener('change', () => {
      state.userInfo[field] = el.value;
      localStorage.setItem(`helpdesk_${field}`, el.value);
    });
  }
}

// ---- Init ----------------------------------------------------------

export function initApp() {
  setupUserInfo();
  setupChatInput();
  renderChat();
  renderGraph();
}

