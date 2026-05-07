/**
 * Dynamic thinking-flow visualization
 * -----------------------------------
 * 渲染 plan-and-execute 流程：节点数和类型均由后端 LLM 决定。
 * 每个非 retrieval/answer 节点带 2 条备选思路；answer 节点带风格切换；
 * 点击备选 → 调 /threads/branch 流式重跑后续节点。
 */
import { state } from './app.js';
import { runFromAlternative, openModificationPanel } from './branch.js';

const KIND_META = {
  analysis:   { label: '分析', iconPath: 'M9 6.5a2.5 2.5 0 1 1 5 0v.5a3 3 0 0 1-1.5 2.6c-.6.4-.9 1-.9 1.6V12M11.5 16.5h0', tone: 'analysis' },
  retrieval:  { label: '检索', iconPath: 'M5 4.5h7.5a3 3 0 0 1 3 3v7.5h-9.5a1 1 0 0 1-1-1V4.5z M5 12.5h10.5', tone: 'retrieval' },
  reasoning:  { label: '推断', iconPath: 'M4 10c0-3.3 2.7-6 6-6s6 2.7 6 6-2.7 6-6 6 M10 7.5v3l2 1.5', tone: 'reasoning' },
  summarize:  { label: '总结', iconPath: 'M5 5h10M5 9h10M5 13h6', tone: 'summarize' },
  answer:     { label: '回答', iconPath: 'M4 6.5h12M4 11h8M4 15.5h10', tone: 'answer' },
};

const STYLE_OPTIONS = [
  { id: '',             label: '默认',   desc: '平实清晰的语气' },
  { id: 'concise',      label: '更简洁', desc: '三句话以内的核心回答' },
  { id: 'detailed',     label: '更详细', desc: '加入背景、原理与示例' },
  { id: 'step_by_step', label: '分步骤', desc: '拆成清晰可执行的步骤' },
];

const PATH_DOT = '<span class="branch-bullet"></span>';

// ---- helpers --------------------------------------------------------

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str == null ? '' : String(str);
  return d.innerHTML;
}

function escapeAttr(s) {
  return s.replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;');
}

function nodeIconSvg(path) {
  return `<svg viewBox="0 0 20 20" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="${path}"/></svg>`;
}

function chevronSvg(open) {
  return `<svg class="chev${open ? ' open' : ''}" viewBox="0 0 12 12" width="10" height="10" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 4.5L6 7.5L9 4.5"/></svg>`;
}

function truncate(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '…' : s;
}

// ---- node body ------------------------------------------------------

function renderNodeBody(item) {
  const { step, output } = item;
  if (!output) return '';

  const kind = step?.kind;

  if (kind === 'retrieval') {
    const meta = output.metadata || {};
    const performed = meta.performed;
    const sources = meta.sources || [];
    if (!performed) {
      return `<div class="node-summary node-summary-muted">未在知识库找到相关内容</div>`;
    }
    const tags = sources.length
      ? sources.map(s => `<span class="kb-tag">${escapeHtml(s)}</span>`).join('')
      : `<span class="kb-tag kb-tag-empty">无来源</span>`;
    return `
      <div class="node-summary node-summary-stack">
        <div class="kb-line">
          <span class="metric-label">检索 query</span>
          <span class="kb-query">${escapeHtml(truncate(meta.query || '', 40))}</span>
        </div>
        <div class="kb-tags">${tags}</div>
      </div>`;
  }

  if (kind === 'answer') {
    const styleId = output.style_hint || '';
    const styleLabel = (STYLE_OPTIONS.find(s => s.id === styleId) || STYLE_OPTIONS[0]).label;
    return `
      <div class="node-summary node-summary-stack">
        <div class="answer-source">
          <span class="source-tag source-answer">最终回答</span>
          <span class="answer-style-label">风格 · ${escapeHtml(styleLabel)}</span>
        </div>
        <div class="answer-source-desc">${escapeHtml(truncate(output.primary, 100))}</div>
      </div>`;
  }

  // analysis / reasoning / summarize: show primary text
  return `<div class="node-summary">${escapeHtml(truncate(output.primary, 200))}</div>`;
}

// ---- branches -------------------------------------------------------

function buildBranchesFor(item) {
  const { step, output, selectedAltIdx, selectedStyleKey } = item;
  if (!output) return [];

  if (step?.kind === 'retrieval') {
    return [];
  }

  if (step?.kind === 'answer') {
    return STYLE_OPTIONS.map(o => {
      const k = o.id ? `style-${o.id}` : 'default';
      return {
        key: k,
        label: o.label,
        content: o.desc,
        selected: (selectedStyleKey || 'default') === k,
        cta: o.id ? '换风格' : '默认',
        kind: 'style',
        payload: { id: o.id },
      };
    });
  }

  // analysis / reasoning / summarize
  const summaries = output.alt_summaries || [];
  const list = [
    {
      key: 'primary',
      label: '主路径',
      content: output.primary,
      selected: (selectedAltIdx ?? 0) === 0,
      cta: '回主路径',
      kind: 'primary',
      payload: { idx: 0, text: output.primary, isPrimary: true },
    },
    ...((output.alternatives || []).map((alt, i) => ({
      key: `alt-${i}`,
      label: `思路 ${String.fromCharCode(65 + i)}`,
      summary: summaries[i] || '',
      content: alt,
      selected: (selectedAltIdx ?? 0) === i + 1,
      cta: '用这个',
      kind: 'alt',
      payload: { idx: i + 1, text: alt, isPrimary: false },
    }))),
  ];
  return list;
}

function renderBranches(item, expanded) {
  const branches = buildBranchesFor(item);
  if (branches.length === 0) return '';
  const altCount = branches.filter(b => !b.selected).length;
  if (altCount === 0) return '';

  const toggleLabel = expanded ? '收起' : `${altCount} 个备选`;

  if (!expanded) {
    return `
      <button class="branches-toggle" data-step-idx="${item.index}" aria-expanded="false">
        ${chevronSvg(false)} <span>${toggleLabel}</span>
      </button>`;
  }

  const items = branches.map(b => {
    const sCls = b.selected ? ' is-selected' : '';
    const ctaHtml = !b.selected
      ? `<button class="branch-cta" data-step-idx="${item.index}" data-kind="${b.kind}" data-payload='${escapeAttr(JSON.stringify(b.payload))}'>${b.cta}</button>`
      : `<span class="branch-current">当前</span>`;
    const isAlt = b.kind === 'alt';
    const summaryTag = isAlt && b.summary
      ? `<span class="branch-summary">${escapeHtml(b.summary)}</span>`
      : '';
    const detailHtml = isAlt
      ? `<div class="branch-detail">${escapeHtml(truncate(b.content, 120))}</div>`
      : `<div class="branch-content">${escapeHtml(truncate(b.content, 200))}</div>`;
    return `
      <li class="branch-item${sCls}">
        <div class="branch-line"></div>
        <div class="branch-card">
          <div class="branch-head">
            ${PATH_DOT}
            <span class="branch-label">${escapeHtml(b.label)}</span>
            ${summaryTag}
            ${ctaHtml}
          </div>
          ${detailHtml}
        </div>
      </li>`;
  }).join('');

  return `
    <button class="branches-toggle" data-step-idx="${item.index}" aria-expanded="true">
      ${chevronSvg(true)} <span>${toggleLabel}</span>
    </button>
    <ul class="branches">${items}</ul>`;
}

// ---- node card ------------------------------------------------------

function renderNode(item, isLast) {
  const { index, step, output, status } = item;
  const meta = step ? (KIND_META[step.kind] || KIND_META.reasoning)
                    : { label: '步骤', iconPath: '', tone: 'reasoning' };
  const isDone = status === 'completed';
  const isRunning = status === 'running' || (state.isStreaming && !isDone && firstPendingIndex() === index);

  let cls = `node node-tone-${meta.tone}`;
  if (isDone) cls += ' node-done';
  if (isRunning) cls += ' node-running';

  const statusLabel = isRunning
    ? `<span class="node-status">思考中</span>`
    : isDone
      ? `<span class="node-status node-status-done">完成</span>`
      : `<span class="node-status node-status-pending">待执行</span>`;

  const expanded = state.expandedBranches?.[index] === true;
  const showBranches = isDone;
  const editLink = isDone && index === 0
    ? `<button class="node-edit" data-step-idx="${index}">编辑提问 →</button>`
    : '';

  const title = step?.title || `步骤 ${index + 1}`;
  const desc = step?.instruction || '';

  return `
    <div class="${cls}" data-step-idx="${index}">
      <div class="node-rail">
        <span class="node-dot"></span>
        ${!isLast ? '<span class="node-line"></span>' : ''}
      </div>
      <div class="node-card">
        <header class="node-head">
          <span class="node-icon">${nodeIconSvg(meta.iconPath)}</span>
          <span class="node-title">${escapeHtml(title)}</span>
          <span class="node-kind-tag tone-${meta.tone}">${meta.label}</span>
          ${statusLabel}
        </header>
        ${desc ? `<div class="node-desc">${escapeHtml(desc)}</div>` : ''}
        ${isRunning && !isDone ? `<div class="node-thinking"><span></span><span></span><span></span></div>` : ''}
        ${output ? renderNodeBody(item) : ''}
        ${showBranches ? `<div class="node-branches">${renderBranches(item, expanded)}</div>` : ''}
        ${editLink}
      </div>
    </div>`;
}

function firstPendingIndex() {
  for (const it of state.steps) {
    if (it.status !== 'completed') return it.index;
  }
  return -1;
}

function renderTerminus() {
  if (state.isStreaming || state.steps.length === 0) return '';
  const allDone = state.steps.every(it => it.status === 'completed');
  if (!allDone) return '';
  return `
    <div class="node node-terminus">
      <div class="node-rail"><span class="node-dot terminus-dot"></span></div>
      <div class="node-card terminus-card">
        <span>流程已完成</span>
      </div>
    </div>`;
}

function renderEmpty() {
  return `
    <div class="graph-empty">
      <div class="graph-empty-icon">
        <svg viewBox="0 0 32 32" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="10" cy="10" r="3"/>
          <circle cx="22" cy="10" r="3"/>
          <circle cx="16" cy="22" r="3"/>
          <path d="M12.5 11.5L19.5 19M19.5 11.5L12.5 19"/>
        </svg>
      </div>
      <p class="graph-empty-title">动态思考流程</p>
      <p class="graph-empty-sub">发送消息后，AI 会先规划<br>需要哪几步，然后逐步执行</p>
    </div>`;
}

function renderPlanningPlaceholder() {
  return `
    <div class="node node-running">
      <div class="node-rail"><span class="node-dot"></span><span class="node-line"></span></div>
      <div class="node-card">
        <header class="node-head">
          <span class="node-title">规划流程中</span>
          <span class="node-status">规划中</span>
        </header>
        <div class="node-desc">AI 正在判断这个问题需要分几步思考…</div>
        <div class="node-thinking"><span></span><span></span><span></span></div>
      </div>
    </div>`;
}

// ---- main render ----------------------------------------------------

export function renderGraph() {
  const container = document.getElementById('graph-container');
  if (!container) return;

  if (!state.expandedBranches) state.expandedBranches = {};

  if (!state.isStreaming && state.steps.length === 0 && state.messages.length === 0) {
    container.innerHTML = renderEmpty();
    return;
  }

  let html = '<div class="thinking-flow">';

  if (state.steps.length === 0 && state.isStreaming) {
    html += renderPlanningPlaceholder();
  } else {
    for (let i = 0; i < state.steps.length; i++) {
      const item = state.steps[i];
      const isLast = i === state.steps.length - 1 && !state.isStreaming;
      html += renderNode(item, isLast);
    }
  }

  html += renderTerminus();
  html += '</div>';

  container.innerHTML = html;
  attachHandlers();
}

export function resetGraph() {
  const container = document.getElementById('graph-container');
  if (container) container.innerHTML = renderEmpty();
}

// ---- handlers -------------------------------------------------------

function attachHandlers() {
  document.querySelectorAll('.branches-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = parseInt(btn.dataset.stepIdx);
      state.expandedBranches[i] = !state.expandedBranches[i];
      renderGraph();
    });
  });

  document.querySelectorAll('.branch-cta').forEach(btn => {
    btn.addEventListener('click', async () => {
      const i = parseInt(btn.dataset.stepIdx);
      const kind = btn.dataset.kind;
      let payload = {};
      try { payload = JSON.parse(btn.dataset.payload || '{}'); } catch {}

      btn.disabled = true;
      const original = btn.textContent;
      btn.textContent = '执行中…';
      try {
        if (kind === 'alt' || kind === 'primary') {
          await runFromAlternative({
            stepIndex: i,
            overrideState: { primary: payload.text, alt_idx: payload.idx },
            applyLocal: () => {
              const it = state.steps[i];
              if (it && it.output) {
                it.output.primary = payload.text;
                it.selectedAltIdx = payload.idx;
              }
            },
          });
        } else if (kind === 'style') {
          await runFromAlternative({
            stepIndex: i,
            overrideState: { style_hint: payload.id || '' },
            applyLocal: () => {
              const it = state.steps[i];
              if (it) it.selectedStyleKey = payload.id ? `style-${payload.id}` : 'default';
            },
          });
        }
      } catch (e) {
        btn.disabled = false;
        btn.textContent = original;
      }
    });
  });

  document.querySelectorAll('.node-edit').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = parseInt(btn.dataset.stepIdx);
      const it = state.steps[i];
      if (it?.checkpointId) {
        openModificationPanel(it.step?.title || '步骤', it, it.checkpointId);
      }
    });
  });
}
