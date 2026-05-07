/**
 * Onboarding tour — a small 4-step intro to the agent's main ideas.
 * Lightweight: no element highlighting, just centered cards with mini SVG.
 */

const STORAGE_KEY = 'helpdesk_onboarded_v1';

const STEPS = [
  {
    title: '一个会展示思考的助手',
    body: '这不是一个只给你结果的对话框。它会把每一步思考拆开摆出来 —— 你能看到它怎么理解你的问题、又是怎么得出回答的。',
    art: 'intro',
  },
  {
    title: '看见每一步过程',
    body: '左侧面板会按节点逐步展开：「问题分析」 → 「生成回答」。每个节点完成后都会留下一个检查点，可以回到任意一步。',
    art: 'flow',
  },
  {
    title: '切换不同的思路',
    body: '每个节点都保留了 agent 考虑过的其他角度。展开「备选」换一个，就会从那里重新跑。也可以让回答更简洁、更详细，或者分步骤。',
    art: 'branches',
  },
  {
    title: '随时上手',
    body: '输入栏会基于你已经写下的内容预测可能想问的问题。需要再看本指引时，点右上角「介绍」即可。',
    art: 'input',
  },
];

// ---- SVG illustrations ---------------------------------------------

function art(kind) {
  switch (kind) {
    case 'intro':
      return `
        <svg viewBox="0 0 220 100" width="100%" height="100" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <g stroke-width="1.4" opacity="0.9">
            <circle cx="58" cy="50" r="14"/>
            <circle cx="110" cy="34" r="10" stroke-dasharray="2 3"/>
            <circle cx="110" cy="66" r="10" stroke-dasharray="2 3"/>
            <circle cx="162" cy="50" r="14"/>
            <path d="M70 46 L100 36 M70 54 L100 64 M120 36 L152 46 M120 64 L152 54"/>
          </g>
          <text x="58" y="54" font-size="11" font-family="ui-sans-serif" text-anchor="middle" fill="currentColor" stroke="none">问</text>
          <text x="162" y="54" font-size="11" font-family="ui-sans-serif" text-anchor="middle" fill="currentColor" stroke="none">答</text>
        </svg>`;
    case 'flow':
      return `
        <svg viewBox="0 0 220 110" width="100%" height="110" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <g stroke-width="1.4">
            <circle cx="46" cy="28" r="5" fill="currentColor"/>
            <line x1="46" y1="33" x2="46" y2="78" stroke-dasharray="2 3"/>
            <circle cx="46" cy="83" r="5" fill="currentColor"/>
          </g>
          <g stroke-width="1" opacity="0.4">
            <rect x="68" y="14" rx="6" width="130" height="28"/>
            <rect x="68" y="68" rx="6" width="130" height="28"/>
          </g>
          <g fill="currentColor" stroke="none" font-family="ui-sans-serif" font-size="10">
            <text x="80" y="32">问题分析</text>
            <text x="80" y="86">生成回答</text>
          </g>
          <g fill="currentColor" stroke="none" font-family="ui-sans-serif" font-size="9" opacity="0.5">
            <text x="170" y="32">完成</text>
            <text x="170" y="86">完成</text>
          </g>
        </svg>`;
    case 'branches':
      return `
        <svg viewBox="0 0 220 110" width="100%" height="110" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <g stroke-width="1.4">
            <circle cx="34" cy="55" r="5" fill="currentColor"/>
            <path d="M40 55 L72 55"/>
          </g>
          <g stroke-width="1" opacity="0.5">
            <path d="M72 55 L92 22 M72 55 L92 55 M72 55 L92 88"/>
            <circle cx="92" cy="22" r="3"/>
            <circle cx="92" cy="55" r="3" fill="currentColor"/>
            <circle cx="92" cy="88" r="3"/>
          </g>
          <g stroke-width="1" opacity="0.5">
            <rect x="100" y="12" rx="5" width="100" height="20"/>
            <rect x="100" y="45" rx="5" width="100" height="20"/>
            <rect x="100" y="78" rx="5" width="100" height="20"/>
          </g>
          <g fill="currentColor" stroke="none" font-family="ui-sans-serif" font-size="9">
            <text x="108" y="25" opacity="0.6">备选思路 1</text>
            <text x="108" y="58">主路径</text>
            <text x="108" y="91" opacity="0.6">备选思路 2</text>
          </g>
        </svg>`;
    case 'input':
      return `
        <svg viewBox="0 0 220 110" width="100%" height="110" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
          <g stroke-width="1" opacity="0.55">
            <rect x="20" y="20" rx="6" width="180" height="22"/>
            <rect x="20" y="48" rx="6" width="180" height="18"/>
            <rect x="20" y="70" rx="6" width="180" height="18"/>
            <rect x="20" y="92" rx="6" width="180" height="14" stroke-dasharray="2 3"/>
          </g>
          <g fill="currentColor" stroke="none" font-family="ui-sans-serif" font-size="9">
            <text x="28" y="34" opacity="0.5">如何重置</text>
            <text x="28" y="60">如何重置 Windows 开机密码？</text>
            <text x="28" y="82">如何重置邮箱密码并启用 MFA？</text>
            <text x="28" y="102" opacity="0.5">AI 预测中…</text>
          </g>
        </svg>`;
  }
  return '';
}

// ---- DOM ------------------------------------------------------------

let currentStep = 0;
let rootEl = null;

function ensureRoot() {
  if (rootEl) return rootEl;
  rootEl = document.createElement('div');
  rootEl.className = 'onboarding';
  rootEl.id = 'onboarding';
  document.body.appendChild(rootEl);
  return rootEl;
}

function render() {
  const el = ensureRoot();
  const step = STEPS[currentStep];
  const total = STEPS.length;
  const isLast = currentStep === total - 1;
  const isFirst = currentStep === 0;

  const dots = STEPS.map((_, i) =>
    `<span class="ob-dot${i === currentStep ? ' active' : ''}${i < currentStep ? ' done' : ''}"></span>`
  ).join('');

  el.innerHTML = `
    <div class="ob-overlay"></div>
    <div class="ob-card" role="dialog" aria-modal="true" aria-labelledby="ob-title">
      <div class="ob-head">
        <div class="ob-dots">${dots}</div>
        <button class="ob-close" id="ob-close" aria-label="关闭">
          <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M3 3l10 10M13 3L3 13"/></svg>
        </button>
      </div>
      <div class="ob-art" key="${step.art}">${art(step.art)}</div>
      <h2 id="ob-title" class="ob-title">${step.title}</h2>
      <p class="ob-body">${step.body}</p>
      <div class="ob-foot">
        <button class="ob-skip" id="ob-skip">跳过</button>
        <div class="ob-nav">
          ${!isFirst ? `<button class="ob-btn ob-btn-ghost" id="ob-prev">上一步</button>` : ''}
          <button class="ob-btn ob-btn-primary" id="ob-next">${isLast ? '开始使用' : '下一步'}</button>
        </div>
      </div>
    </div>`;

  el.classList.add('open');
  bindHandlers();

  // focus next button for keyboard accessibility
  const nextBtn = document.getElementById('ob-next');
  if (nextBtn) nextBtn.focus();
}

function bindHandlers() {
  document.getElementById('ob-close')?.addEventListener('click', close);
  document.getElementById('ob-skip')?.addEventListener('click', close);
  document.getElementById('ob-prev')?.addEventListener('click', () => {
    currentStep = Math.max(0, currentStep - 1);
    render();
  });
  document.getElementById('ob-next')?.addEventListener('click', () => {
    if (currentStep === STEPS.length - 1) {
      close();
    } else {
      currentStep += 1;
      render();
    }
  });

  // Close on overlay click
  rootEl.querySelector('.ob-overlay')?.addEventListener('click', close);
}

function close() {
  if (!rootEl) return;
  rootEl.classList.remove('open');
  setTimeout(() => {
    if (rootEl) rootEl.innerHTML = '';
  }, 220);
  try { localStorage.setItem(STORAGE_KEY, '1'); } catch {}
  document.removeEventListener('keydown', onKey);
}

function onKey(e) {
  if (!rootEl?.classList.contains('open')) return;
  if (e.key === 'Escape') {
    close();
  } else if (e.key === 'ArrowRight') {
    if (currentStep < STEPS.length - 1) { currentStep += 1; render(); }
  } else if (e.key === 'ArrowLeft') {
    if (currentStep > 0) { currentStep -= 1; render(); }
  }
}

// ---- Public --------------------------------------------------------

export function openOnboarding() {
  currentStep = 0;
  render();
  document.addEventListener('keydown', onKey);
}

export function maybeShowOnboarding() {
  let seen = false;
  try { seen = !!localStorage.getItem(STORAGE_KEY); } catch {}
  if (seen) return;
  // Slight delay so it appears after the first paint (smoother feel)
  setTimeout(openOnboarding, 320);
}
