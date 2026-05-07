/**
 * API client + SSE helper
 */

const API_BASE = window.location.origin;

export async function streamChat({ message, userId, userName, department, threadId }, signal) {
  const resp = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      user_id: userId || 'anonymous',
      user_name: userName || '',
      user_department: department || '',
      thread_id: threadId || null,
    }),
    signal,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `API error: ${resp.status}`);
  }
  return resp;
}

export async function streamBranch(body, signal) {
  const resp = await fetch(`${API_BASE}/threads/branch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Branch failed: ${resp.status}`);
  }
  return resp;
}

export async function fetchCheckpoints(threadId, limit = 100) {
  const resp = await fetch(`${API_BASE}/threads/${encodeURIComponent(threadId)}/checkpoints?limit=${limit}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch checkpoints: ${resp.status}`);
  }
  return resp.json();
}

/**
 * Iterate through SSE events from a fetch Response.
 * Calls onEvent({type, ...}) for every parsed JSON event.
 */
export async function consumeSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith('data: ')) continue;
        try { onEvent(JSON.parse(t.slice(6))); }
        catch (e) { console.warn('SSE parse error:', e, t); }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
