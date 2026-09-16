/**
 * The Geo Intelligence Assistant panel.
 *
 * A chat surface over the backend's /api/assistant endpoint. Two things make it
 * more than a chat box:
 *
 *   1. It sends the *current scene* with every question -- whatever is on
 *      screen, image or video frame -- so the model reasons over measured
 *      positions rather than guessing from an image it cannot see.
 *   2. Its replies can carry actions. When the model calls a tool, the answer
 *      comes back with instructions like "select these ids", and the panel
 *      hands them to the app, which applies them to every view at once.
 *
 * The tool calls are shown rather than hidden. Seeing `select_objects
 * {classes: [person], near_object_id: 0, near_radius_m: 25} -> 4 matched` is
 * what turns a black box into an instrument.
 */

const SUGGESTION_POOL = [
  { text: 'Summarize this scene', always: true },
  { text: 'Which object is closest to the camera?', always: true },
  { text: 'How many people are visible?', needs: ['person'] },
  { text: 'Show only vehicles', needsCategory: 'vehicles' },
  { text: 'Highlight all people', needs: ['person'] },
  { text: 'Which objects are within 30 meters?', always: true },
  { text: 'Are there any people near vehicles?', needsBoth: ['person', 'vehicles'] },
  { text: 'What is on the left side?', always: true },
  { text: 'Where is the bus?', needs: ['bus'] },
  { text: 'Show me the animals', needsCategory: 'animals' },
  { text: 'Which object is farthest away?', always: true },
];

export class Assistant {
  constructor({
    chatEl, formEl, inputEl, sendBtn, suggestionsEl, modelEl, clearBtn,
    getScene, onActions, onError,
  }) {
    this.chatEl = chatEl;
    this.inputEl = inputEl;
    this.sendBtn = sendBtn;
    this.suggestionsEl = suggestionsEl;
    this.modelEl = modelEl;

    this.getScene = getScene;          // () => { camera, detections }
    this.onActions = onActions || (() => {});
    this.onError = onError || (() => {});

    this.history = [];
    this.busy = false;
    this.available = false;

    formEl.addEventListener('submit', (e) => {
      e.preventDefault();
      this.send(this.inputEl.value);
    });

    // Enter sends, Shift+Enter makes a newline -- the convention people expect.
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.send(this.inputEl.value);
      }
    });

    // Grow the textarea with its content, up to the CSS max-height.
    inputEl.addEventListener('input', () => {
      inputEl.style.height = 'auto';
      inputEl.style.height = `${Math.min(inputEl.scrollHeight, 110)}px`;
    });

    clearBtn?.addEventListener('click', () => this.reset());
  }

  setStatus(status) {
    this.available = !!status?.available;
    this.modelEl.textContent = this.available
      ? status.model
      : 'unavailable — limited fallback';
    this.modelEl.title = this.available ? '' : (status?.reason || '');
  }

  /** Offer questions that actually make sense for what is on screen. */
  updateSuggestions(detections, categoryOf) {
    const classes = new Set((detections || []).map((d) => d.class_name));
    const cats = new Set((detections || []).map((d) => categoryOf(d.class_name)));

    const usable = SUGGESTION_POOL.filter((s) => {
      if (s.always) return detections?.length > 0;
      if (s.needs) return s.needs.every((c) => classes.has(c));
      if (s.needsCategory) return cats.has(s.needsCategory);
      if (s.needsBoth) return classes.has(s.needsBoth[0]) && cats.has(s.needsBoth[1]);
      return false;
    }).slice(0, 5);

    this.suggestionsEl.innerHTML = '';
    for (const s of usable) {
      const chip = document.createElement('button');
      chip.className = 'chip';
      chip.type = 'button';
      chip.textContent = s.text;
      chip.addEventListener('click', () => this.send(s.text));
      this.suggestionsEl.appendChild(chip);
    }
  }

  // ------------------------------------------------------------------ sending

  async send(text) {
    const question = (text || '').trim();
    if (!question || this.busy) return;

    const scene = this.getScene();
    if (!scene || !scene.detections?.length) {
      this.addMessage('bot', 'Load an image or video first — I need a scene to reason about.');
      return;
    }

    this.inputEl.value = '';
    this.inputEl.style.height = 'auto';
    this.addMessage('user', question);
    this._setBusy(true);
    const typing = this._addTyping();

    try {
      const { askAssistant } = await import('./api.js');
      const reply = await askAssistant(
        question, scene.camera, scene.detections, this.history,
      );

      typing.remove();
      if (reply.tool_calls?.length) this._addToolTrace(reply.tool_calls);
      this.addMessage('bot', reply.answer, reply.fallback);

      this.history.push({ role: 'user', content: question });
      this.history.push({ role: 'assistant', content: reply.answer });
      // Keep the context window small; the scene is re-sent each turn anyway.
      if (this.history.length > 16) this.history = this.history.slice(-16);

      if (reply.actions?.length) this.onActions(reply.actions);
    } catch (err) {
      typing.remove();
      this.addMessage('bot', `I could not answer that: ${err.message}`);
      this.onError(err);
    } finally {
      this._setBusy(false);
    }
  }

  _setBusy(busy) {
    this.busy = busy;
    this.sendBtn.disabled = busy;
    this.inputEl.disabled = busy;
    if (!busy) this.inputEl.focus();
  }

  // ------------------------------------------------------------------ render

  addMessage(role, text, isFallback = false) {
    this.chatEl.querySelector('.chat-intro')?.remove();

    const wrap = document.createElement('div');
    wrap.className = `msg msg-${role === 'user' ? 'user' : 'bot'}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = this._format(text);
    if (isFallback) bubble.style.borderStyle = 'dashed';

    wrap.appendChild(bubble);
    this.chatEl.appendChild(wrap);
    this._scroll();
    return wrap;
  }

  _addToolTrace(calls) {
    const el = document.createElement('div');
    el.className = 'tool-trace';
    el.innerHTML = calls.map((c) => {
      const args = Object.entries(c.arguments || {})
        .filter(([, v]) => v !== undefined && v !== null)
        .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(',')}]` : v}`)
        .join(' ');
      const matched = c.matched != null ? ` → ${c.matched} matched` : '';
      return `<div><b>${c.tool}</b> ${this._escape(args)}${matched}</div>`;
    }).join('');
    this.chatEl.appendChild(el);
    this._scroll();
  }

  _addTyping() {
    const el = document.createElement('div');
    el.className = 'msg msg-bot';
    el.innerHTML = '<div class="bubble typing"><i></i><i></i><i></i></div>';
    this.chatEl.appendChild(el);
    this._scroll();
    return el;
  }

  _escape(s) {
    return String(s).replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  /** Escape first, then allow the small amount of markdown the model emits. */
  _format(text) {
    return this._escape(text)
      .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }

  _scroll() {
    this.chatEl.scrollTop = this.chatEl.scrollHeight;
  }

  reset() {
    this.history = [];
    this.chatEl.innerHTML = `
      <div class="chat-intro">
        <p>I reason over the <b>measured</b> scene graph — classes, distances, bearings and coordinates — not over the image itself.</p>
        <p class="dim">Ask a question, or tell me what to highlight. Selections apply to every view at once.</p>
      </div>`;
  }
}
