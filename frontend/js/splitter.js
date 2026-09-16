/**
 * Draggable panel splitters.
 *
 * Each splitter writes a single CSS custom property on the element that owns
 * the grid, and the grid template does the rest. That keeps the layout
 * declarative -- there is no JavaScript setting widths on individual panels,
 * only one number per axis.
 *
 * Sizes are remembered per viewer in localStorage. That is a convenience, not
 * state the app depends on, so every access is wrapped: a private window or
 * blocked site data must not stop the layout working.
 */

const STORE_KEY = 'geodetect.layout.v2';

function readStored() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
  } catch {
    return {}; // private window, blocked storage, or corrupt value
  }
}

function writeStored(patch) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ ...readStored(), ...patch }));
  } catch {
    /* not important enough to surface */
  }
}

/**
 * Wire one splitter.
 *
 * @param {object}  opts
 * @param {Element} opts.handle    the splitter element
 * @param {Element} opts.target    the element carrying the CSS variable
 * @param {string}  opts.prop      the CSS custom property to write
 * @param {string}  opts.axis      'x' or 'y'
 * @param {number}  opts.min       smallest allowed size in px
 * @param {number}  opts.max       largest allowed size in px
 * @param {boolean} opts.invert    true when dragging right should shrink
 * @param {string}  opts.key       localStorage key suffix
 * @param {Function} opts.onResize called during and after a drag
 */
export function makeSplitter({
  handle, target, prop, axis = 'x',
  min = 200, max = 640, invert = false, key, onResize,
}) {
  if (!handle || !target) return;

  const apply = (px) => {
    const clamped = Math.max(min, Math.min(max, px));
    target.style.setProperty(prop, `${clamped}px`);
    return clamped;
  };

  // Restore a remembered size.
  const stored = readStored()[key];
  if (typeof stored === 'number') apply(stored);

  let dragging = false;

  const onPointerMove = (event) => {
    if (!dragging) return;
    const rect = target.getBoundingClientRect();
    const px = axis === 'x'
      ? (invert ? rect.right - event.clientX : event.clientX - rect.left)
      : (invert ? rect.bottom - event.clientY : event.clientY - rect.top);
    apply(px);
    onResize?.();
  };

  const stop = () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.classList.remove('resizing', 'resizing-v');
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', stop);

    const current = parseFloat(target.style.getPropertyValue(prop));
    if (!Number.isNaN(current)) writeStored({ [key]: current });
    onResize?.();
  };

  handle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    dragging = true;
    handle.classList.add('dragging');
    document.body.classList.add(axis === 'x' ? 'resizing' : 'resizing-v');
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', stop);
  });

  // Double click resets to the stylesheet's default.
  handle.addEventListener('dblclick', () => {
    target.style.removeProperty(prop);
    writeStored({ [key]: null });
    onResize?.();
  });

  // Keyboard access: the handle is focusable, so arrows should move it.
  handle.addEventListener('keydown', (event) => {
    const step = event.shiftKey ? 40 : 12;
    const forward = axis === 'x' ? 'ArrowRight' : 'ArrowDown';
    const back = axis === 'x' ? 'ArrowLeft' : 'ArrowUp';
    if (event.key !== forward && event.key !== back) return;

    event.preventDefault();
    const computed = parseFloat(
      getComputedStyle(target).getPropertyValue(prop)
    ) || min;
    const direction = (event.key === forward ? 1 : -1) * (invert ? -1 : 1);
    const next = apply(computed + direction * step);
    writeStored({ [key]: next });
    onResize?.();
  });
}
