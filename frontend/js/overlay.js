/**
 * The DETECTION view: the image or video frame with boxes drawn over it.
 *
 * The canvas is sized to the media's real pixel dimensions and scaled down by
 * CSS to fit the pane. Every coordinate we draw is therefore in the same pixel
 * space the backend reported, with no conversion scattered through the drawing
 * code -- only pointer events need mapping back, and that happens in one place.
 *
 * Images draw once. Video redraws on a frame loop while playing, because the
 * underlying <video> element is a moving picture and the boxes have to keep up.
 */

import { classColor, withAlpha, SELECTED_COLOR } from './palette.js';

export class DetectionOverlay {
  constructor(canvas, { onSelect } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSelect = onSelect || (() => {});

    this.source = null;           // { kind: 'image' | 'video', el }
    this.detections = [];
    this.selectedIds = new Set();
    this.visibleIds = null;       // null means "no filter, everything visible"
    this.options = { anchors: true, labels: true };

    this._loop = null;

    canvas.addEventListener('click', (e) => this._handleClick(e));
    canvas.addEventListener('mousemove', (e) => this._handleHover(e));
    canvas.addEventListener('mouseleave', () => { canvas.style.cursor = 'crosshair'; });
  }

  // ------------------------------------------------------------------ source

  /** Load a still image. Resolves once it is decoded and drawn. */
  async setImage(url) {
    this.stopLoop();
    this._detachVideo();
    const img = new Image();
    img.src = url;
    await img.decode();

    this.source = { kind: 'image', el: img };
    this.canvas.width = img.naturalWidth;
    this.canvas.height = img.naturalHeight;
    this.canvas.classList.add('ready');
    this.draw();
    return { width: img.naturalWidth, height: img.naturalHeight };
  }

  /** Attach a <video> element as the source. */
  setVideo(videoEl) {
    this.stopLoop();
    this._detachVideo();

    this.source = { kind: 'video', el: videoEl };
    this.canvas.width = videoEl.videoWidth || 1280;
    this.canvas.height = videoEl.videoHeight || 720;
    this.canvas.classList.add('ready');

    // A freshly loaded, still-paused video reports readyState 4 before it has
    // actually painted anything, so this first drawImage can come back empty
    // and -- because nothing else would redraw a paused video -- stay empty
    // until the operator pressed play. Redraw on the events that do guarantee a
    // decoded frame, and once more on the next couple of animation frames.
    this._videoRedraw = () => this.draw();
    for (const event of ['loadeddata', 'canplay', 'seeked']) {
      videoEl.addEventListener(event, this._videoRedraw);
    }
    requestAnimationFrame(() => {
      this.draw();
      requestAnimationFrame(() => this.draw());
    });

    this.draw();
  }

  /** Drop the redraw listeners from a previous source. */
  _detachVideo() {
    if (!this._videoRedraw || this.source?.kind !== 'video') return;
    for (const event of ['loadeddata', 'canplay', 'seeked']) {
      this.source.el.removeEventListener(event, this._videoRedraw);
    }
    this._videoRedraw = null;
  }

  /** Redraw continuously -- used while a video is playing. */
  startLoop() {
    if (this._loop) return;
    const tick = () => {
      this._loop = requestAnimationFrame(tick);
      this.draw();
    };
    this._loop = requestAnimationFrame(tick);
  }

  stopLoop() {
    if (this._loop) {
      cancelAnimationFrame(this._loop);
      this._loop = null;
    }
  }

  // ------------------------------------------------------------------ state

  setDetections(detections) {
    this.detections = detections || [];
    this.draw();
  }

  setSelected(ids) {
    this.selectedIds = new Set(ids || []);
    this.draw();
  }

  /** Restrict what is drawn solidly. Pass null to clear the filter. */
  setVisible(ids) {
    this.visibleIds = ids ? new Set(ids) : null;
    this.draw();
  }

  setOptions(opts) {
    Object.assign(this.options, opts);
    this.draw();
  }

  clear() {
    this.stopLoop();
    this._detachVideo();
    this.source = null;
    this.detections = [];
    this.selectedIds.clear();
    this.visibleIds = null;
    this.canvas.classList.remove('ready');
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // ----------------------------------------------------------------- drawing

  draw() {
    const { ctx, canvas, source } = this;
    if (!source) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    try {
      ctx.drawImage(source.el, 0, 0, canvas.width, canvas.height);
    } catch {
      return; // a video frame that is not decodable yet
    }

    // Scale strokes and type with the media, so a 4000px photo does not get
    // hairline boxes and a 640px one does not get slabs.
    const k = Math.max(1, Math.min(canvas.width, canvas.height) / 640);
    const lineW = 2 * k;
    const fontSize = Math.round(13 * k);
    const anySelected = this.selectedIds.size > 0;

    for (const det of this.detections) {
      const selected = this.selectedIds.has(det.id);
      const filteredOut = this.visibleIds !== null && !this.visibleIds.has(det.id);
      // Dim anything filtered out, and anything unselected while a selection
      // is active. The selected object should be the only thing that pops.
      const faded = filteredOut || (anySelected && !selected);

      const color = selected ? SELECTED_COLOR : classColor(det.class_id);
      const { x1, y1, x2, y2, anchor_x, anchor_y } = det.bbox;
      const w = x2 - x1;
      const h = y2 - y1;

      ctx.globalAlpha = faded ? 0.22 : 1;

      // --- box ---
      ctx.lineWidth = selected ? lineW * 1.7 : lineW;
      ctx.strokeStyle = color;
      ctx.setLineDash(det.world.valid ? [] : [6 * k, 4 * k]); // dashed = no ground fix
      ctx.strokeRect(x1, y1, w, h);
      ctx.setLineDash([]);

      ctx.fillStyle = withAlpha(color, selected ? 0.2 : 0.08);
      ctx.fillRect(x1, y1, w, h);

      // --- corner ticks, the way a CAD tool marks a selection ---
      const tick = Math.min(w, h) * 0.18;
      ctx.lineWidth = lineW * 1.4;
      ctx.beginPath();
      ctx.moveTo(x1, y1 + tick); ctx.lineTo(x1, y1); ctx.lineTo(x1 + tick, y1);
      ctx.moveTo(x2 - tick, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + tick);
      ctx.moveTo(x2, y2 - tick); ctx.lineTo(x2, y2); ctx.lineTo(x2 - tick, y2);
      ctx.moveTo(x1 + tick, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - tick);
      ctx.stroke();

      // --- ground anchor: the pixel we actually geo-reference ---
      if (this.options.anchors && det.world.valid && !faded) {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineW * 0.8;
        ctx.setLineDash([3 * k, 3 * k]);
        ctx.beginPath();
        ctx.moveTo(anchor_x, y1);
        ctx.lineTo(anchor_x, anchor_y);
        ctx.stroke();
        ctx.setLineDash([]);

        const r = 5 * k;
        ctx.beginPath();
        ctx.moveTo(anchor_x - r, anchor_y); ctx.lineTo(anchor_x + r, anchor_y);
        ctx.moveTo(anchor_x, anchor_y - r); ctx.lineTo(anchor_x, anchor_y + r);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(anchor_x, anchor_y, r * 0.55, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }

      // --- label ---
      if (this.options.labels && !faded) {
        const name = det.track_id != null
          ? `${det.class_name}_${String(det.track_id).padStart(2, '0')}`
          : det.class_name;
        const range = det.world.valid
          ? `  ${det.world.ground_range_m.toFixed(1)}m`
          : '  no fix';
        const text = `${name} ${(det.confidence * 100).toFixed(0)}%${range}`;

        ctx.font = `600 ${fontSize}px ui-monospace, Menlo, Consolas, monospace`;
        const padX = 5 * k;
        const padY = 3 * k;
        const tw = ctx.measureText(text).width;
        const th = fontSize + padY * 2;
        // Flip the label inside the box when it would run off the top edge.
        const labelY = y1 - th < 0 ? y1 : y1 - th;

        ctx.fillStyle = color;
        ctx.fillRect(x1, labelY, tw + padX * 2, th);
        ctx.fillStyle = '#0b0c0e';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x1 + padX, labelY + th / 2);
      }
    }

    ctx.globalAlpha = 1;
  }

  /** The annotated frame as a PNG blob, for the screenshot export. */
  toBlob() {
    return new Promise((resolve) => this.canvas.toBlob(resolve, 'image/png'));
  }

  // ------------------------------------------------------------- interaction

  _toMediaXY(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * this.canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * this.canvas.height,
    };
  }

  /** Smallest box containing the point, so a box inside a box stays reachable. */
  _hitTest(x, y) {
    let best = null;
    let bestArea = Infinity;
    for (const det of this.detections) {
      if (this.visibleIds !== null && !this.visibleIds.has(det.id)) continue;
      const { x1, y1, x2, y2 } = det.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        const area = (x2 - x1) * (y2 - y1);
        if (area < bestArea) { bestArea = area; best = det; }
      }
    }
    return best;
  }

  _handleClick(event) {
    if (!this.source) return;
    const { x, y } = this._toMediaXY(event);
    const hit = this._hitTest(x, y);
    this.onSelect(hit ? hit.id : null);
  }

  _handleHover(event) {
    if (!this.source) return;
    const { x, y } = this._toMediaXY(event);
    this.canvas.style.cursor = this._hitTest(x, y) ? 'pointer' : 'crosshair';
  }
}
