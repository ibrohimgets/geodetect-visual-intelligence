/**
 * The timeline: transport controls plus one row per tracked object.
 *
 * This is the piece borrowed most directly from tools like Foxglove and Rerun,
 * and it is borrowed because it answers a question none of the other panes can:
 * *when* was each thing visible. A map shows where objects are now; a table
 * shows what is in this frame. Only a stream view shows that `car_03` was
 * tracked for the whole drive while `person_21` appeared for a tenth of a
 * second — which is exactly how you spot a flickering detection.
 *
 * It spans the full window rather than living inside the centre column, because
 * time is global: scrubbing it moves every pane at once.
 *
 * Each row draws the track's visible span, with a tick per observation, so gaps
 * where the tracker lost the object are visible as holes rather than smoothed
 * over.
 */

import { classColor } from './palette.js';

export class Timeline {
  constructor({
    root, labelsEl, tracksEl, rulerEl, rowsEl, playheadEl,
    clockEl, subEl, tracksCountEl, toggleEl,
    onSeek, onSelectTrack,
  }) {
    this.root = root;
    this.labelsEl = labelsEl;
    this.tracksEl = tracksEl;
    this.rulerEl = rulerEl;
    this.rowsEl = rowsEl;
    this.playheadEl = playheadEl;
    this.clockEl = clockEl;
    this.subEl = subEl;
    this.tracksCountEl = tracksCountEl;

    this.onSeek = onSeek || (() => {});
    this.onSelectTrack = onSelectTrack || (() => {});

    this.duration = 0;
    this.trails = [];
    this.selectedTrack = null;
    this.rows = new Map();   // track_id -> { row, label }

    // Click or drag anywhere on the track area to scrub.
    let scrubbing = false;
    const seekFromEvent = (event) => {
      if (!this.duration) return;
      const rect = tracksEl.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      this.onSeek(ratio * this.duration);
    };
    tracksEl.addEventListener('pointerdown', (e) => {
      scrubbing = true;
      tracksEl.setPointerCapture(e.pointerId);
      seekFromEvent(e);
    });
    tracksEl.addEventListener('pointermove', (e) => { if (scrubbing) seekFromEvent(e); });
    tracksEl.addEventListener('pointerup', (e) => {
      scrubbing = false;
      try { tracksEl.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    });

    // Keep the label gutter aligned with the tracks when either scrolls.
    tracksEl.addEventListener('scroll', () => { labelsEl.scrollTop = tracksEl.scrollTop; });
    labelsEl.addEventListener('scroll', () => { tracksEl.scrollTop = labelsEl.scrollTop; });

    toggleEl?.addEventListener('click', () => this.toggle());
  }

  toggle(force) {
    const collapsed = force !== undefined ? force : !this.root.classList.contains('collapsed');
    this.root.classList.toggle('collapsed', collapsed);
  }

  /** Give the timeline a sequence to display. */
  setSequence({ duration, trails, label }) {
    this.duration = duration || 0;
    this.trails = (trails || []).filter((t) => t.points?.length);
    this.subEl.textContent = label || '';
    this.tracksCountEl.textContent = this.trails.length
      ? `${this.trails.length} tracks` : '';

    this.root.classList.toggle('collapsed', this.trails.length === 0);
    this.playheadEl.hidden = !this.duration;

    this._buildRuler();
    this._buildRows();
  }

  clear() {
    this.duration = 0;
    this.trails = [];
    this.rows.clear();
    this.labelsEl.innerHTML = '';
    this.rowsEl.innerHTML = '';
    this.rulerEl.innerHTML = '';
    this.playheadEl.hidden = true;
    this.clockEl.textContent = '--:--.---';
    this.subEl.textContent = 'no sequence';
    this.tracksCountEl.textContent = '';
    this.root.classList.add('collapsed');
  }

  // ------------------------------------------------------------------ build

  /** A time ruler with ticks at a round interval that fits the duration. */
  _buildRuler() {
    this.rulerEl.innerHTML = '';
    if (!this.duration) return;

    // Aim for roughly 8 labelled ticks, snapped to a human interval.
    const candidates = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];
    const step = candidates.find((s) => this.duration / s <= 10) || 60;

    for (let t = 0; t <= this.duration + 1e-6; t += step) {
      const tick = document.createElement('div');
      tick.className = 'tl-tick';
      tick.style.left = `${(t / this.duration) * 100}%`;
      tick.textContent = t < 10 ? `${t.toFixed(1)}s` : `${t.toFixed(0)}s`;
      this.rulerEl.appendChild(tick);
    }
  }

  _buildRows() {
    this.labelsEl.innerHTML = '';
    this.rowsEl.innerHTML = '';
    this.rows.clear();

    if (!this.trails.length) {
      this.rowsEl.innerHTML = '<div class="tl-empty">no tracked objects</div>';
      return;
    }

    // Longest-lived first: the tracks worth looking at are the persistent ones.
    const ordered = [...this.trails].sort(
      (a, b) => (b.last_seen_s - b.first_seen_s) - (a.last_seen_s - a.first_seen_s),
    );

    for (const trail of ordered) {
      const color = classColor(trail.class_id);

      const label = document.createElement('div');
      label.className = 'tl-label';
      label.dataset.track = String(trail.track_id);
      label.innerHTML =
        `<span class="sw" style="background:${color}"></span>`
        + `<span class="nm">${trail.label}</span>`;
      label.title =
        `${trail.label}\n`
        + `visible ${trail.first_seen_s.toFixed(1)}s – ${trail.last_seen_s.toFixed(1)}s\n`
        + `${trail.points.length} observations\n`
        + `travelled ${trail.distance_m ?? 0} m (net ${trail.displacement_m ?? 0} m)`;
      label.addEventListener('click', () => this.onSelectTrack(trail.track_id));
      this.labelsEl.appendChild(label);

      const row = document.createElement('div');
      row.className = 'tl-row';
      row.dataset.track = String(trail.track_id);

      const left = (trail.first_seen_s / this.duration) * 100;
      const width = Math.max(
        0.4, ((trail.last_seen_s - trail.first_seen_s) / this.duration) * 100,
      );
      const span = document.createElement('div');
      span.className = 'tl-span';
      span.style.cssText = `left:${left}%;width:${width}%;background:${color}`;
      row.appendChild(span);

      // One tick per observation: gaps show where tracking dropped out.
      for (const point of trail.points) {
        const tick = document.createElement('div');
        tick.className = 'tl-sample';
        tick.style.left = `${(point.t / this.duration) * 100}%`;
        row.appendChild(tick);
      }

      row.addEventListener('click', (e) => {
        // A click on a row still scrubs; only the label selects.
        e.stopPropagation();
        this.onSelectTrack(trail.track_id);
        const rect = this.tracksEl.getBoundingClientRect();
        this.onSeek(((e.clientX - rect.left) / rect.width) * this.duration);
      });

      this.rowsEl.appendChild(row);
      this.rows.set(trail.track_id, { row, label });
    }
  }

  // ----------------------------------------------------------------- update

  /** Move the playhead and the clock. Called on every timeupdate. */
  setTime(seconds, frameIndex = null, frameCount = null) {
    if (!this.duration) return;
    const ratio = Math.max(0, Math.min(1, seconds / this.duration));
    this.playheadEl.style.left = `${ratio * 100}%`;

    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    const clock = `${String(minutes).padStart(2, '0')}:${rest.toFixed(3).padStart(6, '0')}`;
    this.clockEl.textContent = clock;

    if (frameIndex !== null && frameCount) {
      this.clockEl.title = `frame ${frameIndex + 1} of ${frameCount}`;
    }
  }

  /** Highlight the row for the currently selected object. */
  setSelectedTrack(trackId) {
    this.selectedTrack = trackId;
    for (const [id, { row, label }] of this.rows) {
      const on = id === trackId;
      row.classList.toggle('sel', on);
      label.classList.toggle('sel', on);
    }
  }
}
