/**
 * The object dock: one row per detection, carrying every number the pipeline
 * produced -- class, confidence, pixel position, local ENU metres, WGS84
 * lat/lon, distance, bearing and estimated size.
 *
 * The table is the ledger that ties the views together: clicking a row selects
 * the same object on the image, the map and in 3D, and a category filter here
 * dims it everywhere.
 */

import { classColor, categoryOf } from './palette.js';

export class ObjectPanel {
  constructor({
    tbody, countEl, filterInput, emptyEl, filtersEl,
    onSelect, onFocus, onFilterChange,
  }) {
    this.tbody = tbody;
    this.countEl = countEl;
    this.filterInput = filterInput;
    this.emptyEl = emptyEl;
    this.filtersEl = filtersEl;
    this.onSelect = onSelect || (() => {});
    this.onFocus = onFocus || (() => {});
    this.onFilterChange = onFilterChange || (() => {});

    this.detections = [];
    this.selectedIds = new Set();
    this.visibleIds = null;      // externally imposed (assistant selection)
    this.category = 'all';
    this.search = '';
    this.sortKey = null;
    this.sortDir = 1;

    filterInput.addEventListener('input', () => {
      this.search = filterInput.value.trim().toLowerCase();
      this.render();
      this.onFilterChange();
    });

    // Sortable headers, in the order the columns appear.
    const keys = [
      'id', 'class_name', 'confidence', 'bbox.cx', 'bbox.cy',
      'world.east_m', 'world.north_m', 'world.lat', 'world.lon',
      'world.ground_range_m', 'world.bearing_deg', 'world.est_width_m',
    ];
    const head = tbody.closest('table').querySelector('thead tr');
    [...head.children].forEach((th, i) => {
      th.dataset.base = th.textContent;
      th.addEventListener('click', () => this._sortBy(keys[i], th, head));
    });
  }

  /** Build the category filter chips from the backend's taxonomy. */
  buildFilters(categories) {
    this.filtersEl.innerHTML = '';
    const make = (id, label) => {
      const btn = document.createElement('button');
      btn.className = 'fchip' + (id === this.category ? ' on' : '');
      btn.dataset.cat = id;
      btn.textContent = label;
      btn.addEventListener('click', () => this.setCategory(id));
      this.filtersEl.appendChild(btn);
      return btn;
    };

    make('all', 'All');
    for (const cat of categories.filter((c) => c.primary)) make(cat.id, cat.label);
    this._allCategories = categories;
  }

  /**
   * Only offer categories that this scene actually contains, so the operator is
   * not clicking through chips that can never match anything.
   */
  refreshFilterCounts() {
    const present = new Set(this.detections.map((d) => categoryOf(d.class_name)));
    for (const chip of this.filtersEl.children) {
      const id = chip.dataset.cat;
      const has = id === 'all' || present.has(id);
      chip.disabled = !has;
      chip.style.display = has ? '' : 'none';
    }
  }

  setCategory(id) {
    this.category = id;
    for (const chip of this.filtersEl.children) {
      chip.classList.toggle('on', chip.dataset.cat === id);
    }
    this.render();
    this.onFilterChange();
  }

  _sortBy(key, th, head) {
    if (this.sortKey === key) this.sortDir *= -1;
    else { this.sortKey = key; this.sortDir = 1; }

    for (const h of head.children) h.textContent = h.dataset.base;
    th.textContent = `${th.dataset.base} ${this.sortDir > 0 ? '▲' : '▼'}`;
    this.render();
  }

  // -------------------------------------------------------------------- data

  setData(detections) {
    this.detections = detections || [];
    this.refreshFilterCounts();
    this.render();
  }

  setSelected(ids) {
    this.selectedIds = new Set(ids || []);
    for (const tr of this.tbody.children) {
      tr.classList.toggle('sel', this.selectedIds.has(Number(tr.dataset.id)));
    }
    const first = [...this.selectedIds][0];
    if (first !== undefined) {
      this.tbody.querySelector(`tr[data-id="${first}"]`)
        ?.scrollIntoView({ block: 'nearest' });
    }
  }

  /** An externally imposed visibility set, e.g. from the assistant. */
  setVisible(ids) {
    this.visibleIds = ids ? new Set(ids) : null;
    this.render();
  }

  clear() {
    this.detections = [];
    this.selectedIds.clear();
    this.visibleIds = null;
    this.tbody.innerHTML = '';
    this.countEl.textContent = '0';
    this.emptyEl.hidden = false;
  }

  /** The ids currently passing every filter -- what the other views should show. */
  visibleIdSet() {
    const rows = this._rows();
    // No filtering at all means "no restriction", which the views read as null.
    if (this.category === 'all' && !this.search && this.visibleIds === null) return null;
    return new Set(rows.map((d) => d.id));
  }

  // ----------------------------------------------------------------- filtering

  static _get(obj, path) {
    return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
  }

  _rows() {
    let rows = this.detections;

    if (this.visibleIds !== null) {
      rows = rows.filter((d) => this.visibleIds.has(d.id));
    }
    if (this.category !== 'all') {
      rows = rows.filter((d) => categoryOf(d.class_name) === this.category);
    }
    if (this.search) {
      rows = rows.filter((d) => d.class_name.toLowerCase().includes(this.search));
    }

    if (this.sortKey) {
      const key = this.sortKey;
      rows = [...rows].sort((a, b) => {
        const va = ObjectPanel._get(a, key);
        const vb = ObjectPanel._get(b, key);
        if (typeof va === 'string') return va.localeCompare(vb) * this.sortDir;
        return ((va ?? 0) - (vb ?? 0)) * this.sortDir;
      });
    }
    return rows;
  }

  // ------------------------------------------------------------------ render

  render() {
    const rows = this._rows();
    this.countEl.textContent = String(rows.length);
    this.emptyEl.hidden = rows.length > 0;

    const frag = document.createDocumentFragment();

    for (const det of rows) {
      const w = det.world;
      const tr = document.createElement('tr');
      tr.dataset.id = String(det.id);
      if (this.selectedIds.has(det.id)) tr.classList.add('sel');
      if (!w.valid) tr.classList.add('nofix');

      const name = det.track_id != null
        ? `${det.class_name}_${String(det.track_id).padStart(2, '0')}`
        : det.class_name;

      const cells = [
        `<td class="idx">${det.id}</td>`,
        `<td><div class="obj">
           <span class="sw" style="background:${classColor(det.class_id)}"></span>
           <span class="nm">${name}</span>
           ${w.valid && w.source === 'lidar'
             ? `<span class="tag tag-ok" title="Measured from ${w.lidar_points} real LiDAR returns">LIDAR</span>`
             : ''}
           ${det.is_ground_class || (w.valid && w.source === 'lidar') ? '' :
             '<span class="tag tag-warn" title="This class does not normally rest on the ground, so the flat-ground assumption is weaker here">OFF-GND</span>'}
         </div></td>`,
        `<td class="n"><div class="conf">
           <span class="conf-rail"><span class="conf-fill" style="width:${(det.confidence * 100).toFixed(0)}%"></span></span>
           <span class="conf-v">${det.confidence.toFixed(2)}</span>
         </div></td>`,
        `<td class="n">${det.bbox.cx.toFixed(0)}</td>`,
        `<td class="n">${det.bbox.cy.toFixed(0)}</td>`,
      ];

      if (w.valid) {
        cells.push(
          `<td class="n">${w.east_m.toFixed(1)}</td>`,
          `<td class="n">${w.north_m.toFixed(1)}</td>`,
          `<td class="n">${w.lat.toFixed(6)}</td>`,
          `<td class="n">${w.lon.toFixed(6)}</td>`,
          `<td class="n">${w.ground_range_m.toFixed(1)}</td>`,
          `<td class="n">${w.bearing_deg.toFixed(0)}</td>`,
          `<td class="n">${w.est_width_m.toFixed(1)}\u00d7${w.est_height_m.toFixed(1)}</td>`,
        );
      } else {
        // One honest cell rather than seven columns of zeros.
        cells.push(
          `<td class="n" colspan="7" title="${w.reason}">
             <span class="tag tag-warn">NO FIX</span>
           </td>`,
        );
      }

      tr.innerHTML = cells.join('');
      tr.addEventListener('click', (e) => this.onSelect(det.id, e));
      tr.addEventListener('dblclick', () => this.onFocus(det.id));
      frag.appendChild(tr);
    }

    this.tbody.replaceChildren(frag);
  }
}
