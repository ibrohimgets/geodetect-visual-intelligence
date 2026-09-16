/**
 * Application shell: owns the state, wires the controls, and keeps every pane
 * showing the same thing.
 *
 * Layout follows the sensor-workbench convention: a source/settings tree on the
 * left, titled panes in the middle, an inspector and assistant on the right, a
 * global timeline across the bottom, and a status bar of live figures. The
 * timeline is global rather than per-pane because time applies to all of them
 * at once -- scrubbing it moves the image, the map and the 3D scene together.
 *
 * Pipeline entry points
 * ---------------------
 *   runImage()   detect + project a still
 *   runVideo()   track a clip, then project every frame
 *   runKitti()   fuse YOLO with real LiDAR, GPS/IMU and calibration
 *   applyData()  hand one payload to every pane
 */

import * as api from './api.js';
import { DetectionOverlay } from './overlay.js';
import { MapView } from './mapview.js';
import { Scene3D } from './scene3d.js';
import { ObjectPanel } from './panel.js';
import { Assistant } from './assistant.js';
import { Timeline } from './timeline.js';
import { makeSplitter } from './splitter.js';
import { primeCategories, categoryOf, classColor } from './palette.js';

const $ = (id) => document.getElementById(id);

const PRESETS = {
  drone_nadir:   { altitude: 80,   pitch: 90, hfov: 84, heading: 0,  maxRange: 400 },
  drone_oblique: { altitude: 60,   pitch: 45, hfov: 78, heading: 0,  maxRange: 500 },
  street:        { altitude: 1.55, pitch: 6,  hfov: 70, heading: 30, maxRange: 300 },
};

const state = {
  mediaKind: 'image',        // image | video | kitti
  file: null,
  data: null,
  imageId: null,
  jobId: null,
  kittiSequence: null,
  calibration: null,
  sensors: null,
  timeline: null,
  videoMeta: null,
  videoCamera: null,
  videoFootprint: [],
  metrics: null,
  trails: [],
  frameIndex: -1,
  selected: new Set(),
  assistantIds: null,
  reprojectSeq: 0,
  busy: false,
  sourceKey: null,
};

let overlay, mapView, scene3d, panel, assistant, timeline;

/* ═════════════════════════ small helpers ═════════════════════════ */

function toast(message, kind = 'info', ms = 4000) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('toasts').appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .2s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 200);
  }, ms);
}

function setBusy(on, text = 'working') {
  state.busy = on;
  $('busy').hidden = !on;
  $('busyText').textContent = text;
  $('runBtn').disabled = on || (!state.file && state.mediaKind !== 'kitti');
}

function setStages(map) {
  for (const [name, value] of Object.entries(map)) {
    const el = document.querySelector(`.stage[data-stage="${name}"]`);
    if (!el || value === undefined) continue;
    el.classList.toggle('done', value === 'done');
    el.classList.toggle('active', value === 'active');
  }
}
const resetStages = () =>
  setStages({ source: '', detect: '', fuse: '', world: '', reason: '' });

function setProgress(on, text = '', pct = 0) {
  $('prog').hidden = !on;
  if (!on) return;
  $('progText').textContent = text;
  $('progPct').textContent = `${Math.round(pct)}%`;
  $('progFill').style.width = `${pct}%`;
}

/* ═════════════════════════ control readout ═════════════════════════ */

function readCamera() {
  const heading = ((Number($('heading').value) % 360) + 360) % 360;
  return {
    hfov_deg: Number($('hfov').value),
    altitude_m: Number($('altitude').value),
    pitch_deg: Number($('pitch').value),
    heading_deg: heading,
    roll_deg: 0,
    lat: Number($('lat').value),
    lon: Number($('lon').value),
    max_range_m: Number($('maxRange').value),
  };
}

const readInference = () => ({
  confidence: Number($('confidence').value),
  iou: Number($('iou').value),
  max_detections: 100,
});

function setPair(name, value) {
  const num = $(name);
  const range = $(`${name}Range`);
  if (num) num.value = value;
  if (range) range.value = value;
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  setPair('altitude', p.altitude);
  setPair('pitch', p.pitch);
  setPair('heading', p.heading);
  setPair('hfov', p.hfov);
  $('maxRange').value = p.maxRange;
  document.querySelectorAll('[data-preset]').forEach((b) =>
    b.classList.toggle('on', b.dataset.preset === name));
  scheduleReproject();
}

function linkPair(name, onChange) {
  const num = $(name);
  const range = $(`${name}Range`);
  if (!num) return;
  const push = (from, to) => {
    if (to) to.value = from.value;
    document.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('on'));
    onChange();
  };
  num.addEventListener('input', () => push(num, range));
  range?.addEventListener('input', () => push(range, num));
}

/* ═════════════════════════ stage 3: reproject ═════════════════════════ */

let reprojectTimer = null;

function scheduleReproject() {
  if (!state.imageId && !state.jobId) return;
  clearTimeout(reprojectTimer);
  reprojectTimer = setTimeout(doReproject, 140);
}

async function doReproject() {
  // Slider drags overlap, so stamp each request and ignore anything stale.
  const seq = ++state.reprojectSeq;
  setStages({ source: 'done', detect: 'done', fuse: 'active' });

  try {
    if (state.mediaKind === 'kitti' && state.jobId && state.timeline) {
      // The pose is measured; only the fallback's range bound is ours to vary.
      const data = await api.kittiTimeline(state.jobId, Number($('maxRange').value));
      if (seq !== state.reprojectSeq) return;
      adoptTimeline(data, { refit: false });
    } else if (state.mediaKind === 'video' && state.jobId && state.timeline) {
      const data = await api.videoTimeline(state.jobId, readCamera());
      if (seq !== state.reprojectSeq) return;
      adoptTimeline(data, { refit: false });
    } else if (state.mediaKind === 'image' && state.imageId) {
      const data = await api.reproject(state.imageId, readCamera());
      if (seq !== state.reprojectSeq) return;
      applyData(data, { refit: false });
    }
  } catch (err) {
    if (seq !== state.reprojectSeq) return;
    toast(`reprojection failed: ${err.message}`, 'err');
    setStages({ fuse: '' });
  }
}

/* ═════════════════════════ stage 4: paint everything ═════════════════════════ */

async function applyData(data, { refit = false, newMedia = false } = {}) {
  state.data = data;
  state.metrics = data.metrics;
  if (data.image_id) state.imageId = data.image_id;

  if (newMedia && data.image?.url) {
    await overlay.setImage(data.image.url);
    $('imageEmpty').hidden = true;
    clearSelection(false);
  }

  overlay.setDetections(data.detections);
  mapView.render(data, { refit: refit || newMedia });
  scene3d.render(data, { refit: refit || newMedia });
  panel.setData(data.detections);

  applyFilters();
  syncSelection();
  updateStatus(data.metrics, data.camera);
  updateDerived(data.camera);
  updateSceneTree(data.scene?.tree);
  updateSourceBadge(data.detections);
  updateImageInfo(data);
  assistant.updateSuggestions(data.detections, categoryOf);

  $('exportBtn').disabled = data.detections.length === 0;
  setStages({ source: 'done', detect: 'done', fuse: 'done', world: 'done', reason: 'done' });
  showExifHint(data.exif);
}

/* ═════════════════════════ readouts ═════════════════════════ */

function updateStatus(m, cam) {
  const set = (id, v) => { $(id).textContent = v; };
  if (m) {
    set('stObjects', m.object_count ?? 0);
    set('stClasses', m.class_count ?? 0);
    set('stConf', m.avg_confidence ? `${(m.avg_confidence * 100).toFixed(0)}%` : '—');
    set('stInfer', m.inference_ms ? `${m.inference_ms.toFixed(0)}ms` : '—');
    set('stFps', m.fps ? m.fps.toFixed(1) : '—');
    set('stNear', m.nearest ? `${m.nearest.class} ${m.nearest.distance_m}m` : '—');
    set('stFar', m.farthest ? `${m.farthest.class} ${m.farthest.distance_m}m` : '—');
    const a = m.visible_area_m2;
    set('stArea', a ? (a > 10000 ? `${(a / 10000).toFixed(2)}ha` : `${a.toFixed(0)}m²`) : 'open');
  }
  if (cam) {
    set('stPos', `${cam.lat.toFixed(6)}, ${cam.lon.toFixed(6)}  hdg ${cam.heading_deg.toFixed(0)}°`);
  }
}

function updateDerived(cam) {
  if (!cam) return;
  $('derivedFov').textContent = `${cam.hfov_deg.toFixed(1)}° × ${cam.vfov_deg.toFixed(1)}°`;
  $('derivedFocal').textContent = `${cam.fx.toFixed(1)} px`;
}

function updateSceneTree(tree) {
  $('sceneTree').textContent = tree && tree.trim() ? tree : '—';
}

function updateImageInfo(data) {
  const info = $('imageInfo');
  if (state.mediaKind === 'kitti' && state.calibration) {
    const c = state.calibration;
    info.textContent = `${c.image_size[0]}×${c.image_size[1]} · f=${c.fx.toFixed(0)}px`;
  } else if (data.image) {
    info.textContent = `${data.image.width}×${data.image.height}`;
  } else if (state.videoMeta) {
    info.textContent = `${state.videoMeta.width}×${state.videoMeta.height}`;
  } else {
    info.textContent = '';
  }
}

/**
 * The provenance badge in the detection pane header.
 *
 * With one camera every position is a projection onto an assumed ground plane
 * and the badge says so. With LiDAR most are measured. Overclaiming either way
 * would be wrong, so it reports the actual split for the frame on screen.
 */
function updateSourceBadge(detections) {
  const list = detections || [];
  const measured = list.filter((d) => d.world.valid && d.world.source === 'lidar').length;
  const estimated = list.filter((d) => d.world.valid && d.world.source !== 'lidar').length;
  const badge = $('srcBadge');

  if (measured === 0) {
    badge.textContent = 'estimated';
    badge.title = 'Projected onto an assumed flat ground plane from a single camera';
  } else if (estimated === 0) {
    badge.textContent = `lidar ${measured}`;
    badge.title = 'Measured from real LiDAR returns';
  } else {
    badge.textContent = `lidar ${measured} · est ${estimated}`;
    badge.title = 'Some positions measured by LiDAR, the rest projected onto the ground plane';
  }
  badge.classList.toggle('measured', measured > 0 && estimated === 0);
}

/* ═════════════════════════ sources tree ═════════════════════════ */

function markActiveSource(key) {
  state.sourceKey = key;
  for (const item of $('sourceList').children) {
    item.classList.toggle('on', item.dataset.key === key);
  }
}

/** Show which sensors the active source actually provides. */
function setSensors(rows) {
  const list = $('sensorList');
  list.innerHTML = '';
  if (!rows.length) {
    list.innerHTML = '<div class="sensor-row off"><span class="nm">no source loaded</span></div>';
    $('sensorMeta').textContent = '—';
    return;
  }
  for (const r of rows) {
    const row = document.createElement('div');
    row.className = 'sensor-row';
    row.innerHTML =
      `<svg class="ic ic-sm"><use href="#${r.icon}"/></svg>`
      + `<span class="nm">${r.name}</span>`
      + `<span class="val">${r.value}</span>`;
    row.title = r.title || '';
    list.appendChild(row);
  }
  $('sensorMeta').textContent = `${rows.length}`;
}

function sensorsForImage(data) {
  return [{
    icon: 'i-cam', name: 'camera',
    value: `${data.image.width}×${data.image.height}`,
    title: 'Single RGB camera; no depth sensor, so positions are projected',
  }];
}

function sensorsForKitti(data) {
  const c = data.calibration;
  return [
    { icon: 'i-cam', name: 'camera', value: `${c.image_size[0]}×${c.image_size[1]}`,
      title: 'Point Grey Flea 2, rectified' },
    { icon: 'i-lidar', name: 'velodyne', value: 'HDL-64E',
      title: 'Velodyne HDL-64E, roughly 120k returns per frame' },
    { icon: 'i-gps', name: 'oxts', value: 'RT3003',
      title: 'OXTS RT3003 GNSS/IMU: lat, lon, alt, roll, pitch, yaw' },
  ];
}

/* ═════════════════════════ image ═════════════════════════ */

async function pickFile(file) {
  const isVideo = /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.name || '')
    || (file.type || '').startsWith('video/');

  setCameraLocked(false);
  state.mediaKind = isVideo ? 'video' : 'image';
  state.file = file;
  state.imageId = null;
  state.jobId = null;
  state.kittiSequence = null;
  state.timeline = null;
  state.trails = [];
  state.frameIndex = -1;
  markActiveSource(null);
  clearSelection(false);
  timeline.clear();

  $('stSource').textContent = file.name;
  $('runBtn').disabled = false;
  resetStages();
  setStages({ source: 'done' });

  await (isVideo ? runVideo() : runImage());
}

async function runImage() {
  if (!state.file || state.busy) return;
  setBusy(true, 'running detection');
  setStages({ source: 'done', detect: 'active' });
  stopVideo();
  timeline.clear();

  try {
    const data = await api.analyze(state.file, readCamera(), readInference());
    $('rerunHint').hidden = true;
    await applyData(data, { newMedia: true, refit: true });
    setSensors(sensorsForImage(data));

    const n = data.detections.length;
    const g = data.georeferenced_count;
    if (n === 0) toast('no objects found — try a lower confidence', 'warn');
    else if (g === 0) toast(`${n} detected, none placed on the ground — check pitch and altitude`, 'warn', 6000);
    else toast(`${n} detected · ${g} positioned · ${data.inference_ms.toFixed(0)} ms`, 'ok');
  } catch (err) {
    toast(`detection failed: ${err.message}`, 'err', 7000);
    setStages({ detect: '' });
  } finally {
    setBusy(false);
  }
}

/* ═════════════════════════ video ═════════════════════════ */

async function runVideo() {
  if (!state.file || state.busy) return;
  setStages({ source: 'done', detect: 'active' });
  setProgress(true, 'tracking', 0);
  $('runBtn').disabled = true;

  try {
    const job = await api.analyzeVideo(state.file, readInference(), 5);
    state.jobId = job.job_id;
    await api.waitForVideo(job.job_id, (s) =>
      setProgress(true, s.message || 'tracking', (s.progress || 0) * 100));

    setStages({ detect: 'done', fuse: 'active' });
    const data = await api.videoTimeline(job.job_id, readCamera());
    await adoptTimeline(data, { refit: true, newMedia: true });
    setSensors([{ icon: 'i-cam', name: 'camera',
                  value: `${data.video.width}×${data.video.height}`,
                  title: 'Single RGB camera; positions are projected' }]);
    toast(`${data.video.frame_count} frames · ${data.trails.length} tracks`, 'ok');
  } catch (err) {
    toast(`video failed: ${err.message}`, 'err', 8000);
    setStages({ detect: '' });
  } finally {
    setProgress(false);
    $('runBtn').disabled = !state.file;
    $('rerunHint').hidden = true;
  }
}

/* ═════════════════════════ KITTI ═════════════════════════ */

async function runKitti(sequenceName) {
  if (state.busy || !sequenceName) return;

  state.mediaKind = 'kitti';
  state.file = null;
  state.imageId = null;
  state.jobId = null;
  state.timeline = null;
  state.trails = [];
  state.frameIndex = -1;
  clearSelection(false);
  timeline.clear();

  resetStages();
  setStages({ source: 'done', detect: 'active' });
  setProgress(true, 'fusing sensors', 0);
  $('runBtn').disabled = true;
  stopVideo();
  state.kittiSequence = sequenceName;
  $('stSource').textContent = sequenceName;

  try {
    const job = await api.kittiAnalyze(sequenceName, readInference());
    state.jobId = job.job_id;
    await api.waitForVideo(job.job_id, (s) =>
      setProgress(true, s.message || 'fusing', (s.progress || 0) * 100));

    setStages({ detect: 'done', fuse: 'active' });
    const data = await api.kittiTimeline(job.job_id, Number($('maxRange').value));
    await adoptTimeline(data, { refit: true, newMedia: true });
    setSensors(sensorsForKitti(data));

    const m = data.metrics;
    toast(`${data.video.frame_count} frames · ${m.measured_count} lidar · `
          + `${m.estimated_count} estimated · ${data.trails.length} tracks`, 'ok', 6000);
  } catch (err) {
    toast(`kitti failed: ${err.message}`, 'err', 8000);
    setStages({ detect: '' });
  } finally {
    setProgress(false);
    $('runBtn').disabled = false;
    $('rerunHint').hidden = true;
  }
}

/**
 * Lock the camera controls when the pose is measured rather than assumed.
 *
 * With real sensors there is nothing useful to drag: altitude, pitch, heading
 * and position all come from the GPS/IMU and change every frame, and the
 * intrinsics come from a calibration file.
 */
function setCameraLocked(locked) {
  for (const id of ['altitude', 'pitch', 'heading', 'hfov', 'lat', 'lon',
                    'altitudeRange', 'pitchRange', 'headingRange', 'hfovRange']) {
    const el = $(id);
    if (el) el.disabled = locked;
  }
  document.querySelectorAll('[data-preset]').forEach((b) => { b.disabled = locked; });
  $('camSourceMeta').textContent = locked ? 'oxts' : 'manual';
  $('poseNote').hidden = !locked;
}

/** Live sensor readings for the frame on screen. */
function updatePoseNote(frame) {
  const note = $('poseNote');
  if (note.hidden || !frame) return;
  const cam = frame.camera || {};
  note.innerHTML =
    'pose measured by OXTS GPS/IMU'
    + `<span class="mono">${(cam.lat ?? 0).toFixed(7)}, ${(cam.lon ?? 0).toFixed(7)}</span>`
    + `<span class="mono">hdg ${(cam.heading_deg ?? 0).toFixed(1)}° · `
    + `${(frame.speed_kmh ?? 0).toFixed(0)} km/h · cam ${(cam.altitude_m ?? 0).toFixed(2)} m agl</span>`
    + `<span class="mono">${(frame.lidar_points ?? 0).toLocaleString()} lidar returns</span>`;
}

/* ═════════════════════════ timeline payloads ═════════════════════════ */

async function adoptTimeline(data, { refit = false, newMedia = false } = {}) {
  state.mediaKind = data.media_kind === 'kitti' ? 'kitti' : 'video';
  state.kittiSequence = data.sequence || null;
  state.calibration = data.calibration || null;
  state.sensors = data.sensors || null;
  state.timeline = data.frames;
  state.trails = data.trails || [];
  state.videoMeta = data.video;
  state.videoCamera = data.camera;
  state.videoFootprint = data.footprint || [];
  state.metrics = data.metrics;

  setCameraLocked(state.mediaKind === 'kitti');

  const video = $('videoEl');
  if (newMedia) {
    video.src = data.video.url;
    // Never wait forever: a container the server could read is not necessarily
    // one the browser can decode, and hanging here looks like a freeze.
    await new Promise((resolve, reject) => {
      if (video.readyState >= 2) return resolve();
      const cleanup = () => {
        video.removeEventListener('loadeddata', done);
        video.removeEventListener('error', fail);
      };
      const done = () => { clearTimeout(timer); cleanup(); resolve(); };
      const fail = () => { clearTimeout(timer); cleanup(); reject(new Error('browser could not decode this clip')); };
      const timer = setTimeout(
        () => { cleanup(); reject(new Error('browser could not decode this clip (timed out)')); },
        15000,
      );
      video.addEventListener('loadeddata', done, { once: true });
      video.addEventListener('error', fail, { once: true });
    });

    overlay.setVideo(video);
    await primeFirstFrame(video);
    $('imageEmpty').hidden = true;
    state.frameIndex = -1;
  }

  timeline.setSequence({
    duration: data.video.duration_s,
    trails: state.trails,
    label: `${data.video.frame_count} frames @ ${data.video.source_fps} Hz`,
  });
  for (const id of ['btnPlay', 'btnStart', 'btnEnd']) $(id).disabled = false;

  updateStatus(data.metrics, data.camera);
  updateDerived(data.camera);
  $('exportBtn').disabled = false;
  setStages({ source: 'done', detect: 'done', fuse: 'done', world: 'done', reason: 'done' });

  showFrameAt(video.currentTime || 0, { force: true, refit });
}

/**
 * Force the first frame to decode.
 *
 * A paused video that has never been played or seeked can report readyState 4
 * while holding no composited frame: drawImage() then yields nothing and the
 * pane comes up blank until you press play. The nudge has to wait for the
 * browser to accept a seek at all -- at `loadeddata` the seekable range is
 * often still empty and the assignment is silently ignored.
 */
async function primeFirstFrame(video) {
  if (video.currentTime > 0) return;
  for (let i = 0; i < 12 && video.seekable.length === 0; i += 1) {
    await new Promise((r) => setTimeout(r, 50));
  }
  if (video.seekable.length === 0) return;

  await new Promise((resolve) => {
    const done = () => { clearTimeout(timer); resolve(); };
    const timer = setTimeout(resolve, 900);
    video.addEventListener('seeked', done, { once: true });
    try {
      video.currentTime = Math.min(0.04, (video.duration || 1) / 4);
    } catch {
      done();
    }
  });
}

function frameAt(seconds) {
  const frames = state.timeline;
  if (!frames?.length) return -1;
  let lo = 0;
  let hi = frames.length - 1;
  let best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].time_s <= seconds) { best = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return best;
}

/** Trails clipped to now, so paths draw themselves as the clip plays. */
function trailsUpTo(seconds) {
  return state.trails
    .map((t) => ({ ...t, points: t.points.filter((p) => p.t <= seconds + 1e-6) }))
    .filter((t) => t.points.length >= 2);
}

function showFrameAt(seconds, { force = false, refit = false } = {}) {
  const index = frameAt(seconds);
  if (index < 0) return;
  timeline.setTime(seconds, index, state.timeline.length);
  // The overlay redraws every animation frame; the map and 3D only need
  // touching when the processed frame actually changes.
  if (index === state.frameIndex && !force) return;
  state.frameIndex = index;

  const frame = state.timeline[index];
  const data = {
    // A KITTI frame carries its own pose because the vehicle is moving.
    camera: frame.camera || state.videoCamera,
    detections: frame.detections,
    footprint: state.videoFootprint,
    metrics: state.metrics,
    media_kind: state.mediaKind,
  };
  state.data = data;

  overlay.setDetections(frame.detections);
  const trails = trailsUpTo(seconds);
  mapView.render(data, { refit, trails });
  scene3d.render(data, { refit, trails });
  panel.setData(frame.detections);

  applyFilters();
  syncSelection();
  updateSourceBadge(frame.detections);
  updateStatus(state.metrics, data.camera);
  updateDerived(data.camera);
  updateImageInfo(data);
  updatePoseNote(frame);
  assistant.updateSuggestions(frame.detections, categoryOf);
  refreshSceneTree();
}

/* The scene graph changes as the clip plays, so it is fetched on demand. The
 * debounce is longer than the gap between processed frames on purpose -- a
 * shorter one would turn a side panel into several requests a second. */
let treeTimer = null;
function refreshSceneTree() {
  clearTimeout(treeTimer);
  treeTimer = setTimeout(async () => {
    const scene = currentScene();
    if (!scene?.detections?.length) return updateSceneTree(null);
    try {
      const result = await api.sceneGraph(scene.camera, scene.detections);
      updateSceneTree(result.tree);
    } catch { /* cosmetic; never disturb the rest */ }
  }, 600);
}

/* ═════════════════════════ transport ═════════════════════════ */

function wireTransport() {
  const video = $('videoEl');
  let loop = false;

  const setIcon = (playing) => {
    $('playIcon').innerHTML = `<use href="#${playing ? 'i-pause' : 'i-play'}"/>`;
  };

  $('btnPlay').addEventListener('click', () => {
    if (video.paused) video.play(); else video.pause();
  });
  $('btnStart').addEventListener('click', () => { video.currentTime = 0; });
  $('btnEnd').addEventListener('click', () => {
    video.currentTime = Math.max(0, (video.duration || 0) - 0.05);
  });
  $('btnLoop').addEventListener('click', () => {
    loop = !loop;
    video.loop = loop;
    $('btnLoop').classList.toggle('on', loop);
  });
  $('tlSpeed').addEventListener('change', (e) => {
    video.playbackRate = Number(e.target.value);
  });

  video.addEventListener('play', () => { setIcon(true); overlay.startLoop(); });
  video.addEventListener('pause', () => { setIcon(false); overlay.stopLoop(); overlay.draw(); });
  video.addEventListener('ended', () => setIcon(false));
  video.addEventListener('timeupdate', () => {
    if (state.timeline) showFrameAt(video.currentTime);
  });
  video.addEventListener('seeked', () => {
    if (state.timeline) showFrameAt(video.currentTime, { force: true });
    overlay.draw();
  });
}

function stopVideo() {
  const video = $('videoEl');
  video.pause();
  video.removeAttribute('src');
  video.load();
  overlay.stopLoop();
  state.timeline = null;
  state.trails = [];
  state.jobId = null;
  for (const id of ['btnPlay', 'btnStart', 'btnEnd']) $(id).disabled = true;
}

/* ═════════════════════════ selection + filters ═════════════════════════ */

function syncSelection() {
  const ids = [...state.selected];
  overlay.setSelected(ids);
  mapView.setSelected(ids);
  scene3d.setSelected(ids);
  panel.setSelected(ids);
  renderInspector();

  // Mirror the selection onto the timeline, by track rather than by row index.
  const first = ids[0];
  const det = first !== undefined
    ? (state.data?.detections || []).find((d) => d.id === first) : null;
  timeline.setSelectedTrack(det?.track_id ?? null);
}

function setSelection(ids, origin) {
  state.selected = new Set(ids);
  syncSelection();
  if (origin === 'panel' || origin === 'assistant' || origin === 'timeline') {
    const first = [...state.selected][0];
    if (first !== undefined) { mapView.focus(first); scene3d.focus(first); }
  }
}

function toggleSelection(id, event) {
  if (id === null) return clearSelection();
  const additive = event && (event.ctrlKey || event.metaKey || event.shiftKey);
  if (additive) {
    const next = new Set(state.selected);
    next.has(id) ? next.delete(id) : next.add(id);
    setSelection(next, 'panel');
  } else {
    setSelection(state.selected.has(id) && state.selected.size === 1 ? [] : [id], 'panel');
  }
}

function clearSelection(sync = true) {
  state.selected = new Set();
  if (sync) syncSelection();
}

function applyFilters() {
  const visible = panel.visibleIdSet();
  overlay.setVisible(visible);
  mapView.setVisible(visible);
  scene3d.setVisible(visible);
}

function setAssistantFilter(ids, label) {
  state.assistantIds = ids;
  panel.setVisible(ids);
  applyFilters();
  const chip = $('selChip');
  chip.hidden = !ids;
  if (ids) $('selLabel').textContent = `${label || 'selection'} · ${ids.length}`;
}

function applyAssistantActions(actions) {
  for (const action of actions) {
    if (action.type === 'select') {
      if (!action.ids?.length) { toast('nothing matched', 'warn'); continue; }
      setAssistantFilter(action.ids, action.label);
      setSelection(action.ids, 'assistant');
      mapView.fitTo(action.ids);
      setStages({ reason: 'done' });
    } else if (action.type === 'clear') {
      setAssistantFilter(null); clearSelection();
    } else if (action.type === 'focus') {
      setSelection([action.id], 'assistant');
    } else if (action.type === 'view') {
      switchView(action.view);
    }
  }
}

const currentScene = () =>
  state.data ? { camera: state.data.camera, detections: state.data.detections } : null;

/* ═════════════════════════ inspector ═════════════════════════ */

/** Properties of whatever is selected, as label/value rows. */
function renderInspector() {
  const box = $('inspector');
  const ids = [...state.selected];

  if (ids.length === 0) {
    box.innerHTML = '<div class="inspect-empty">Nothing selected.</div>';
    return;
  }
  if (ids.length > 1) {
    box.innerHTML = `<div class="inspect-empty">${ids.length} objects selected.</div>`;
    return;
  }

  const det = (state.data?.detections || []).find((d) => d.id === ids[0]);
  if (!det) { box.innerHTML = '<div class="inspect-empty">Nothing selected.</div>'; return; }

  const w = det.world;
  const name = det.track_id != null
    ? `${det.class_name}_${String(det.track_id).padStart(2, '0')}`
    : det.class_name;

  const row = (k, v, title = '') =>
    `<div class="prop" ${title ? `title="${title}"` : ''}>
       <span class="k">${k}</span><span class="v">${v}</span></div>`;

  let html =
    `<div class="inspect-title">
       <span class="sw" style="background:${classColor(det.class_id)}"></span>
       <span class="nm">${name}</span>
       ${w.valid && w.source === 'lidar'
         ? '<span class="tag tag-ok">LIDAR</span>'
         : '<span class="tag tag-warn">ESTIMATED</span>'}
     </div>`;

  html += row('class', det.class_name);
  html += row('confidence', det.confidence.toFixed(3));
  if (det.track_id != null) html += row('track id', det.track_id);
  html += row('bbox', `${det.bbox.x1.toFixed(0)}, ${det.bbox.y1.toFixed(0)} → `
                      + `${det.bbox.x2.toFixed(0)}, ${det.bbox.y2.toFixed(0)}`);
  html += row('anchor px', `${det.bbox.anchor_x.toFixed(0)}, ${det.bbox.anchor_y.toFixed(0)}`,
              'The pixel that was geo-referenced');

  if (w.valid) {
    html += row('latitude', w.lat.toFixed(7));
    html += row('longitude', w.lon.toFixed(7));
    html += row('east', `${w.east_m.toFixed(2)} m`);
    html += row('north', `${w.north_m.toFixed(2)} m`);
    html += row('distance', `${w.ground_range_m.toFixed(2)} m`);
    html += row('bearing', `${w.bearing_deg.toFixed(1)}°`);
    html += row('size est.', `${w.est_width_m.toFixed(2)} × ${w.est_height_m.toFixed(2)} m`);
    if (w.source === 'lidar') {
      html += row('depth', `${w.depth_m.toFixed(2)} m`, 'Range along the optical axis');
      html += row('lidar pts', w.lidar_points, 'Returns used for this measurement');
    }
  } else {
    html += `<div class="section-note warn">${w.reason || 'no ground fix'}</div>`;
  }

  box.innerHTML = html;
}

/* ═════════════════════════ views ═════════════════════════ */

function switchView(name) {
  $('viewport').dataset.view = name;
  for (const b of $('viewSeg').children) b.classList.toggle('on', b.dataset.view === name);
  requestAnimationFrame(() => { mapView.invalidate(); scene3d.resize(); });
}

function toggleFullscreen(force) {
  const on = force !== undefined ? force : !document.body.classList.contains('fs');
  document.body.classList.toggle('fs', on);
  requestAnimationFrame(() => { mapView.invalidate(); scene3d.resize(); });
}

/* ═════════════════════════ EXIF ═════════════════════════ */

function showExifHint(exif) {
  const note = $('exifNote');
  if (!exif?.found?.length) { note.hidden = true; return; }

  const bits = [];
  if (exif.lat !== undefined) bits.push(`${exif.lat.toFixed(5)}, ${exif.lon.toFixed(5)}`);
  if (exif.altitude_m !== undefined) bits.push(`alt ${exif.altitude_m.toFixed(1)} m`);
  if (exif.heading_deg !== undefined) bits.push(`hdg ${exif.heading_deg.toFixed(0)}°`);
  if (exif.pitch_deg !== undefined) bits.push(`pitch ${exif.pitch_deg.toFixed(0)}°`);
  if (exif.hfov_deg !== undefined) bits.push(`fov ${exif.hfov_deg.toFixed(1)}°`);

  note.hidden = false;
  note.innerHTML =
    `${[exif.make, exif.model].filter(Boolean).join(' ') || 'image metadata'} carries camera data`
    + `<span class="mono">${bits.join(' · ')}</span>`
    + '<button type="button" id="applyExif">apply</button>';

  $('applyExif').addEventListener('click', () => {
    if (exif.lat !== undefined) {
      $('lat').value = exif.lat.toFixed(6);
      $('lon').value = exif.lon.toFixed(6);
    }
    if (exif.altitude_m !== undefined) setPair('altitude', exif.altitude_m.toFixed(1));
    if (exif.heading_deg !== undefined) setPair('heading', exif.heading_deg.toFixed(0));
    if (exif.pitch_deg !== undefined) setPair('pitch', exif.pitch_deg.toFixed(0));
    if (exif.hfov_deg !== undefined) setPair('hfov', exif.hfov_deg.toFixed(1));
    document.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('on'));
    note.hidden = true;
    toast('camera updated from metadata', 'ok');
    scheduleReproject();
  });
}

/* ═════════════════════════ export ═════════════════════════ */

async function doExport(fmt) {
  const scene = currentScene();
  if (!scene) return;
  $('exportMenu').hidden = true;

  const source = state.data.image?.filename || state.kittiSequence
    || state.videoMeta?.filename || 'scene';

  try {
    if (fmt === 'png') {
      const blob = await overlay.toBlob();
      if (!blob) throw new Error('nothing to capture');
      api.saveBlob(blob, `${source.replace(/\.[^.]+$/, '')}_detections.png`);
      toast('screenshot saved', 'ok');
      return;
    }
    const visible = panel.visibleIdSet();
    const detections = visible
      ? scene.detections.filter((d) => visible.has(d.id))
      : scene.detections;

    await api.exportScene(fmt, {
      camera: scene.camera,
      detections,
      footprint: state.data.footprint || [],
      trails: state.timeline ? state.trails : [],
      source,
      metrics: state.metrics || {},
      scene: state.data.scene?.context || {},
    });
    toast(`${detections.length} objects exported as ${fmt}`, 'ok');
  } catch (err) {
    toast(`export failed: ${err.message}`, 'err');
  }
}

/* ═════════════════════════ wiring ═════════════════════════ */

function wireControls() {
  // tree groups collapse
  for (const group of document.querySelectorAll('.tree-group')) {
    group.querySelector('.tree-head').addEventListener('click', () =>
      group.classList.toggle('closed'));
  }

  const input = $('fileInput');
  const dz = $('dropzone');
  const open = () => input.click();
  $('openBtn').addEventListener('click', open);
  dz.addEventListener('click', open);
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
  });
  input.addEventListener('change', () => {
    if (input.files[0]) pickFile(input.files[0]);
    input.value = '';
  });

  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f && (f.type.startsWith('image/') || f.type.startsWith('video/'))) pickFile(f);
  });

  ['altitude', 'pitch', 'heading', 'hfov'].forEach((n) => linkPair(n, scheduleReproject));
  ['lat', 'lon', 'maxRange'].forEach((n) => $(n).addEventListener('input', scheduleReproject));
  document.querySelectorAll('[data-preset]').forEach((b) =>
    b.addEventListener('click', () => applyPreset(b.dataset.preset)));

  const flagRerun = () => {
    if (!state.imageId && !state.jobId) return;
    $('rerunHint').hidden = false;
  };
  $('confidence').addEventListener('input', (e) => {
    $('confVal').textContent = Number(e.target.value).toFixed(2);
    flagRerun();
  });
  $('iou').addEventListener('input', (e) => {
    $('iouVal').textContent = Number(e.target.value).toFixed(2);
    flagRerun();
  });
  $('runBtn').addEventListener('click', () => {
    if (state.mediaKind === 'kitti') runKitti(state.kittiSequence);
    else if (state.mediaKind === 'video') runVideo();
    else runImage();
  });

  for (const b of $('viewSeg').children) {
    b.addEventListener('click', () => switchView(b.dataset.view));
  }
  document.querySelectorAll('[data-solo]').forEach((b) =>
    b.addEventListener('click', () => {
      const view = b.dataset.solo;
      switchView($('viewport').dataset.view === view ? 'grid' : view);
    }));

  $('showRays').addEventListener('change', (e) => scene3d.setShowRays(e.target.checked));
  $('showTrails').addEventListener('change', (e) => {
    mapView.setShowTrails(e.target.checked);
    scene3d.setShowTrails(e.target.checked);
  });

  $('exportBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    $('exportMenu').hidden = !$('exportMenu').hidden;
  });
  $('exportMenu').querySelectorAll('button').forEach((b) =>
    b.addEventListener('click', () => doExport(b.dataset.fmt)));
  document.addEventListener('click', () => { $('exportMenu').hidden = true; });

  $('selClear').addEventListener('click', () => { setAssistantFilter(null); clearSelection(); });

  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea, select')) return;
    if (e.key >= '1' && e.key <= '4') {
      switchView(['image', 'map', 'scene', 'grid'][Number(e.key) - 1]);
    } else if (e.key === 'f' || e.key === 'F') {
      toggleFullscreen();
    } else if (e.key === 'Escape') {
      if (document.body.classList.contains('fs')) toggleFullscreen(false);
      else { setAssistantFilter(null); clearSelection(); }
    } else if (e.key === ' ' && state.timeline) {
      e.preventDefault();
      const v = $('videoEl');
      v.paused ? v.play() : v.pause();
    }
  });
}

function wireSplitters() {
  const work = $('work');
  const onResize = () => requestAnimationFrame(() => {
    mapView.invalidate(); scene3d.resize();
  });
  makeSplitter({ handle: $('splitL'), target: work, prop: '--w-l', axis: 'x',
                 min: 190, max: 460, key: 'l', onResize });
  makeSplitter({ handle: $('splitR'), target: work, prop: '--w-r', axis: 'x',
                 min: 240, max: 560, invert: true, key: 'r', onResize });
  makeSplitter({ handle: $('splitTable'), target: $('centre'), prop: '--h-table', axis: 'y',
                 min: 0, max: 460, invert: true, key: 'tbl', onResize });
}

/* ═════════════════════════ boot ═════════════════════════ */

async function loadSources() {
  const list = $('sourceList');
  list.innerHTML = '';
  let count = 0;

  const addItem = ({ key, icon, name, side, tag, title, onOpen }) => {
    const el = document.createElement('div');
    el.className = 'tree-item';
    el.dataset.key = key;
    el.title = title || '';
    el.innerHTML =
      `<svg class="ic ic-sm"><use href="#${icon}"/></svg>`
      + `<span class="nm">${name}</span>`
      + (tag ? `<span class="tag ${tag.cls}">${tag.text}</span>` : '')
      + (side ? `<span class="side">${side}</span>` : '');
    el.addEventListener('click', () => { markActiveSource(key); onOpen(); });
    list.appendChild(el);
    count += 1;
  };

  // Real sensor data first: it is the only source whose positions are measured.
  try {
    const { sequences } = await api.kittiSequences();
    for (const seq of sequences) {
      const short = seq.name.replace(/^\d{4}_\d{2}_\d{2}_drive_/, 'drive ').replace('_sync', '');
      addItem({
        key: `kitti:${seq.name}`,
        icon: 'i-lidar',
        name: `kitti ${short}`,
        side: `${seq.frames}f`,
        tag: { cls: 'tag-ok', text: 'REAL' },
        title: `KITTI raw — camera + Velodyne HDL-64E + OXTS GNSS/IMU\n`
             + `${seq.frames} frames @ ${seq.fps} Hz, ${seq.image_size[0]}×${seq.image_size[1]}\n`
             + `origin ${seq.origin.lat.toFixed(5)}, ${seq.origin.lon.toFixed(5)}`,
        onOpen: () => runKitti(seq.name),
      });
    }
  } catch { /* no dataset present */ }

  try {
    const { samples } = await api.getSamples();
    for (const s of samples) {
      addItem({
        key: `sample:${s.filename}`,
        icon: s.kind === 'video' ? 'i-vid' : 'i-img',
        name: s.label.toLowerCase(),
        side: `${(s.size_bytes / 1024).toFixed(0)}K`,
        title: s.filename,
        onOpen: async () => {
          try {
            setBusy(true, 'loading');
            const res = await fetch(s.url);
            if (!res.ok) throw new Error(`could not load ${s.filename}`);
            const file = new File([await res.blob()], s.filename,
                                  { type: (await res.clone().blob()).type });
            if (s.preset) applyPreset(s.preset);
            setBusy(false);
            await pickFile(file);
          } catch (err) {
            setBusy(false);
            toast(`sample failed: ${err.message}`, 'err');
          }
        },
      });
    }
  } catch { /* none */ }

  $('sourceCount').textContent = String(count);
  if (!count) list.innerHTML = '<div class="sensor-row off"><span class="nm">no datasets</span></div>';
}

async function checkHealth() {
  $('healthDot').className = 'dot busy';
  $('healthText').textContent = 'connecting';
  try {
    const h = await api.getHealth();
    $('healthDot').className = `dot ${h.model_loaded ? 'ok' : 'err'}`;
    $('healthText').textContent = h.model_loaded
      ? `${h.model_name} ${h.device}` : 'model offline';
    $('modelChip').title = `${h.num_classes} COCO classes on ${h.device}`;
    $('modelMeta').textContent = h.model_name.replace('.pt', '');

    const llm = h.assistant || {};
    $('llmDot').className = `dot ${llm.available ? 'ok' : 'err'}`;
    $('llmText').textContent = llm.available ? llm.model : 'assistant off';
    $('llmChip').title = llm.available ? `Assistant: ${llm.model}` : llm.reason;
    assistant.setStatus(llm);
  } catch (err) {
    $('healthDot').className = 'dot err';
    $('healthText').textContent = 'backend unreachable';
    toast(`cannot reach backend: ${err.message}`, 'err', 9000);
  }
}

async function loadCategories() {
  try {
    const { categories } = await api.getCategories();
    primeCategories(categories);
    panel.buildFilters(categories);
  } catch { panel.buildFilters([]); }
}

function init() {
  overlay = new DetectionOverlay($('overlayCanvas'), {
    onSelect: (id) => setSelection(id === null ? [] : [id], 'image'),
  });
  mapView = new MapView('map', { onSelect: (id) => setSelection([id], 'map') });
  scene3d = new Scene3D($('scene'), {
    onSelect: (id) => setSelection(id === null ? [] : [id], 'scene'),
  });

  // Keep the compass rose aligned with the orbit camera.
  let lastAz = null;
  const rose = $('compassRose');
  scene3d.onFrame = (az) => {
    if (lastAz !== null && Math.abs(az - lastAz) < 0.5) return;
    lastAz = az;
    rose?.setAttribute('transform', `rotate(${az} 23 23)`);
  };

  panel = new ObjectPanel({
    tbody: $('objBody'),
    countEl: $('objCount'),
    filterInput: $('qfilter'),
    emptyEl: $('tableEmpty'),
    filtersEl: $('filters'),
    onSelect: (id, e) => toggleSelection(id, e),
    onFocus: (id) => { setSelection([id], 'panel'); switchView('grid'); },
    onFilterChange: applyFilters,
  });

  assistant = new Assistant({
    chatEl: $('chat'),
    formEl: $('composer'),
    inputEl: $('chatInput'),
    sendBtn: $('sendBtn'),
    suggestionsEl: $('suggest'),
    modelEl: $('assistantModel'),
    clearBtn: $('clearChat'),
    getScene: currentScene,
    onActions: applyAssistantActions,
  });

  timeline = new Timeline({
    root: $('timeline'),
    labelsEl: $('tlLabels'),
    tracksEl: $('tlTracks'),
    rulerEl: $('tlRuler'),
    rowsEl: $('tlRows'),
    playheadEl: $('tlPlayhead'),
    clockEl: $('tlClock'),
    subEl: $('tlSub'),
    tracksCountEl: $('tlTrackCount'),
    toggleEl: $('tlToggle'),
    onSeek: (t) => { $('videoEl').currentTime = t; },
    onSelectTrack: (trackId) => {
      const det = (state.data?.detections || []).find((d) => d.track_id === trackId);
      if (det) setSelection([det.id], 'timeline');
      else toast('that object is not in this frame', 'warn', 2500);
    },
  });

  wireControls();
  wireTransport();
  wireSplitters();

  applyPreset('drone_oblique');
  loadCategories();
  checkHealth();
  loadSources();
  renderInspector();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
