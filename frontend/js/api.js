/**
 * Everything that talks to the FastAPI backend.
 *
 * Kept in one module so the rest of the frontend never touches fetch directly,
 * and so backend errors arrive as ordinary Error objects carrying the message
 * FastAPI put in its `detail` field.
 */

/** Turn a non-2xx response into an Error with the server's own explanation. */
async function toError(res) {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body.detail) {
      detail = Array.isArray(body.detail)
        ? body.detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join('; ')
        : body.detail;
    }
  } catch {
    /* not JSON; the status line is the best we have */
  }
  return new Error(detail);
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw await toError(res);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await toError(res);
  return res.json();
}

/* ------------------------------------------------------------------ system */

export const getHealth = () => getJSON('/api/health');
export const getSamples = () => getJSON('/api/samples');
export const getCategories = () => getJSON('/api/categories');

/* ------------------------------------------------------------------ image */

/** Stages 1-3: upload an image and get detections back already geo-referenced. */
export async function analyze(file, camera, inference) {
  const form = new FormData();
  form.append('file', file, file.name || 'upload.jpg');
  for (const [k, v] of Object.entries({ ...camera, ...inference })) {
    form.append(k, String(v));
  }
  const res = await fetch('/api/analyze', { method: 'POST', body: form });
  if (!res.ok) throw await toError(res);
  return res.json();
}

/** Stage 3 only: re-project cached detections against a new camera pose. */
export const reproject = (imageId, camera) =>
  postJSON('/api/reproject', { image_id: imageId, camera });

/* ------------------------------------------------------------------ video */

/** Start a tracking job. Returns a job descriptor to poll. */
export async function analyzeVideo(file, inference, targetFps = 5) {
  const form = new FormData();
  form.append('file', file, file.name || 'clip.mp4');
  form.append('confidence', String(inference.confidence));
  form.append('iou', String(inference.iou));
  form.append('target_fps', String(targetFps));
  const res = await fetch('/api/video/analyze', { method: 'POST', body: form });
  if (!res.ok) throw await toError(res);
  return res.json();
}

export const videoJob = (jobId) => getJSON(`/api/video/job/${jobId}`);

/** Geo-reference every tracked frame against a camera pose. */
export const videoTimeline = (jobId, camera) =>
  postJSON('/api/video/timeline', { job_id: jobId, camera });

/**
 * Poll a tracking job to completion.
 * `onProgress` receives each status payload so the UI can show a bar.
 */
export async function waitForVideo(jobId, onProgress, intervalMs = 700) {
  for (;;) {
    const status = await videoJob(jobId);
    onProgress?.(status);
    if (status.status === 'done') return status;
    if (status.status === 'error') throw new Error(status.error || 'video processing failed');
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

/* ------------------------------------------------------------------ KITTI */

/** Real KITTI raw drives available on the server. */
export const kittiSequences = () => getJSON('/api/kitti/sequences');

/** Start the real-sensor fusion job for a drive. Returns a job to poll. */
export const kittiAnalyze = (sequence, inference) =>
  postJSON('/api/kitti/analyze', {
    sequence,
    confidence: inference.confidence,
    iou: inference.iou,
  });

/**
 * Build the timeline from a finished KITTI job.
 * The camera pose comes from the GPS/IMU, so there is nothing to pass except
 * the range bound for the ground-plane fallback.
 */
export const kittiTimeline = (jobId, maxRangeM = 120) =>
  postJSON('/api/kitti/timeline', { job_id: jobId, max_range_m: maxRangeM });

/* ------------------------------------------------------- scene + assistant */

export const sceneGraph = (camera, detections) =>
  postJSON('/api/scene', { camera, detections });

export const askAssistant = (question, camera, detections, history) =>
  postJSON('/api/assistant', { question, camera, detections, history });

/* ------------------------------------------------------------------ export */

/**
 * Download the current scene in one of the supported formats.
 * The payload is whatever is on screen, so an export always matches the view
 * -- including any filter or assistant selection.
 */
export async function exportScene(fmt, payload) {
  const res = await fetch(`/api/export/${fmt}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await toError(res);

  const name = (res.headers.get('content-disposition') || '')
    .match(/filename="([^"]+)"/)?.[1] || `geodetect.${fmt}`;
  saveBlob(await res.blob(), name);
}

/** Hand the browser a blob as a download. */
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the download a tick to start before releasing the blob.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
