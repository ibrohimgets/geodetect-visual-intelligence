/**
 * The 3D WORLD view: detections as real solids standing on a ground plane.
 *
 * Axis convention
 * ---------------
 * The backend works in ENU (X=East, Y=North, Z=Up). Three.js is Y-up, so
 *
 *      three.x =  east
 *      three.y =  up
 *      three.z = -north        (north points "into" the screen)
 *
 * That mapping lives in `enuToScene` and nothing else in this file converts
 * coordinates. One conversion, one place to get it wrong.
 *
 * Honesty
 * -------
 * Box sizes are the metre estimates the geometry produced, so a bus really is
 * larger than a person. Depth is *not* observable from a single camera, so the
 * measured width is reused for it. Every object is drawn translucent with a ray
 * back to the camera precisely because these are projected estimates, not
 * measured points -- the UI says so too.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { classColorHex, classColor } from './palette.js';

const MIN_SIZE_M = 0.35;
const SELECT_COLOR = 0xffffff;

/** The one and only ENU -> Three.js conversion. */
function enuToScene(east, north, up = 0) {
  return new THREE.Vector3(east, up, -north);
}

/** A "nice" ring spacing for the current scene size: 1, 2, 5, 10, 20, 50... */
function niceStep(extent) {
  const raw = extent / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = norm >= 5 ? 5 : norm >= 2 ? 2 : 1;
  return step * mag;
}

export class Scene3D {
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.onSelect = onSelect || (() => {});

    this.selectedIds = new Set();
    this.visibleIds = null;
    this.showFootprint = true;
    this.showLabels = true;
    this.showRays = true;
    this.showTrails = true;
    this.objects = new Map();

    // preserveDrawingBuffer keeps the rendered frame readable after the
    // browser composites it, which is what makes toBlob() below possible. It
    // costs a little memory bandwidth; being able to export the view is worth
    // more than that here.
    this.renderer = new THREE.WebGLRenderer({
      antialias: true, alpha: true, preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth || 1, container.clientHeight || 1);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      50, (container.clientWidth || 1) / (container.clientHeight || 1), 0.1, 8000,
    );
    this.camera.position.set(40, 45, 70);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // stay above the ground
    this.controls.target.set(0, 0, 0);

    // Lighting tuned for a light scene: bright ambient, soft key, no harsh rim.
    this.scene.add(new THREE.HemisphereLight(0xc8d8ee, 0x0b0c0e, 1.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(60, 140, 60);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x4c8dff, 0.4);
    fill.position.set(-70, 40, -60);
    this.scene.add(fill);

    this.gridGroup = new THREE.Group();
    this.ringGroup = new THREE.Group();
    this.detGroup = new THREE.Group();
    this.camGroup = new THREE.Group();
    this.footGroup = new THREE.Group();
    this.rayGroup = new THREE.Group();
    this.trailGroup = new THREE.Group();
    this.scene.add(this.gridGroup, this.ringGroup, this.detGroup,
                   this.camGroup, this.footGroup, this.rayGroup, this.trailGroup);

    this._extent = 0;
    this._buildGrid(80);
    this._buildAxes();

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this._pointerDown = null;

    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => {
      this._pointerDown = { x: e.clientX, y: e.clientY };
    });
    dom.addEventListener('pointerup', (e) => {
      // Only a click if the pointer barely moved; an orbit drag that happens to
      // end over a box must not hijack the selection.
      if (!this._pointerDown) return;
      const moved = Math.hypot(e.clientX - this._pointerDown.x, e.clientY - this._pointerDown.y);
      this._pointerDown = null;
      if (moved < 4) this._handleClick(e);
    });

    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);

    this._animate = this._animate.bind(this);
    this.renderer.setAnimationLoop(this._animate);
  }

  /** Where the orbit camera is looking, as a compass bearing. Drives the rose. */
  get azimuthDeg() {
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
    // scene -z is north, scene +x is east
    return (Math.atan2(dir.x, -dir.z) * 180) / Math.PI;
  }

  // ---------------------------------------------------------------- scaffold

  _buildGrid(extent) {
    this._disposeGroup(this.gridGroup);

    const divisions = Math.max(8, Math.round(extent / niceStep(extent) * 2));
    const grid = new THREE.GridHelper(extent * 2, divisions, 0x39414d, 0x232932);
    grid.material.transparent = true;
    grid.material.opacity = 0.9;
    this.gridGroup.add(grid);

    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(extent * 2, extent * 2),
      new THREE.MeshStandardMaterial({
        color: 0x121519, roughness: 0.97, metalness: 0,
        transparent: true, opacity: 0.9,
      }),
    );
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.02; // just under the grid, to avoid z-fighting
    this.gridGroup.add(plane);

    this._extent = extent;
  }

  /**
   * Concentric distance rings around the camera.
   *
   * The single most useful piece of furniture in this view: it turns "that box
   * is over there" into "that box is about 40 m away" without clicking
   * anything.
   */
  _buildRings(extent) {
    this._disposeGroup(this.ringGroup);

    const step = niceStep(extent);
    const labelSize = this._labelSize();

    for (let r = step; r <= extent * 1.05; r += step) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(r - extent * 0.0016, r + extent * 0.0016, 96),
        new THREE.MeshBasicMaterial({
          color: 0x5a6675, side: THREE.DoubleSide,
          transparent: true, opacity: 0.5,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.01;
      this.ringGroup.add(ring);

      // Label each ring once, out along the east axis.
      const text = r >= 1000 ? `${(r / 1000).toFixed(1)} km` : `${r.toFixed(r < 10 ? 1 : 0)} m`;
      this.ringGroup.add(
        this._makeLabel(text, enuToScene(r, 0, labelSize * 0.35), '#98a0ac', labelSize * 0.7),
      );
    }
  }

  _buildAxes() {
    const extent = this._extent || 60;
    const L = extent * 0.4;
    const labelSize = this._labelSize();
    const lift = extent * 0.0015;

    const line = (from, to, color) => new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([from, to]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
    );

    this.gridGroup.add(line(enuToScene(0, 0, lift), enuToScene(L, 0, lift), 0xff6b6b)); // East
    this.gridGroup.add(line(enuToScene(0, 0, lift), enuToScene(0, L, lift), 0x57d97f)); // North
    this.gridGroup.add(line(enuToScene(0, 0, 0), enuToScene(0, 0, L * 0.55), 0x4cc2ff)); // Up

    const pad = extent * 0.05;
    this.gridGroup.add(this._makeLabel('E', enuToScene(L + pad, 0, pad * 0.4), '#ff6b6b', labelSize));
    this.gridGroup.add(this._makeLabel('N', enuToScene(0, L + pad, pad * 0.4), '#57d97f', labelSize));
    this.gridGroup.add(this._makeLabel('UP', enuToScene(0, 0, L * 0.55 + pad), '#4cc2ff', labelSize));
  }

  /** Label height in metres, so text stays legible without swamping the scene. */
  _labelSize() {
    return Math.max(0.45, (this._extent || 60) * 0.035);
  }

  /**
   * A text label as a camera-facing sprite.
   *
   * Sprites keep everything inside the WebGL canvas, so there is no second DOM
   * layer to keep aligned. The canvas is drawn at a fixed pixel size and then
   * scaled to metres, which is why `worldHeight` is a separate argument.
   */
  _makeLabel(text, position, color = '#d6dae1', worldHeight = 1.2) {
    const pad = 10;
    const fontPx = 42;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const font = `600 ${fontPx}px ui-monospace, Menlo, Consolas, monospace`;

    ctx.font = font;
    canvas.width = Math.ceil(ctx.measureText(text).width + pad * 2);
    canvas.height = fontPx + pad * 2;

    // Resizing a canvas resets its 2D context, so set the font again.
    ctx.font = font;
    ctx.fillStyle = 'rgba(11, 12, 14, 0.82)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
    ctx.fillStyle = color;
    ctx.textBaseline = 'middle';
    ctx.fillText(text, pad, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;

    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: true }),
    );
    sprite.scale.set(worldHeight * (canvas.width / canvas.height), worldHeight, 1);
    sprite.position.copy(position);
    return sprite;
  }

  // ------------------------------------------------------------------ render

  render(data, { refit = false, trails = null } = {}) {
    this._disposeGroup(this.detGroup);
    this._disposeGroup(this.camGroup);
    this._disposeGroup(this.footGroup);
    this._disposeGroup(this.rayGroup);
    this._disposeGroup(this.trailGroup);
    this.objects.clear();

    const cam = data.camera;
    const located = data.detections.filter((d) => d.world.valid);

    // Two scales. contentReach frames the subject; gridReach also has to cover
    // the footprint, which on a shallow oblique shot stretches far past every
    // detection -- framing on that would leave the objects as specks.
    let contentReach = Math.max(6, cam.altitude_m * 1.3);
    for (const d of located) {
      const size = Math.max(d.world.est_width_m, d.world.est_height_m);
      contentReach = Math.max(contentReach, (d.world.ground_range_m + size) * 1.25);
    }
    let gridReach = contentReach;
    for (const p of data.footprint || []) {
      gridReach = Math.max(gridReach, Math.hypot(p.east_m, p.north_m) * 1.1);
    }
    contentReach = Math.min(contentReach, 2000);
    gridReach = Math.min(gridReach, 2000);

    const rescaled = !this._extent
      || Math.abs(gridReach - this._extent) / this._extent > 0.25;
    if (rescaled) {
      this._buildGrid(gridReach);
      this._buildAxes();
      this._buildRings(gridReach);
    }

    this._buildCameraRig(cam, data.footprint || []);

    const camPos = enuToScene(0, 0, cam.altitude_m);

    for (const det of located) {
      const w = Math.max(MIN_SIZE_M, det.world.est_width_m);
      const h = Math.max(MIN_SIZE_M, det.world.est_height_m);
      const colorHex = classColorHex(det.class_id);
      const ground = enuToScene(det.world.east_m, det.world.north_m, 0);

      const mesh = new THREE.Mesh(
        // Depth is unobservable from one view, so the measured width is reused.
        new THREE.BoxGeometry(w, h, w),
        new THREE.MeshStandardMaterial({
          color: colorHex, roughness: 0.45, metalness: 0.1,
          transparent: true, opacity: 0.6,
        }),
      );
      mesh.position.copy(enuToScene(det.world.east_m, det.world.north_m, h / 2));
      mesh.userData.detectionId = det.id;

      // Crisp edges are what make this read as a CAD object rather than a lump
      // of translucent plastic.
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 1 }),
      );
      mesh.add(edges);
      this.detGroup.add(mesh);

      // Ground marker ring at the exact projected position.
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(w * 0.6, w * 0.78, 28),
        new THREE.MeshBasicMaterial({
          color: colorHex, side: THREE.DoubleSide, transparent: true, opacity: 0.6,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.copy(enuToScene(det.world.east_m, det.world.north_m, 0.04));
      this.detGroup.add(ring);

      // The ray the position was derived from: camera lens to ground point.
      const ray = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([camPos, ground]),
        new THREE.LineBasicMaterial({
          color: colorHex, transparent: true, opacity: 0.28,
        }),
      );
      this.rayGroup.add(ray);

      // Selection glow: a taller translucent shell that only shows when picked.
      const glow = new THREE.Mesh(
        new THREE.BoxGeometry(w * 1.3, h * 1.15, w * 1.3),
        new THREE.MeshBasicMaterial({
          color: 0x4c8dff, transparent: true, opacity: 0.22,
          side: THREE.BackSide, depthWrite: false,
        }),
      );
      glow.position.copy(mesh.position);
      glow.visible = false;
      this.detGroup.add(glow);

      const labelSize = this._labelSize() * 0.8;
      const name = det.track_id != null
        ? `${det.class_name}_${String(det.track_id).padStart(2, '0')}`
        : det.class_name;
      const label = this._makeLabel(
        `${name}  ${det.world.ground_range_m.toFixed(1)}m`,
        enuToScene(det.world.east_m, det.world.north_m, h + labelSize * 0.8),
        classColor(det.class_id),
        labelSize,
      );
      label.visible = this.showLabels;
      this.detGroup.add(label);

      this.objects.set(det.id, { mesh, edges, ring, ray, glow, label, colorHex });
    }

    if (trails) this._drawTrails(trails);

    this.rayGroup.visible = this.showRays;
    this.ringGroup.visible = this.showRays;
    this.footGroup.visible = this.showFootprint;
    this.trailGroup.visible = this.showTrails;

    if (refit || rescaled) this.frameAll(contentReach);
    this.applyVisibility();
    this.setSelected([...this.selectedIds]);
  }

  _buildCameraRig(cam, footprint) {
    const camPos = enuToScene(0, 0, cam.altitude_m);
    const s = Math.max(0.3, (this._extent || 60) * 0.022);

    const body = new THREE.Mesh(
      new THREE.ConeGeometry(s * 0.8, s * 2, 4),
      new THREE.MeshStandardMaterial({
        color: 0x4c8dff, emissive: 0x11294d, roughness: 0.35, metalness: 0.4,
      }),
    );
    body.position.copy(camPos);
    body.rotation.order = 'YXZ';
    body.rotation.y = -THREE.MathUtils.degToRad(cam.heading_deg);
    body.rotation.x = THREE.MathUtils.degToRad(cam.pitch_deg - 90);
    this.camGroup.add(body);

    // Mast to the ground, so the altitude is legible at a glance.
    this.camGroup.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([camPos, enuToScene(0, 0, 0)]),
        new THREE.LineDashedMaterial({
          color: 0x4c8dff, dashSize: s, gapSize: s * 0.7,
          transparent: true, opacity: 0.6,
        }),
      ).computeLineDistances(),
    );

    const alt = cam.altitude_m < 10 ? cam.altitude_m.toFixed(1) : cam.altitude_m.toFixed(0);
    this.camGroup.add(this._makeLabel(
      `CAM ${alt}m ${cam.pitch_deg.toFixed(0)}°`,
      enuToScene(0, 0, cam.altitude_m + this._labelSize() * 1.1),
      '#4cc2ff', this._labelSize(),
    ));

    // View frustum: one edge per footprint corner, plus the ground outline.
    if (footprint.length >= 2) {
      const corners = footprint.map((p) => enuToScene(p.east_m, p.north_m, 0.05));
      const mat = new THREE.LineBasicMaterial({
        color: 0x3fd6c8, transparent: true, opacity: 0.45,
      });
      for (const c of corners) {
        this.footGroup.add(
          new THREE.Line(new THREE.BufferGeometry().setFromPoints([camPos, c]), mat),
        );
      }
      const outline = footprint.length >= 3 ? [...corners, corners[0]] : corners;
      this.footGroup.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(outline),
        new THREE.LineBasicMaterial({ color: 0x3fd6c8, transparent: true, opacity: 0.85 }),
      ));
    }
  }

  _drawTrails(trails) {
    for (const trail of trails) {
      if (!trail.points || trail.points.length < 2) continue;
      const points = trail.points.map((p) => enuToScene(p.east_m, p.north_m, 0.08));
      this.trailGroup.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({
          color: classColorHex(trail.class_id), transparent: true, opacity: 0.6,
        }),
      ));
    }
  }

  /** Pull the orbit camera back so the whole subject is in shot. */
  frameAll(reach) {
    const r = reach || this._extent || 80;
    this.camera.position.set(r * 0.55, r * 0.75, r * 1.05);
    this.controls.target.set(0, 0, -r * 0.15);
    this.controls.update();
  }

  // --------------------------------------------------------------- selection

  setSelected(ids) {
    this.selectedIds = new Set(ids || []);
    const any = this.selectedIds.size > 0;

    for (const [id, obj] of this.objects) {
      const on = this.selectedIds.has(id);
      obj.mesh.material.opacity = on ? 0.9 : (any ? 0.34 : 0.6);
      obj.mesh.material.emissive.setHex(on ? 0x2a4a7a : 0x000000);
      obj.edges.material.color.setHex(on ? SELECT_COLOR : obj.colorHex);
      obj.edges.material.opacity = on ? 1 : (any ? 0.5 : 1);
      obj.ring.material.opacity = on ? 0.95 : (any ? 0.3 : 0.6);
      obj.ray.material.opacity = on ? 0.75 : (any ? 0.12 : 0.28);
      obj.glow.visible = on;
    }
  }

  /** Restrict which objects are shown. Pass null to show everything. */
  setVisible(ids) {
    this.visibleIds = ids ? new Set(ids) : null;
    this.applyVisibility();
  }

  applyVisibility() {
    for (const [id, obj] of this.objects) {
      const visible = this.visibleIds === null || this.visibleIds.has(id);
      obj.mesh.visible = visible;
      obj.ring.visible = visible;
      obj.ray.visible = visible && this.showRays;
      obj.label.visible = visible && this.showLabels;
      if (!visible) obj.glow.visible = false;
    }
  }

  /** Swing the orbit target onto an object, keeping the current distance. */
  focus(id) {
    const obj = this.objects.get(id);
    if (!obj) return;
    const target = obj.mesh.position.clone();
    const offset = this.camera.position.clone().sub(this.controls.target);
    this.controls.target.copy(target);
    this.camera.position.copy(target.clone().add(offset));
    this.controls.update();
  }

  setShowFootprint(show) { this.showFootprint = show; this.footGroup.visible = show; }
  setShowTrails(show) { this.showTrails = show; this.trailGroup.visible = show; }
  setShowLabels(show) {
    this.showLabels = show;
    this.applyVisibility();
  }
  setShowRays(show) {
    this.showRays = show;
    this.rayGroup.visible = show;
    this.ringGroup.visible = show;
    this.applyVisibility();
  }

  _handleClick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);

    const meshes = [...this.objects.values()]
      .filter((o) => o.mesh.visible)
      .map((o) => o.mesh);
    const hits = this.raycaster.intersectObjects(meshes, false);
    this.onSelect(hits.length ? hits[0].object.userData.detectionId : null);
  }

  // ------------------------------------------------------------------ plumbing

  /**
   * The current 3D view as a PNG blob.
   *
   * Renders once immediately beforehand rather than trusting whatever the
   * animation loop last left in the buffer, so the exported image always
   * matches what is on screen at the moment of capture.
   */
  toBlob() {
    this.renderer.render(this.scene, this.camera);
    return new Promise((resolve) =>
      this.renderer.domElement.toBlob(resolve, 'image/png'));
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return; // pane hidden
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // Note: no third argument. setSize(w, h, false) would resize the drawing
    // buffer but leave the canvas element's CSS size alone, and this renderer
    // is built inside a hidden pane where that size starts at 1x1.
    this.renderer.setSize(w, h);
  }

  _animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.onFrame?.(this.azimuthDeg);
  }

  /** Free GPU memory before rebuilding a group -- Three does not do this for us. */
  _disposeGroup(group) {
    group.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) {
          if (m.map) m.map.dispose();
          m.dispose();
        }
      }
    });
    group.clear();
  }

  reset() {
    for (const g of [this.detGroup, this.camGroup, this.footGroup,
                     this.rayGroup, this.trailGroup]) {
      this._disposeGroup(g);
    }
    this.objects.clear();
  }
}
