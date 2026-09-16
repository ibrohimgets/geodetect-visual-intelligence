/**
 * The GIS view: detections placed on a real basemap, in WGS84.
 *
 * This is where the geometry gets audited. Every marker sits at a latitude and
 * longitude computed from a pixel, so if the projection is working the objects
 * land on the road and the pavement rather than inside buildings -- switch to
 * the satellite basemap to check exactly that.
 *
 * Built on Leaflet, loaded as a global from the CDN in index.html.
 */

import { classColor, categoryOf, categoryGlyph } from './palette.js';

// Above this many markers, cluster them. Below it, clustering would merge the
// handful of objects in a street scene into a single blob for no benefit.
const CLUSTER_THRESHOLD = 30;

const METRES_PER_DEG_LAT = 111132.0;

/** Offset a lat/lon by a distance and compass bearing. Local tangent plane. */
function destination(lat, lon, bearingDeg, metres) {
  const rad = (bearingDeg * Math.PI) / 180;
  const dNorth = metres * Math.cos(rad);
  const dEast = metres * Math.sin(rad);
  const mPerDegLon = METRES_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180);
  return [
    lat + dNorth / METRES_PER_DEG_LAT,
    lon + (Math.abs(mPerDegLon) > 1e-9 ? dEast / mPerDegLon : 0),
  ];
}

export class MapView {
  constructor(elementId, { onSelect } = {}) {
    this.onSelect = onSelect || (() => {});

    this.map = L.map(elementId, {
      zoomControl: true,
      attributionControl: true,
      preferCanvas: false, // divIcon markers need DOM
    }).setView([47.3769, 8.5417], 18);

    // Basemaps that need no API key. Esri's tile services are open for this
    // kind of use with attribution. Carto's free tiles now watermark every
    // tile with "API KEY REQUIRED", so they are deliberately absent.
    this.baseLayers = {
      Satellite: L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        { maxZoom: 19, attribution: 'Tiles &copy; Esri, Maxar, Earthstar Geographics' },
      ),
      Dark: L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        { maxZoom: 16, attribution: 'Tiles &copy; Esri' },
      ),
      Streets: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19, attribution: '&copy; OpenStreetMap contributors',
      }),
    };
    this.baseLayers.Dark.addTo(this.map);
    L.control.layers(this.baseLayers, null, { position: 'topright' }).addTo(this.map);
    L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(this.map);

    // Separate layers so each can be rebuilt or hidden independently.
    this.cameraLayer = L.layerGroup().addTo(this.map);
    this.footprintLayer = L.layerGroup().addTo(this.map);
    this.trailLayer = L.layerGroup().addTo(this.map);
    this.plainLayer = L.layerGroup().addTo(this.map);
    this.clusterLayer = L.markerClusterGroup
      ? L.markerClusterGroup({ maxClusterRadius: 45, disableClusteringAtZoom: 20 })
      : L.layerGroup();

    this.markers = new Map(); // detection id -> marker
    this.selectedIds = new Set();
    this.visibleIds = null;
    this.showFootprint = true;
    this.showTrails = true;
    this._hasFitted = false;
    this._pendingFit = null;
  }

  /**
   * Leaflet measures its container when created and when told to. This map is
   * built inside a hidden pane, so until that pane opens it believes it is 0x0
   * -- and fitting bounds to a 0x0 viewport gives a nonsense zoom. A fit asked
   * for while hidden is parked and replayed here, once there is a real size.
   */
  invalidate() {
    this.map.invalidateSize();
    this._tryFit();
  }

  _requestFit(bounds) { this._pendingFit = bounds; this._tryFit(); }

  _tryFit() {
    if (!this._pendingFit) return;
    const el = this.map.getContainer();
    if (!el.clientWidth || !el.clientHeight) return; // still hidden
    this.map.invalidateSize();
    this.map.fitBounds(this._pendingFit.pad(0.25), { maxZoom: 19 });
    this._pendingFit = null;
    this._hasFitted = true;
  }

  setShowFootprint(show) {
    this.showFootprint = show;
    this.map[show ? 'addLayer' : 'removeLayer'](this.footprintLayer);
  }

  setShowTrails(show) {
    this.showTrails = show;
    this.map[show ? 'addLayer' : 'removeLayer'](this.trailLayer);
  }

  // ------------------------------------------------------------------ render

  render(data, { refit = false, trails = null } = {}) {
    this.cameraLayer.clearLayers();
    this.footprintLayer.clearLayers();
    this.plainLayer.clearLayers();
    this.clusterLayer.clearLayers();
    this.trailLayer.clearLayers();
    this.markers.clear();

    const cam = data.camera;
    const points = [[cam.lat, cam.lon]];

    this._drawCamera(cam, data.footprint || []);
    if (data.footprint?.length >= 2) {
      for (const p of data.footprint) points.push([p.lat, p.lon]);
    }

    // Choose plain markers or a cluster group based on how many there are.
    const located = data.detections.filter((d) => d.world.valid);
    const useCluster = located.length > CLUSTER_THRESHOLD && L.markerClusterGroup;
    this.map.removeLayer(this.clusterLayer);
    this.map.removeLayer(this.plainLayer);
    const target = useCluster ? this.clusterLayer : this.plainLayer;
    this.map.addLayer(target);

    for (const det of located) {
      const marker = this._makeMarker(det);
      marker.addTo(target);
      this.markers.set(det.id, marker);
      points.push([det.world.lat, det.world.lon]);
    }

    if (trails) this._drawTrails(trails);

    // Fit for a new scene, or when a change pushed results off screen.
    if (points.length) {
      const bounds = L.latLngBounds(points);
      const offscreen = this._hasFitted && !this._pendingFit
        && !this.map.getBounds().contains(bounds);
      if (refit || !this._hasFitted || offscreen) this._requestFit(bounds);
    }

    this.applyVisibility();
    this.setSelected([...this.selectedIds]);
  }

  _makeMarker(det) {
    const color = classColor(det.class_id);
    const category = categoryOf(det.class_name);
    const name = det.track_id != null
      ? `${det.class_name}_${String(det.track_id).padStart(2, '0')}`
      : det.class_name;

    const icon = L.divIcon({
      className: '',
      html: `<div class="obj-marker" style="background:${color}">
               <svg viewBox="0 0 24 24"><path d="${categoryGlyph(category)}"/></svg>
             </div>`,
      iconSize: [22, 22],
      iconAnchor: [11, 11],
    });

    const marker = L.marker([det.world.lat, det.world.lon], { icon, riseOnHover: true });
    marker.bindPopup(
      `<b>${name}</b> <span style="color:#78859a">#${det.id}</span>
       <div class="popup-grid">
         <span>confidence</span><span>${(det.confidence * 100).toFixed(1)}%</span>
         <span>lat, lon</span><span>${det.world.lat.toFixed(6)}, ${det.world.lon.toFixed(6)}</span>
         <span>local ENU</span><span>E ${det.world.east_m.toFixed(1)} / N ${det.world.north_m.toFixed(1)} m</span>
         <span>distance</span><span>${det.world.ground_range_m.toFixed(1)} m @ ${det.world.bearing_deg.toFixed(0)}&deg;</span>
         <span>size est.</span><span>${det.world.est_width_m.toFixed(1)} &times; ${det.world.est_height_m.toFixed(1)} m</span>
         <span>pixel</span><span>${det.bbox.cx.toFixed(0)}, ${det.bbox.cy.toFixed(0)}</span>
       </div>
       ${det.world.source === 'lidar'
         ? `<div class="popup-src" style="color:var(--ok)">measured &middot; ${det.world.lidar_points} lidar returns &middot; depth ${det.world.depth_m.toFixed(1)} m</div>`
         : '<div class="popup-src" style="color:var(--warn)">estimated &middot; projected onto an assumed flat ground plane</div>'}`,
    );
    marker.on('click', () => this.onSelect(det.id));
    return marker;
  }

  _drawCamera(cam, footprint) {
    // Camera body with a heading arrow.
    const icon = L.divIcon({
      className: '',
      html: `
        <div style="position:relative;width:30px;height:30px">
          <div style="position:absolute;inset:0;transform:rotate(${cam.heading_deg}deg)">
            <div style="position:absolute;left:50%;top:-4px;transform:translateX(-50%);
                        width:0;height:0;border-left:6px solid transparent;
                        border-right:6px solid transparent;border-bottom:11px solid #0b6bcb"></div>
          </div>
          <div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                      width:15px;height:15px;border-radius:50%;background:#0b6bcb;
                      border:2.5px solid #fff;box-shadow:0 1px 5px rgba(0,0,0,.4)"></div>
        </div>`,
      iconSize: [30, 30],
      iconAnchor: [15, 15],
    });

    L.marker([cam.lat, cam.lon], { icon, zIndexOffset: 1000 })
      .bindPopup(
        `<b>Camera</b>
         <div class="popup-grid">
           <span>altitude</span><span>${cam.altitude_m.toFixed(1)} m AGL</span>
           <span>pitch</span><span>${cam.pitch_deg.toFixed(1)}&deg; below horizon</span>
           <span>heading</span><span>${cam.heading_deg.toFixed(1)}&deg;</span>
           <span>FOV</span><span>${cam.hfov_deg.toFixed(1)}&deg; H / ${cam.vfov_deg.toFixed(1)}&deg; V</span>
           <span>position</span><span>${cam.lat.toFixed(6)}, ${cam.lon.toFixed(6)}</span>
         </div>`,
      )
      .addTo(this.cameraLayer);

    // Field-of-view wedge: the horizontal FOV swept around the heading, out to
    // however far the footprint actually reaches.
    let reach = 40;
    for (const p of footprint) {
      reach = Math.max(reach, Math.hypot(p.east_m, p.north_m));
    }
    reach = Math.min(reach, cam.max_range_m || 500);

    const half = cam.hfov_deg / 2;
    const arc = [[cam.lat, cam.lon]];
    for (let a = -half; a <= half; a += Math.max(1, cam.hfov_deg / 24)) {
      arc.push(destination(cam.lat, cam.lon, cam.heading_deg + a, reach));
    }
    arc.push(destination(cam.lat, cam.lon, cam.heading_deg + half, reach));

    L.polygon(arc, {
      color: '#0b6bcb', weight: 1, opacity: 0.5,
      fillColor: '#0b6bcb', fillOpacity: 0.07,
      dashArray: '4,4', interactive: false,
    }).addTo(this.footprintLayer);

    // The true ground footprint from the projected image corners. An oblique
    // shot whose top corners are sky gives fewer than 4 points, so we draw an
    // open line rather than pretend we have a closed polygon.
    if (footprint.length >= 2) {
      const ring = footprint.map((p) => [p.lat, p.lon]);
      const shape = ring.length >= 3
        ? L.polygon(ring, {
            color: '#0891b2', weight: 1.5, opacity: 0.9,
            fillColor: '#0891b2', fillOpacity: 0.1, interactive: false,
          })
        : L.polyline(ring, { color: '#0891b2', weight: 1.5, opacity: 0.9, interactive: false });
      shape.addTo(this.footprintLayer);
    }
  }

  _drawTrails(trails) {
    for (const trail of trails) {
      if (!trail.points || trail.points.length < 2) continue;
      const color = classColor(trail.class_id);
      const latlngs = trail.points.map((p) => [p.lat, p.lon]);

      L.polyline(latlngs, {
        color, weight: 2.5, opacity: 0.55, lineCap: 'round', interactive: false,
      }).addTo(this.trailLayer);

      // A small dot at the start, so the direction of travel is readable.
      L.circleMarker(latlngs[0], {
        radius: 3, color: '#fff', weight: 1.5,
        fillColor: color, fillOpacity: 0.9, interactive: false,
      }).addTo(this.trailLayer);
    }
  }

  // --------------------------------------------------------------- selection

  setSelected(ids) {
    this.selectedIds = new Set(ids || []);
    for (const [id, marker] of this.markers) {
      const el = marker.getElement?.()?.querySelector('.obj-marker');
      if (el) el.classList.toggle('selected', this.selectedIds.has(id));
    }
  }

  /** Restrict which markers are on the map. Pass null to show everything. */
  setVisible(ids) {
    this.visibleIds = ids ? new Set(ids) : null;
    this.applyVisibility();
  }

  applyVisibility() {
    for (const [id, marker] of this.markers) {
      const el = marker.getElement?.();
      if (!el) continue;
      const hidden = this.visibleIds !== null && !this.visibleIds.has(id);
      el.style.display = hidden ? 'none' : '';
    }
  }

  /** Centre on one object without changing zoom. */
  focus(id) {
    const marker = this.markers.get(id);
    if (marker) this.map.panTo(marker.getLatLng());
  }

  /** Zoom to fit a set of objects -- used when the assistant selects some. */
  fitTo(ids) {
    const points = ids
      .map((id) => this.markers.get(id))
      .filter(Boolean)
      .map((m) => m.getLatLng());
    if (points.length) {
      this._requestFit(L.latLngBounds(points));
    }
  }

  reset() {
    for (const layer of [this.cameraLayer, this.footprintLayer, this.plainLayer,
                         this.clusterLayer, this.trailLayer]) {
      layer.clearLayers();
    }
    this.markers.clear();
    this._hasFitted = false;
    this._pendingFit = null;
  }
}
