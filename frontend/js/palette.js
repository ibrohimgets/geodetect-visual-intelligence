/**
 * One colour per object class, plus a glyph per category.
 *
 * The same object must look the same on the image, the map, the 3D scene and
 * the table, otherwise you cannot follow it from one view to the next -- which
 * is the whole point of the app. Keeping the lookup in one module is what
 * guarantees that.
 *
 * These hues are picked for a NEAR-BLACK background: light and saturated, so
 * they stay distinct as an 8px swatch, a 1px timeline tick or a 2px box
 * outline, and all clear 4.5:1 against the panel ground.
 */

const PALETTE = [
  '#4cc2ff', // cyan
  '#ffb454', // amber
  '#57d97f', // green
  '#ff7ab8', // pink
  '#b48cff', // violet
  '#ff8a4c', // orange
  '#3fd6c8', // teal
  '#ff6b6b', // red
  '#b5e05a', // lime
  '#6aa9ff', // blue
  '#ff7a7a', // rose
  '#ffd24c', // yellow
];

/** Stable colour for a COCO class id. The same id always gives the same hue. */
export function classColor(classId) {
  return PALETTE[((classId % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

/** The same colour as an integer, which is what Three.js materials want. */
export function classColorHex(classId) {
  return parseInt(classColor(classId).slice(1), 16);
}

/** Add an alpha channel to one of our hex colours, for canvas fills. */
export function withAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

// Selection reads as white on this ground, the way a highlighted wireframe
// does in a CAD viewport.
export const SELECTED_COLOR = '#ffffff';

/* ---------------------------------------------------------------- glyphs */

/**
 * A small icon per category, drawn inside map markers.
 *
 * Colour alone stops being readable once a dozen classes are on screen, so the
 * marker carries a shape as well. Paths are sized for a 24x24 viewBox.
 */
const GLYPHS = {
  people: 'M12 7a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM5 22v-6a7 7 0 0 1 14 0v6',
  vehicles: 'M4 16h16M6 16v2M18 16v2M5 16l1.6-5A2 2 0 0 1 8.5 9.6h7A2 2 0 0 1 17.4 11L19 16',
  animals: 'M5 11a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm14 0a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM8.5 7.5a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm7 0a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM12 21c-3 0-5-1.8-5-4.2 0-2 2-4.3 5-4.3s5 2.3 5 4.3C17 19.2 15 21 12 21Z',
  furniture: 'M4 18v-6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v6M4 14h16M7 18v2M17 18v2',
  outdoor: 'M12 2v20M7 6h10M7 11h10',
  electronics: 'M3 5h18v11H3zM8 20h8M12 16v4',
  'personal items': 'M6 8h12v13H6zM9 8V5a3 3 0 0 1 6 0v3',
  kitchen: 'M8 3v7a4 4 0 0 0 8 0V3M12 14v7',
  food: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 5v8',
  sports: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM3.5 9h17M3.5 15h17M12 3c3 4 3 14 0 18M12 3c-3 4-3 14 0 18',
  other: 'M12 4.5 20 12l-8 7.5L4 12z',
};

export function categoryGlyph(category) {
  return GLYPHS[category] || GLYPHS.other;
}

/**
 * The category a class belongs to.
 *
 * The backend is the source of truth for this mapping and serves it from
 * /api/categories; `primeCategories` loads it once at start-up so the rest of
 * the frontend can look a class up synchronously while drawing.
 */
let CLASS_TO_CATEGORY = {};

export function primeCategories(categories) {
  CLASS_TO_CATEGORY = {};
  for (const cat of categories) {
    for (const cls of cat.classes) CLASS_TO_CATEGORY[cls] = cat.id;
  }
}

export function categoryOf(className) {
  return CLASS_TO_CATEGORY[className] || 'other';
}
