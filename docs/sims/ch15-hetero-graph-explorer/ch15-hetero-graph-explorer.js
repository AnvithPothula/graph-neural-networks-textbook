// Heterogeneous Graph Explorer (academic graph: papers, authors, fields, institutions)
// CANVAS_HEIGHT: 520
// Learning objective (Apply): toggle node types and select meta-paths to see which
// nodes form a query paper's meta-path neighborhood — how meta-paths define structure
// in a heterogeneous graph.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let cbPaper, cbAuthor, cbField, cbInst, mpSelect, edgeTypeToggle;

// node types: 0 paper,1 author,2 field,3 institution
const NP = 15, NA = 8, NF = 5, NI = 3;
let typeOf = [], pos = [], adj = [], edges = [];   // edges: [a,b,rel]  rel:0 cites,1 writes,2 topic,3 affil
let showEdgeTypes = false, queryPaper = -1, metapath = 'None', hoverNode = -1;
const TYPE_COL = ['#1565c0', '#fb8c00', '#43a047', '#757575'];
const REL_COL = ['#1565c0', '#fb8c00', '#43a047', '#9e9e9e'];
const TYPE_NAME = ['Paper', 'Author', 'Field', 'Institution'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildGraph();

  cbPaper = createCheckbox(' Paper', true); cbPaper.position(10, drawHeight + 8);
  cbAuthor = createCheckbox(' Author', true); cbAuthor.position(90, drawHeight + 8);
  cbField = createCheckbox(' Field', true); cbField.position(178, drawHeight + 8);
  cbInst = createCheckbox(' Institution', true); cbInst.position(252, drawHeight + 8);
  edgeTypeToggle = createCheckbox(' Show edge types', false); edgeTypeToggle.position(360, drawHeight + 8); edgeTypeToggle.changed(() => showEdgeTypes = edgeTypeToggle.checked());
  mpSelect = createSelect(); mpSelect.position(10, drawHeight + 45);
  ['None', 'PP (cites)', 'APA (co-author)', 'PAP (shared author)', 'PFP (shared field)'].forEach(o => mpSelect.option(o));
  mpSelect.changed(() => metapath = mpSelect.value().slice(0, 3).trim());

  describe('Heterogeneous academic graph with paper, author, field, and institution ' +
    'nodes. Toggle node types on and off, click a paper to make it the query node, and ' +
    'select a meta-path (PP, APA, PAP, PFP) to highlight that paper\'s meta-path neighborhood. ' +
    'Toggle edge-type coloring to distinguish cites / writes / has-topic / affiliated relations.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18); text('Heterogeneous Graph Explorer', canvasWidth / 2, 6);

  let vis = [cbPaper.checked(), cbAuthor.checked(), cbField.checked(), cbInst.checked()];
  let hl = computeHighlight();
  hoverNode = -1;
  drawGraph(0, 30, canvasWidth, drawHeight - 38, vis, hl);
  drawLegend();
  if (hoverNode >= 0) drawTip();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13);
  text('Meta-path:', 10, drawHeight + 38);
  if (queryPaper < 0) { textAlign(RIGHT, CENTER); fill('#888'); text('click a paper to set the query node', canvasWidth - margin, drawHeight + 38); }
}

function nodeXY(i, gx, gy, gw, gh) { return { x: gx + 24 + pos[i].x * (gw - 48), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawGraph(gx, gy, gw, gh, vis, hl) {
  // edges
  for (let e of edges) {
    if (!vis[typeOf[e[0]]] || !vis[typeOf[e[1]]]) continue;
    let a = nodeXY(e[0], gx, gy, gw, gh), b = nodeXY(e[1], gx, gy, gw, gh);
    let dim = (queryPaper >= 0 && metapath !== 'Non' && !(hl.edges.has(ek(e[0], e[1]))));
    stroke(showEdgeTypes ? color(REL_COL[e[2]]) : color(170)); if (dim) drawingContext.globalAlpha = 0.12;
    strokeWeight(hl.edges.has(ek(e[0], e[1])) ? 3 : 1.2); line(a.x, a.y, b.x, b.y); drawingContext.globalAlpha = 1;
  }
  // nodes
  for (let i = 0; i < typeOf.length; i++) {
    if (!vis[typeOf[i]]) continue;
    let p = nodeXY(i, gx, gy, gw, gh), t = typeOf[i];
    if (dist(mouseX, mouseY, p.x, p.y) <= 12) hoverNode = i;
    let dim = (queryPaper >= 0 && metapath !== 'Non' && !hl.nodes.has(i) && !hl.dim.has(i) && i !== queryPaper);
    push(); if (dim) drawingContext.globalAlpha = 0.2; else if (hl.dim.has(i)) drawingContext.globalAlpha = 0.55;
    let isQ = i === queryPaper;
    stroke(isQ ? '#d32f2f' : 50); strokeWeight(isQ ? 3 : 1);
    fill(hl.nodes.has(i) ? '#ffd54f' : TYPE_COL[t]);
    let r = t === 0 ? 11 : t === 1 ? 8 : 8;
    if (t === 0) circle(p.x, p.y, r * 2);
    else if (t === 1) circle(p.x, p.y, r * 2);
    else if (t === 2) { rectMode(CENTER); rect(p.x, p.y, r * 2, r * 2); rectMode(CORNER); }
    else { push(); translate(p.x, p.y); rotate(QUARTER_PI); rectMode(CENTER); rect(0, 0, r * 1.7, r * 1.7); rectMode(CORNER); pop(); }
    pop();
  }
}

function drawLegend() {
  let x = canvasWidth - 150, y = 40;
  // opaque backing so graph nodes/edges don't show through the key
  fill(255, 255, 255, 240); stroke('silver'); strokeWeight(1);
  rect(x - 14, y - 14, 152, 4 * 16 + 12, 6);
  noStroke(); textAlign(LEFT, CENTER); textSize(11);
  for (let t = 0; t < 4; t++) { fill(TYPE_COL[t]); circle(x, y + t * 16, 9); fill('black'); text(TYPE_NAME[t], x + 10, y + t * 16); }
}

function drawTip() {
  let i = hoverNode, t = typeOf[i];
  let lines = [TYPE_NAME[t] + ' ' + localId(i), 'type: ' + TYPE_NAME[t], 'degree: ' + adj[i].length];
  let w = 150, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 30, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- meta-path highlight ----------

function ek(a, b) { return Math.min(a, b) + '_' + Math.max(a, b); }
function neighborsByRel(i, rel) { return edges.filter(e => (e[0] === i || e[1] === i) && e[2] === rel).map(e => e[0] === i ? e[1] : e[0]); }

function computeHighlight() {
  let nodes = new Set(), dim = new Set(), edgs = new Set();
  if (queryPaper < 0 || metapath === 'Non') return { nodes, dim, edges: edgs };
  let p = queryPaper;
  if (metapath === 'PP') {
    for (let q of neighborsByRel(p, 0)) { nodes.add(q); edgs.add(ek(p, q)); }
  } else if (metapath === 'APA' || metapath === 'PAP') {
    let authors = neighborsByRel(p, 1);
    for (let a of authors) { dim.add(a); edgs.add(ek(p, a)); for (let q of neighborsByRel(a, 1)) if (q !== p) { nodes.add(q); edgs.add(ek(a, q)); } }
  } else if (metapath === 'PFP') {
    let fields = neighborsByRel(p, 2);
    for (let f of fields) { dim.add(f); edgs.add(ek(p, f)); for (let q of neighborsByRel(f, 2)) if (q !== p) { nodes.add(q); edgs.add(ek(f, q)); } }
  }
  return { nodes, dim, edges: edgs };
}

function localId(i) { let t = typeOf[i], base = [0, NP, NP + NA, NP + NA + NF][t]; return i - base; }

// ---------- graph construction ----------

function buildGraph() {
  let rng = mulberry32(7);
  typeOf = [];
  for (let i = 0; i < NP; i++) typeOf.push(0);
  for (let i = 0; i < NA; i++) typeOf.push(1);
  for (let i = 0; i < NF; i++) typeOf.push(2);
  for (let i = 0; i < NI; i++) typeOf.push(3);
  let total = NP + NA + NF + NI;
  edges = [];
  const A0 = NP, F0 = NP + NA, I0 = NP + NA + NF;
  // writes: each paper 1-2 authors
  for (let p = 0; p < NP; p++) { let na = 1 + (rng() < 0.5 ? 1 : 0); for (let k = 0; k < na; k++) edges.push([A0 + Math.floor(rng() * NA), p, 1]); }
  // cites: ~10 citations among papers
  for (let c = 0; c < 10; c++) { let a = Math.floor(rng() * NP), b = Math.floor(rng() * NP); if (a !== b) edges.push([a, b, 0]); }
  // topic: each paper 1 field
  for (let p = 0; p < NP; p++) edges.push([p, F0 + Math.floor(rng() * NF), 2]);
  // affiliated: each author 1 institution
  for (let a = 0; a < NA; a++) edges.push([A0 + a, I0 + Math.floor(rng() * NI), 3]);
  adj = Array.from({ length: total }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  pos = springLayout(total);
}

function springLayout(n) {
  let p = []; for (let i = 0; i < n; i++) { let a = TWO_PI * i / n; p.push({ x: 0.5 + 0.4 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a) }); }
  for (let it = 0; it < 220; it++) {
    let fx = new Array(n).fill(0), fy = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), r = 0.0006 / d2; fx[i] += dx / d * r; fy[i] += dy / d * r; fx[j] -= dx / d * r; fy[j] -= dy / d * r; }
    for (let e of edges) { let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, s = (d - 0.1) * 0.02; fx[e[0]] += dx / d * s; fy[e[0]] += dy / d * s; fx[e[1]] -= dx / d * s; fy[e[1]] -= dy / d * s; }
    for (let i = 0; i < n; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}

function mousePressed() {
  if (mouseY < 28 || mouseY > drawHeight) return;
  for (let i = 0; i < NP; i++) { let p = nodeXY(i, 0, 30, canvasWidth, drawHeight - 38); if (dist(mouseX, mouseY, p.x, p.y) <= 13) { queryPaper = (queryPaper === i ? -1 : i); return; } }
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
