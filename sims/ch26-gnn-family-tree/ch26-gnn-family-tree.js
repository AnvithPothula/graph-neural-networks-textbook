// GNN Architecture Family Tree
// CANVAS_HEIGHT: 600
// Learning objective (Analyze): locate any major GNN architecture within the design-space
// taxonomy, trace its lineage, and find the chapter that covers it.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 960;
let drawHeight = 540;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let dimSelect, resetButton;
let nodes = [], edges = [], pos = [], sel = -1, hoverN = -1, filterDim = 'all';

const DIMS = {
  homog: { c: '#1565c0', label: 'Homogeneous' }, kg: { c: '#ef6c00', label: 'Knowledge graph' },
  hetero: { c: '#6a1b9a', label: 'Heterogeneous' }, temporal: { c: '#2e7d32', label: 'Temporal' },
  gen: { c: '#c62828', label: 'Generative' }, ssl: { c: '#00838f', label: 'Self-supervised / FM' },
};

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildTree();

  dimSelect = createSelect(); dimSelect.position(150, drawHeight + 14);
  dimSelect.option('All dimensions', 'all'); Object.keys(DIMS).forEach(k => dimSelect.option(DIMS[k].label, k));
  dimSelect.changed(() => filterDim = dimSelect.value());
  resetButton = createButton('Reset'); resetButton.position(360, drawHeight + 14); resetButton.mousePressed(() => { sel = -1; filterDim = 'all'; dimSelect.value('all'); });

  describe('An explorable taxonomy of GNN architectures. About 25 models are arranged by a ' +
    'force layout, colored by design dimension (homogeneous, knowledge-graph, heterogeneous, ' +
    'temporal, generative, self-supervised). Directed edges show lineage (extends / inspired by). ' +
    'Click a model for its year, key innovation, and the chapter that covers it; filter by ' +
    'dimension to isolate a family.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('#f8f8fc'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  stroke(234); strokeWeight(1); for (let x = 0; x < canvasWidth; x += 30) line(x, 0, x, drawHeight); for (let y = 0; y < drawHeight; y += 30) line(0, y, canvasWidth, y);
  noStroke();

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18); text('GNN Architecture Family Tree', canvasWidth / 2, 6);

  let lineage = sel >= 0 ? lineageSet(sel) : null;
  hoverN = -1;
  // edges
  for (let e of edges) {
    if (!dimVisible(e[0]) || !dimVisible(e[1])) continue;
    let a = ptOf(e[0]), b = ptOf(e[1]);
    let on = lineage && (lineage.nodes.has(e[0]) && lineage.nodes.has(e[1]));
    stroke(on ? '#333' : color(150, 150, 150, sel >= 0 ? 40 : 110)); strokeWeight(on ? 2 : 1);
    drawArrowSeg(a.x, a.y, b.x, b.y, nodeR(e[1]) + 2);
    if (on) { noStroke(); fill('#333'); textAlign(CENTER, CENTER); textSize(8); text(e[2], (a.x + b.x) / 2, (a.y + b.y) / 2 - 5); }
  }
  // nodes
  for (let i = 0; i < nodes.length; i++) {
    if (!dimVisible(i)) continue;
    let p = ptOf(i), r = nodeR(i);
    if (dist(mouseX, mouseY, p.x, p.y) <= r) hoverN = i;
    let dim = lineage && !lineage.nodes.has(i);
    push(); if (dim) drawingContext.globalAlpha = 0.25;
    stroke(i === sel ? 'black' : '#fff'); strokeWeight(i === sel ? 3 : 1.5); fill(DIMS[nodes[i].dim].c);
    circle(p.x, p.y, r * 2);
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(9); text(nodes[i].name, p.x, p.y);
    pop();
  }
  drawLegend();
  if (sel >= 0) drawCard(sel); else if (hoverN >= 0) drawMini(hoverN);

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13); text('Filter:', 100, drawHeight + 28);
}

function ptOf(i) { return { x: margin + pos[i].x * (canvasWidth - 2 * margin), y: 30 + pos[i].y * (drawHeight - 50) }; }
function nodeR(i) { return 12 + Math.min(nodes[i].childCount, 5) * 1.6; }
function dimVisible(i) { return filterDim === 'all' || nodes[i].dim === filterDim; }

function drawLegend() {
  let x = margin + 4, y = drawHeight - 14; noStroke(); textAlign(LEFT, CENTER); textSize(10);
  let keys = Object.keys(DIMS), xx = x;
  for (let k of keys) { fill(DIMS[k].c); circle(xx, y, 9); fill('#333'); text(DIMS[k].label, xx + 8, y); xx += textWidth(DIMS[k].label) + 26; }
}

function drawMini(i) {
  let lines = [nodes[i].name + ' (' + nodes[i].year + ')', DIMS[nodes[i].dim].label, 'click for details'];
  tip(lines, 150);
}
function drawCard(i) {
  let n = nodes[i], w = 230, h = 150, p = ptOf(i);
  let cx = constrain(p.x + 20, 4, canvasWidth - w - 4), cy = constrain(p.y - h / 2, 30, drawHeight - h - 4);
  fill(255, 255, 255, 252); stroke(DIMS[n.dim].c); strokeWeight(2); rect(cx, cy, w, h, 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14); textStyle(BOLD); text(n.name + '  (' + n.year + ')', cx + 10, cy + 8); textStyle(NORMAL);
  fill(DIMS[n.dim].c); textSize(11); text(DIMS[n.dim].label, cx + 10, cy + 28);
  fill('#333'); textSize(11); text(n.innov, cx + 10, cy + 46, w - 20, 70);
  // chapter badge
  fill('#1565c0'); rect(cx + 10, cy + h - 30, 74, 20, 4); noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(11); text('Chapter ' + n.ch, cx + 47, cy + h - 20);
  let anc = ancestors(i).size, desc = descendants(i).size;
  fill('#666'); textAlign(LEFT, CENTER); textSize(10); text('ancestors: ' + anc + '   descendants: ' + desc, cx + 92, cy + h - 20);
}
function tip(lines, maxw) {
  let w = maxw, h = lines.length * 16 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(11.5);
  for (let k = 0; k < lines.length; k++) { if (k === 0) textStyle(BOLD); else textStyle(NORMAL); text(lines[k], tx + 8, ty + 6 + k * 16, w - 16); }
  textStyle(NORMAL);
}

// ---------- lineage ----------

function lineageSet(i) { let nodesSet = new Set([i]); for (let a of ancestors(i)) nodesSet.add(a); for (let d of descendants(i)) nodesSet.add(d); return { nodes: nodesSet }; }
function ancestors(i) { let set = new Set(); let stack = [i]; while (stack.length) { let u = stack.pop(); for (let e of edges) if (e[1] === u && !set.has(e[0])) { set.add(e[0]); stack.push(e[0]); } } return set; }
function descendants(i) { let set = new Set(); let stack = [i]; while (stack.length) { let u = stack.pop(); for (let e of edges) if (e[0] === u && !set.has(e[1])) { set.add(e[1]); stack.push(e[1]); } } return set; }

function drawArrowSeg(x1, y1, x2, y2, rad) { let ang = atan2(y2 - y1, x2 - x1), ex = x2 - cos(ang) * rad, ey = y2 - sin(ang) * rad; line(x1, y1, ex, ey); push(); translate(ex, ey); rotate(ang); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -6, 2, -6, -2); pop(); }

// ---------- data ----------

function buildTree() {
  // [name, year, dim, chapter, innovation, [parents]]
  let D = [
    ['GCN', 2017, 'homog', 6, 'Spectral-motivated symmetric-normalized neighborhood averaging.', []],
    ['GraphSAGE', 2017, 'homog', 7, 'Inductive sampling + aggregation over fixed-size neighborhoods.', ['GCN']],
    ['GAT', 2018, 'homog', 7, 'Attention-weighted neighbor aggregation.', ['GCN']],
    ['GIN', 2019, 'homog', 10, 'Sum aggregation as powerful as the 1-WL test.', ['GCN']],
    ['SGC', 2019, 'homog', 20, 'Removes nonlinearities → precomputed feature diffusion.', ['GCN']],
    ['APPNP', 2019, 'homog', 8, 'Personalized PageRank propagation decoupled from transform.', ['GCN']],
    ['PNA', 2020, 'homog', 10, 'Multiple aggregators + degree scalers.', ['GIN']],
    ['SIGN', 2020, 'homog', 20, 'Precompute multi-hop diffusion, train an MLP.', ['SGC']],
    ['Cluster-GCN', 2019, 'homog', 20, 'Subgraph mini-batching via graph clustering.', ['GCN']],
    ['GraphSAINT', 2020, 'homog', 20, 'Unbiased subgraph-sampling mini-batches.', ['GCN']],
    ['Graphormer', 2021, 'homog', 11, 'Transformer with degree/spatial/edge encodings.', ['GAT']],
    ['SAN', 2021, 'homog', 11, 'Spectral attention graph transformer.', ['GAT']],
    ['GPS', 2022, 'homog', 11, 'Hybrid local MPNN + global attention.', ['Graphormer']],
    ['TransE', 2013, 'kg', 12, 'Relations as translations: h + r ≈ t.', []],
    ['RotatE', 2019, 'kg', 12, 'Relations as rotations in complex space.', ['TransE']],
    ['Query2Box', 2020, 'kg', 13, 'Box embeddings for conjunctive queries.', ['TransE']],
    ['NBFNet', 2021, 'kg', 14, 'Neural Bellman-Ford path formulation.', ['GCN']],
    ['ULTRA', 2023, 'kg', 14, 'Inductive relation transfer across KGs.', ['NBFNet']],
    ['RGCN', 2018, 'hetero', 15, 'Relation-specific weight matrices.', ['GCN']],
    ['HAN', 2019, 'hetero', 15, 'Meta-path + node-level hierarchical attention.', ['GAT']],
    ['HGT', 2020, 'hetero', 15, 'Type-aware heterogeneous transformer.', ['GAT']],
    ['GraphVAE', 2018, 'gen', 23, 'VAE generating adjacency matrices.', ['GCN']],
    ['GraphRNN', 2018, 'gen', 21, 'Autoregressive node-by-node generation.', []],
    ['GDSS', 2022, 'gen', 21, 'Score-based diffusion for graphs.', ['GraphRNN']],
    ['TGN', 2020, 'temporal', 22, 'Memory modules over event streams.', ['GAT']],
    ['TGAT', 2020, 'temporal', 22, 'Temporal attention with time encoding.', ['GAT']],
    ['DGI', 2019, 'ssl', 24, 'Maximize local-global mutual information.', ['GCN']],
    ['GraphCL', 2020, 'ssl', 24, 'Contrastive learning over graph augmentations.', ['GIN']],
    ['PRODIGY', 2023, 'ssl', 25, 'In-context learning over prompt graphs.', ['DGI']],
    ['OFA', 2024, 'ssl', 14, 'One-for-all text-attributed foundation model.', ['DGI']],
  ];
  nodes = D.map(d => ({ name: d[0], year: d[1], dim: d[2], ch: d[3], innov: d[4], parents: d[5], childCount: 0 }));
  let idx = {}; nodes.forEach((n, i) => idx[n.name] = i);
  edges = [];
  for (let i = 0; i < nodes.length; i++) for (let p of nodes[i].parents) if (p in idx) { edges.push([idx[p], i, 'extends']); nodes[idx[p]].childCount++; }
  pos = springLayout();
}

function springLayout() {
  let n = nodes.length, rng = mulberry32(3), p = [];
  for (let i = 0; i < n; i++) { let a = TWO_PI * i / n; p.push({ x: 0.5 + 0.4 * Math.cos(a) + (rng() - 0.5) * 0.05, y: 0.5 + 0.4 * Math.sin(a) }); }
  for (let it = 0; it < 320; it++) {
    let fx = new Array(n).fill(0), fy = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), r = 0.0009 / d2; fx[i] += dx / d * r; fy[i] += dy / d * r; fx[j] -= dx / d * r; fy[j] -= dy / d * r; }
    for (let e of edges) { let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, s = (d - 0.1) * 0.02; fx[e[0]] += dx / d * s; fy[e[0]] += dy / d * s; fx[e[1]] -= dx / d * s; fy[e[1]] -= dy / d * s; }
    for (let i = 0; i < n; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: 0.04 + 0.92 * (q.x - mnx) / (mxx - mnx + 1e-9), y: 0.04 + 0.92 * (q.y - mny) / (mxy - mny + 1e-9) }));
}

function mousePressed() {
  if (mouseY < 24 || mouseY > drawHeight) return;
  for (let i = 0; i < nodes.length; i++) { if (!dimVisible(i)) continue; let p = ptOf(i); if (dist(mouseX, mouseY, p.x, p.y) <= nodeR(i) + 2) { sel = (sel === i ? -1 : i); return; } }
  sel = -1;
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
