// WL Color Refinement Visualizer (expressiveness / GIN motivation)
// CANVAS_HEIGHT: 520
// Learning objective (Apply, Bloom L3): run 1-WL color refinement on graph pairs
// it CAN and CANNOT distinguish, with per-graph color histograms making the
// power and limits of neighborhood aggregation tangible.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 460;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let stepButton, runButton, resetButton, pairSelect;

let g1, g2, step = 0, prevDistinct = 0, converged = false;
let statusText = '', statusColor = '#444';
let selLabel = null, hoverInfo = null;
const MAX_STEPS = 8;

const PALETTE = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                 '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#9A6324', '#000075'];
const NODE_R = 18;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  stepButton = createButton('Step');
  stepButton.position(10, drawHeight + 8);
  stepButton.mousePressed(doStep);

  runButton = createButton('Run to convergence');
  runButton.position(64, drawHeight + 8);
  runButton.mousePressed(runToConvergence);

  resetButton = createButton('Reset');
  resetButton.position(210, drawHeight + 8);
  resetButton.mousePressed(initPair);

  pairSelect = createSelect();
  pairSelect.position(278, drawHeight + 8);
  pairSelect.option('Path P₄ vs. Star K₁,₃ (distinguishable)');
  pairSelect.option('Prism vs. K₃,₃ — 3-regular (WL fails)');
  pairSelect.option('6-cycle vs. 2 triangles — 2-regular (WL fails)');
  pairSelect.changed(initPair);

  initPair();
  describe('1-WL color refinement on two graphs side by side, with per-graph color ' +
    'histograms. Step or run to convergence and compare histograms: differing histograms ' +
    'prove the graphs are non-isomorphic; identical histograms on non-isomorphic graphs ' +
    'expose the limits of 1-WL. Click a node to highlight all same-colored nodes.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let half = canvasWidth / 2;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(19);
  text('WL Color Refinement', canvasWidth / 2, 6);
  textSize(15); fill(statusColor);
  text('Step ' + step + '  —  ' + statusText, canvasWidth / 2, 28);

  // graph panels
  fill('dimgray'); textSize(13); textAlign(CENTER, TOP);
  text('Graph A', half / 2, 50); text('Graph B', half + half / 2, 50);
  stroke('silver'); line(half, 48, half, drawHeight - 130);

  let gy = 70, gh = drawHeight - 130 - gy;
  hoverInfo = null;
  drawGraph(g1, 0, half, gy, gh, 0);
  drawGraph(g2, half, half, gy, gh, 1);

  // histograms
  drawHistogram(g1, 0, drawHeight - 122, half, 'A');
  drawHistogram(g2, half, drawHeight - 122, half, 'B');

  if (hoverInfo) drawTooltip();
}

function nodeXY(g, px, pw, gy, gh, i) { return { x: px + 34 + g.nodes[i].rx * (pw - 68), y: gy + 16 + g.nodes[i].ry * (gh - 32) }; }

function drawGraph(g, px, pw, gy, gh, gi) {
  stroke(170); strokeWeight(2);
  for (let e of g.edges) { let a = nodeXY(g, px, pw, gy, gh, e[0]), b = nodeXY(g, px, pw, gy, gh, e[1]); line(a.x, a.y, b.x, b.y); }
  for (let i = 0; i < g.nodes.length; i++) {
    let p = nodeXY(g, px, pw, gy, gh, i);
    let lbl = g.nodes[i].label, col = PALETTE[lbl % PALETTE.length];
    if (dist(mouseX, mouseY, p.x, p.y) <= NODE_R) {
      let nb = g.adj[i].map(j => g.nodes[j].label).sort((a, b) => a - b);
      hoverInfo = { x: p.x, y: p.y, lines: ['Node ' + i + ' (G' + (gi + 1) + ')', 'color = ' + lbl, 'neighbor colors: {' + nb.join(',') + '}'] };
    }
    let same = (selLabel !== null && lbl === selLabel);
    if (same) { stroke('black'); strokeWeight(4); } else { stroke(60); strokeWeight(1.5); }
    fill(col); circle(p.x, p.y, NODE_R * 2);
    noStroke(); fill(textColorFor(col)); textAlign(CENTER, CENTER); textSize(14); text(lbl, p.x, p.y);
  }
}

function drawHistogram(g, px, py, pw, tag) {
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  text('Color histogram ' + tag, px + 10, py);
  let counts = {};
  for (let nd of g.nodes) counts[nd.label] = (counts[nd.label] || 0) + 1;
  let labels = Object.keys(counts).map(Number).sort((a, b) => a - b);
  let maxC = Math.max(...Object.values(counts), 1);
  let bw = 22, gap = 8, x = px + 14, baseY = py + 96;
  for (let lbl of labels) {
    if (x + bw > px + pw - 8) break;
    let bh = (counts[lbl] / maxC) * 64;
    stroke(120); strokeWeight(1); fill(PALETTE[lbl % PALETTE.length]);
    rect(x, baseY - bh, bw, bh, 2);
    noStroke(); fill('black'); textAlign(CENTER, TOP); textSize(11);
    text(counts[lbl], x + bw / 2, baseY + 2);
    fill('dimgray'); text(lbl, x + bw / 2, baseY + 16);
    x += bw + gap;
  }
}

function drawTooltip() {
  let w = 210, h = hoverInfo.lines.length * 17 + 12;
  let tx = constrain(hoverInfo.x + 16, 4, canvasWidth - w - 4), ty = constrain(hoverInfo.y - h / 2, 46, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < hoverInfo.lines.length; k++) text(hoverInfo.lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- WL ----------

function sigOf(g, i) { let nb = g.adj[i].map(j => g.nodes[j].label).sort((a, b) => a - b); return g.nodes[i].label + '|' + nb.join(','); }

function doStep() {
  if (step >= MAX_STEPS || converged) return;
  let sigs = new Set();
  [g1, g2].forEach(g => g.nodes.forEach((_, i) => sigs.add(sigOf(g, i))));
  let sorted = [...sigs].sort(), map = new Map(); sorted.forEach((s, k) => map.set(s, k));
  [g1, g2].forEach(g => g.nodes.forEach((nd, i) => nd.nextLabel = map.get(sigOf(g, i))));
  [g1, g2].forEach(g => g.nodes.forEach(nd => nd.label = nd.nextLabel));
  let distinct = sorted.length;
  if (distinct === prevDistinct) converged = true;
  prevDistinct = distinct; step++;
  updateStatus();
}

function runToConvergence() {
  let guard = 0;
  while (!converged && step < MAX_STEPS && guard < 20) { doStep(); guard++; }
}

function hist(g) { let h = {}; for (let nd of g.nodes) h[nd.label] = (h[nd.label] || 0) + 1; return h; }
function sameHist(a, b) { let ks = new Set([...Object.keys(a), ...Object.keys(b)]); for (let k of ks) if ((a[k] || 0) !== (b[k] || 0)) return false; return true; }

function updateStatus() {
  let differ = !sameHist(hist(g1), hist(g2));
  if (differ) { statusText = 'DISTINGUISHABLE ✓ (histograms differ)'; statusColor = '#2e7d32'; }
  else if (converged) { statusText = 'INDISTINGUISHABLE — same histogram (1-WL fails)'; statusColor = '#c62828'; }
  else { statusText = 'same histogram so far'; statusColor = '#e65100'; }
}

// ---------- pairs ----------

function initPair() {
  let idx = pairSelect ? pairSelect.elt.selectedIndex : 0;
  let d = pairDefs(idx);
  g1 = buildGraph(d[0]); g2 = buildGraph(d[1]);
  step = 0; converged = false; selLabel = null;
  initLabels(g1); initLabels(g2);
  let labs = new Set(); [g1, g2].forEach(g => g.nodes.forEach(n => labs.add(n.label)));
  prevDistinct = labs.size;
  updateStatus();
}

function initLabels(g) { g.nodes.forEach((nd, i) => nd.label = g.adj[i].length); }

function buildGraph(def) {
  let nodes = def.nodes.map(p => ({ rx: p[0], ry: p[1], label: 0 }));
  let adj = nodes.map(() => []);
  for (let e of def.edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  return { nodes, adj, edges: def.edges };
}

function pairDefs(idx) {
  if (idx === 1) {
    let prism = { nodes: [[0.5, 0.08], [0.12, 0.7], [0.88, 0.7], [0.5, 0.32], [0.34, 0.56], [0.66, 0.56]],
      edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]] };
    let k33 = { nodes: [[0.25, 0.12], [0.25, 0.5], [0.25, 0.88], [0.75, 0.12], [0.75, 0.5], [0.75, 0.88]],
      edges: [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]] };
    return [prism, k33];
  }
  if (idx === 2) {
    let cyc = { nodes: ring(6), edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]] };
    let tris = { nodes: [[0.25, 0.2], [0.12, 0.7], [0.38, 0.7], [0.75, 0.2], [0.62, 0.7], [0.88, 0.7]],
      edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]] };
    return [cyc, tris];
  }
  // Pair 0: Path P4 vs Star K1,3
  let path = { nodes: [[0.12, 0.5], [0.38, 0.5], [0.62, 0.5], [0.88, 0.5]], edges: [[0, 1], [1, 2], [2, 3]] };
  let star = { nodes: [[0.5, 0.5], [0.5, 0.12], [0.15, 0.78], [0.85, 0.78]], edges: [[0, 1], [0, 2], [0, 3]] };
  return [path, star];
}

function ring(n) { let p = []; for (let i = 0; i < n; i++) { let a = TWO_PI * i / n - HALF_PI; p.push([0.5 + 0.38 * Math.cos(a), 0.5 + 0.38 * Math.sin(a)]); } return p; }

function textColorFor(hex) { let c = color(hex); let l = 0.299 * red(c) + 0.587 * green(c) + 0.114 * blue(c); return l > 150 ? 'black' : 'white'; }

// ---------- interaction ----------

function mousePressed() {
  if (mouseY < 46 || mouseY > drawHeight - 130) { selLabel = null; return; }
  let half = canvasWidth / 2, gy = 70, gh = drawHeight - 130 - gy;
  for (let gi = 0; gi < 2; gi++) {
    let g = gi === 0 ? g1 : g2, px = gi === 0 ? 0 : half;
    for (let i = 0; i < g.nodes.length; i++) {
      let p = nodeXY(g, px, half, gy, gh, i);
      if (dist(mouseX, mouseY, p.x, p.y) <= NODE_R) { selLabel = (selLabel === g.nodes[i].label) ? null : g.nodes[i].label; return; }
    }
  }
  selLabel = null;
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
