// WL Color Refinement Simulator - side-by-side Weisfeiler-Lehman on two graphs
// CANVAS_HEIGHT: 520
// Learning objective (Analyze, Bloom L4): observe WL refinement on graph pairs
// WL can and cannot distinguish, building intuition for GNN expressiveness limits.
// MicroSim template version 2026.03

// ----- responsive canvas globals -----
let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;     // two graph panels
let controlHeight = 80;   // two control rows (buttons + legend)
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

// ----- controls -----
let stepButton, autoButton, resetButton, pairSelect;

// ----- model -----
let g1, g2;               // each: {nodes:[{rx,ry,label,prevLabel}], adj:[[...]]}
let step = 0;
let prevDistinct = 0;
let converged = false;
let statusText = '';
let statusColor = '#444';
const MAX_STEPS = 6;

// animation
let animStart = -10000;
const ANIM_MS = 700;

// auto mode
let autoOn = false;
let lastAuto = 0;

// selection / infobox
let selG = -1, selNode = -1;

// 12-color palette (distinct, consistent across both graphs)
const PALETTE = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                 '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
                 '#9A6324', '#000075'];

const NODE_R = 20;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  stepButton = createButton('Step');
  stepButton.position(10, drawHeight + 8);
  stepButton.mousePressed(() => { autoOn = false; autoButton.html('Auto'); doStep(); });

  autoButton = createButton('Auto');
  autoButton.position(64, drawHeight + 8);
  autoButton.mousePressed(toggleAuto);

  resetButton = createButton('Reset');
  resetButton.position(128, drawHeight + 8);
  resetButton.mousePressed(() => { autoOn = false; autoButton.html('Auto'); initPair(); });

  pairSelect = createSelect();
  pairSelect.position(196, drawHeight + 8);
  pairSelect.option('Triangle vs. Path (distinguishable)');
  pairSelect.option('3-regular: Prism vs. K₃,₃ (WL fails)');
  pairSelect.option('Karate sample vs. ER (n=10)');
  pairSelect.option('Star K₁₋₅ vs. Path P₆');
  pairSelect.changed(() => { autoOn = false; autoButton.html('Auto'); initPair(); });

  initPair();
  describe('Side-by-side Weisfeiler-Lehman color refinement on two graphs. Step ' +
    'through iterations and watch node colors refine. Some graph pairs become ' +
    'distinguishable; 3-regular pairs never do, illustrating the limits of ' +
    'neighborhood-aggregation algorithms. Click a node to see how its next label ' +
    'is computed from its neighbors.', LABEL);
}

function draw() {
  updateCanvasSize();

  // backgrounds
  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  // auto stepping
  if (autoOn && !isAnimating() && !converged && step < MAX_STEPS) {
    if (millis() - lastAuto > 750) { doStep(); lastAuto = millis(); }
  }
  if (autoOn && (converged || step >= MAX_STEPS)) { autoOn = false; autoButton.html('Auto'); }

  // main title
  fill('black'); noStroke();
  textAlign(CENTER, TOP); textSize(20);
  text('WL Color Refinement Test', canvasWidth / 2, 6);

  // status line
  textSize(15); fill(statusColor);
  text('Step ' + step + '  —  ' + statusText, canvasWidth / 2, 30);

  // panel geometry
  let half = canvasWidth / 2;
  let panelTop = 52;
  // panel titles
  fill('dimgray'); textSize(14); textAlign(CENTER, TOP);
  text('Graph 1', half / 2, panelTop);
  text('Graph 2', half + half / 2, panelTop);

  // divider
  stroke('silver'); strokeWeight(1);
  line(half, 50, half, drawHeight - 8);

  let gy = panelTop + 22;
  let gh = drawHeight - gy - 12;
  drawGraph(g1, 0, half, gy, gh, 0);
  drawGraph(g2, half, half, gy, gh, 1);

  // legend (control row 2)
  drawLegend();

  // infobox
  if (selG >= 0) drawInfobox();
}

// ---------- graph rendering ----------

function nodeXY(g, panelX, panelW, gy, gh, i) {
  return { x: panelX + 30 + g.nodes[i].rx * (panelW - 60),
           y: gy + g.nodes[i].ry * gh };
}

function drawGraph(g, panelX, panelW, gy, gh, gi) {
  // edges
  stroke(170); strokeWeight(2);
  for (let e of g.edges) {
    let a = nodeXY(g, panelX, panelW, gy, gh, e[0]);
    let b = nodeXY(g, panelX, panelW, gy, gh, e[1]);
    line(a.x, a.y, b.x, b.y);
  }

  let anim = isAnimating();
  let t = anim ? (millis() - animStart) / ANIM_MS : 1;
  let phase1 = anim && t < 0.5;   // gather neighborhoods (yellow borders, old colors)

  for (let i = 0; i < g.nodes.length; i++) {
    let p = nodeXY(g, panelX, panelW, gy, gh, i);
    let lbl = phase1 ? g.nodes[i].prevLabel : g.nodes[i].label;
    let col = PALETTE[lbl % PALETTE.length];

    // border
    let isSel = (gi === selG && i === selNode);
    let isNbrOfSel = (gi === selG && selNode >= 0 && g.adj[selNode] && g.adj[selNode].includes(i));
    if (phase1) { stroke('gold'); strokeWeight(4); }
    else if (isSel) { stroke('black'); strokeWeight(4); }
    else if (isNbrOfSel) { stroke('gold'); strokeWeight(3); }
    else { stroke(60); strokeWeight(1.5); }

    fill(col);
    circle(p.x, p.y, NODE_R * 2);

    noStroke();
    fill(textColorFor(col));
    textAlign(CENTER, CENTER); textSize(15);
    text(lbl, p.x, p.y);
  }
}

function drawLegend() {
  // distinct labels currently in use, across both graphs
  let labels = new Set();
  [g1, g2].forEach(g => g.nodes.forEach(n => labels.add(n.label)));
  let sorted = [...labels].sort((a, b) => a - b);

  noStroke(); fill('black');
  textAlign(LEFT, CENTER); textSize(13);
  text('Labels:', 10, drawHeight + 58);

  let x = 70, y = drawHeight + 50, sw = 18;
  for (let lbl of sorted) {
    if (x + sw > canvasWidth - margin) break;
    stroke(120); strokeWeight(1);
    fill(PALETTE[lbl % PALETTE.length]);
    rect(x, y, sw, sw, 3);
    noStroke(); fill('black'); textAlign(CENTER, CENTER); textSize(11);
    text(lbl, x + sw / 2, y + sw / 2);
    x += sw + 16;
    // re-label position for value under swatch is omitted to save space
    textAlign(LEFT, CENTER);
  }
}

function drawInfobox() {
  let g = selG === 0 ? g1 : g2;
  if (selNode < 0 || selNode >= g.nodes.length) { selG = -1; return; }
  let half = canvasWidth / 2;
  let panelX = selG === 0 ? 0 : half;
  let gy = 74, gh = drawHeight - gy - 12;
  let p = nodeXY(g, panelX, half, gy, gh, selNode);

  let nbLabels = g.adj[selNode].map(j => g.nodes[j].label).sort((a, b) => a - b);
  let sig = g.nodes[selNode].label + ' : {' + nbLabels.join(', ') + '}';
  let lines = [
    'Graph ' + (selG + 1) + ' · Node ' + selNode,
    'Current label: ' + g.nodes[selNode].label,
    'Neighbor labels: {' + nbLabels.join(', ') + '}',
    'Next signature: ' + sig
  ];
  let w = 250, h = lines.length * 19 + 14;
  let bx = constrain(p.x + 24, 4, canvasWidth - w - 4);
  let by = constrain(p.y - h / 2, 54, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(80); strokeWeight(1);
  rect(bx, by, w, h, 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(13);
  for (let k = 0; k < lines.length; k++) text(lines[k], bx + 10, by + 8 + k * 19);
}

// ---------- WL algorithm ----------

function sigOf(g, i) {
  let nb = g.adj[i].map(j => g.nodes[j].label).sort((a, b) => a - b);
  return g.nodes[i].label + '|' + nb.join(',');
}

function doStep() {
  if (step >= MAX_STEPS || isAnimating()) return;

  // collect signatures over BOTH graphs (disjoint-union WL) → shared label ids
  let sigs = new Set();
  [g1, g2].forEach(g => g.nodes.forEach((_, i) => sigs.add(sigOf(g, i))));
  let sorted = [...sigs].sort();
  let map = new Map();
  sorted.forEach((s, k) => map.set(s, k));

  // save prev, compute and apply new labels
  [g1, g2].forEach(g => g.nodes.forEach((nd, i) => {
    nd.prevLabel = nd.label;
    nd.nextLabel = map.get(sigOf(g, i));
  }));
  [g1, g2].forEach(g => g.nodes.forEach(nd => nd.label = nd.nextLabel));

  let distinct = sorted.length;
  if (distinct === prevDistinct) converged = true;
  prevDistinct = distinct;
  step++;
  updateStatus();
  animStart = millis();
}

function histogram(g) {
  let h = {};
  for (let nd of g.nodes) h[nd.label] = (h[nd.label] || 0) + 1;
  return h;
}

function sameHistogram(a, b) {
  let keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (let k of keys) if ((a[k] || 0) !== (b[k] || 0)) return false;
  return true;
}

function updateStatus() {
  let differ = !sameHistogram(histogram(g1), histogram(g2));
  if (differ) { statusText = 'Distinguishable ✓'; statusColor = '#2e7d32'; }
  else if (converged) { statusText = 'Converged — same histogram (WL cannot distinguish)'; statusColor = '#1565c0'; }
  else { statusText = 'Same histogram so far'; statusColor = '#e65100'; }
}

// ---------- graph pairs ----------

function initPair() {
  let idx = pairSelect ? pairSelect.elt.selectedIndex : 0;
  let defs = pairDefs(idx);
  g1 = buildGraph(defs[0]);
  g2 = buildGraph(defs[1]);
  step = 0;
  converged = false;
  selG = -1; selNode = -1;
  animStart = -10000;
  // step-0 labels = degree
  initLabels(g1); initLabels(g2);
  let labs = new Set();
  [g1, g2].forEach(g => g.nodes.forEach(n => labs.add(n.label)));
  prevDistinct = labs.size;
  updateStatus();
}

function initLabels(g) {
  g.nodes.forEach((nd, i) => { nd.label = g.adj[i].length; nd.prevLabel = nd.label; });
}

function buildGraph(def) {
  let nodes = def.nodes.map(p => ({ rx: p[0], ry: p[1], label: 0, prevLabel: 0 }));
  let adj = nodes.map(() => []);
  for (let e of def.edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  return { nodes, adj, edges: def.edges };
}

function pairDefs(idx) {
  if (idx === 1) {
    // Prism (two triangles + matching) vs K3,3 — both 3-regular, non-isomorphic
    let prism = {
      nodes: [[0.5, 0.08], [0.12, 0.72], [0.88, 0.72],
              [0.5, 0.34], [0.34, 0.58], [0.66, 0.58]],
      edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]]
    };
    let k33 = {
      nodes: [[0.25, 0.15], [0.25, 0.5], [0.25, 0.85],
              [0.75, 0.15], [0.75, 0.5], [0.75, 0.85]],
      edges: [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
    };
    return [prism, k33];
  }
  if (idx === 2) {
    // Karate sample (10 nodes, 14 edges) vs a fixed ER-style graph (10 nodes, 14 edges)
    let karate = { nodes: ring(10), edges: [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
      [1, 3], [2, 3], [4, 5], [5, 6], [5, 7], [6, 7], [0, 5], [8, 9], [2, 8]] };
    let er = { nodes: ring(10), edges: [[0, 1], [0, 5], [1, 6], [2, 3], [2, 7], [3, 8],
      [4, 9], [4, 0], [5, 6], [6, 7], [7, 8], [8, 9], [9, 2], [1, 4]] };
    return [karate, er];
  }
  if (idx === 3) {
    // Star K1-5 vs Path P6
    let star = { nodes: [[0.5, 0.5]].concat(ringAround(5, 0.5, 0.5, 0.4)),
      edges: [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]] };
    let path = { nodes: [[0.1, 0.5], [0.26, 0.5], [0.42, 0.5], [0.58, 0.5], [0.74, 0.5], [0.9, 0.5]],
      edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]] };
    return [star, path];
  }
  // default idx 0: Triangle vs Path3
  let tri = { nodes: [[0.5, 0.2], [0.22, 0.72], [0.78, 0.72]], edges: [[0, 1], [1, 2], [2, 0]] };
  let path3 = { nodes: [[0.18, 0.5], [0.5, 0.5], [0.82, 0.5]], edges: [[0, 1], [1, 2]] };
  return [tri, path3];
}

function ring(n) {
  let pts = [];
  for (let i = 0; i < n; i++) {
    let a = TWO_PI * i / n - HALF_PI;
    pts.push([0.5 + 0.4 * Math.cos(a), 0.5 + 0.4 * Math.sin(a)]);
  }
  return pts;
}
function ringAround(n, cx, cy, r) {
  let pts = [];
  for (let i = 0; i < n; i++) {
    let a = TWO_PI * i / n - HALF_PI;
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return pts;
}

// ---------- helpers ----------

function isAnimating() { return millis() - animStart < ANIM_MS; }

function toggleAuto() {
  autoOn = !autoOn;
  autoButton.html(autoOn ? 'Stop' : 'Auto');
  lastAuto = 0;
}

function textColorFor(hex) {
  let c = color(hex);
  let lum = 0.299 * red(c) + 0.587 * green(c) + 0.114 * blue(c);
  return lum > 150 ? 'black' : 'white';
}

// ---------- interaction ----------

function mousePressed() {
  if (mouseY < 50 || mouseY > drawHeight) return;
  let half = canvasWidth / 2;
  let gy = 74, gh = drawHeight - gy - 12;
  for (let gi = 0; gi < 2; gi++) {
    let g = gi === 0 ? g1 : g2;
    let panelX = gi === 0 ? 0 : half;
    for (let i = 0; i < g.nodes.length; i++) {
      let p = nodeXY(g, panelX, half, gy, gh, i);
      if (dist(mouseX, mouseY, p.x, p.y) <= NODE_R) {
        if (selG === gi && selNode === i) { selG = -1; selNode = -1; }
        else { selG = gi; selNode = i; }
        return;
      }
    }
  }
  selG = -1; selNode = -1;   // clicked empty → close infobox
}

// ---------- responsive ----------

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
