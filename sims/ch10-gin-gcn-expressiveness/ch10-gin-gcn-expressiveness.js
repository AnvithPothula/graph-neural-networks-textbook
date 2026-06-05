// GIN vs GCN Expressiveness Demonstrator
// CANVAS_HEIGHT: 540
// Learning objective: show why GIN (sum aggregation) is strictly more expressive
// than GCN (mean aggregation). With uniform node features, mean collapses every
// node to the same embedding; sum preserves degree/structure — yet both still fail
// on 3-regular pairs, exposing the shared 1-WL ceiling.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 460;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let gcnButton, ginButton, pairSelect, kSlider;

let model = 'gin', K = 3;
let g1, g2, emb1, emb2, distinguishable = false;
let selNode = null;   // {g, vec}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  gcnButton = createButton('GCN (mean)');
  gcnButton.position(10, drawHeight + 8);
  gcnButton.mousePressed(() => { model = 'gcn'; recompute(); });
  ginButton = createButton('GIN (sum)');
  ginButton.position(112, drawHeight + 8);
  ginButton.mousePressed(() => { model = 'gin'; recompute(); });

  pairSelect = createSelect();
  pairSelect.position(200, drawHeight + 8);
  pairSelect.option('Regular vs. irregular (6-cycle vs. path)');
  pairSelect.option('Triangle vs. no-triangle (C₃ vs. P₃)');
  pairSelect.option('K₃,₃ vs. prism (both 3-regular)');
  pairSelect.changed(recompute);

  kSlider = createSlider(0, 3, 3, 1);
  kSlider.position(230, drawHeight + 45);
  kSlider.size(canvasWidth - 230 - margin);
  kSlider.input(() => K = kSlider.value());

  recompute();
  describe('Expressiveness of GIN (sum) vs GCN (mean) aggregation. Node features start ' +
    'uniform. Mean aggregation keeps every node identical, so GCN cannot tell the two ' +
    'graphs apart; sum aggregation encodes degree and structure, so GIN separates them — ' +
    'except on 3-regular pairs, where both hit the 1-WL ceiling. The scatter shows each ' +
    "node's layer-1 vs layer-2 aggregated value.", LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let half = canvasWidth / 2;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(19);
  text('GIN vs GCN Expressiveness', canvasWidth / 2, 6);

  // banner
  textSize(15); textAlign(CENTER, TOP);
  fill(distinguishable ? '#2e7d32' : '#c62828');
  let agg = model === 'gin' ? 'GIN (sum)' : 'GCN (mean)';
  text(agg + ': ' + (distinguishable ? 'DISTINGUISHABLE ✓' : 'INDISTINGUISHABLE ✗'), canvasWidth / 2, 28);

  // graph panels
  fill('dimgray'); textSize(13);
  text('Graph A', half / 2, 50); text('Graph B', half + half / 2, 50);
  stroke('silver'); line(half, 48, half, 250);
  drawGraph(g1, emb1, 0, half, 68, 178, 0);
  drawGraph(g2, emb2, half, half, 68, 178, 1);

  // scatter
  drawScatter(0, 256, canvasWidth, drawHeight - 256 - 6);

  highlightModel();
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('Aggregation layers K = ' + K, 10, drawHeight + 52);
}

function nodeXY(g, px, pw, gy, gh, i) { return { x: px + 30 + g.nodes[i].rx * (pw - 60), y: gy + 14 + g.nodes[i].ry * (gh - 28) }; }

function drawGraph(g, emb, px, pw, gy, gh, gi) {
  // color range from current-layer values across both graphs
  let allK = emb1.map(e => e[K]).concat(emb2.map(e => e[K]));
  let lo = Math.min(...allK), hi = Math.max(...allK);
  stroke(170); strokeWeight(2);
  for (let e of g.edges) { let a = nodeXY(g, px, pw, gy, gh, e[0]), b = nodeXY(g, px, pw, gy, gh, e[1]); line(a.x, a.y, b.x, b.y); }
  for (let i = 0; i < g.nodes.length; i++) {
    let p = nodeXY(g, px, pw, gy, gh, i);
    let frac = hi > lo ? (emb[i][K] - lo) / (hi - lo) : 0;
    let isSel = selNode && sameVec(selNode.vec, emb[i]);
    if (isSel) { stroke('gold'); strokeWeight(4); } else { stroke(60); strokeWeight(1.5); }
    fill(lerpColor(color('#bbdefb'), color('#1a237e'), frac));
    circle(p.x, p.y, 30);
    noStroke(); fill(frac > 0.5 ? 'white' : 'black'); textAlign(CENTER, CENTER); textSize(11);
    text(round(emb[i][K]), p.x, p.y);
  }
}

function drawScatter(px, py, pw, ph) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px + 4, py, pw - 8 - margin, ph);
  let x0 = px + 44, x1 = px + pw - 12 - margin, y0 = py + ph - 24, y1 = py + 16;
  // axis ranges over layer-1 and layer-2 values of both graphs
  let xs = emb1.map(e => e[1]).concat(emb2.map(e => e[1]));
  let ys = emb1.map(e => e[2]).concat(emb2.map(e => e[2]));
  let xlo = Math.min(...xs), xhi = Math.max(...xs), ylo = Math.min(...ys), yhi = Math.max(...ys);
  let xpad = (xhi - xlo) * 0.15 + 0.5, ypad = (yhi - ylo) * 0.15 + 0.5;
  let xFor = v => map(v, xlo - xpad, xhi + xpad, x0, x1);
  let yFor = v => map(v, ylo - ypad, yhi + ypad, y0, y1);
  // axes
  stroke(200); strokeWeight(1); line(x0, y0, x1, y0); line(x0, y0, x0, y1);
  noStroke(); fill('dimgray'); textSize(11); textAlign(CENTER, TOP);
  text('node embedding scatter — x: layer-1 value,  y: layer-2 value', (x0 + x1) / 2, py + 2);
  textAlign(CENTER, TOP); text('layer-1 aggregated value', (x0 + x1) / 2, y0 + 6);
  push(); translate(px + 14, (y0 + y1) / 2); rotate(-HALF_PI); textAlign(CENTER, CENTER); text('layer-2 value', 0, 0); pop();
  // points (jitter identical points slightly so overlaps are visible as a cluster)
  drawPts(emb1, '#4682b4', xFor, yFor, 0);
  drawPts(emb2, '#dc143c', xFor, yFor, 1);
  // legend
  noStroke(); textAlign(LEFT, CENTER); textSize(12);
  fill('#4682b4'); text('● Graph A', x1 - 150, y1 + 6); fill('#dc143c'); text('● Graph B', x1 - 70, y1 + 6);
}

function drawPts(emb, col, xFor, yFor, gi) {
  for (let i = 0; i < emb.length; i++) {
    let jx = (gi === 0 ? -2 : 2) + (i % 3 - 1) * 2;
    let x = xFor(emb[i][1]) + jx, y = yFor(emb[i][2]) + jx;
    let isSel = selNode && sameVec(selNode.vec, emb[i]);
    stroke(isSel ? 'gold' : 'white'); strokeWeight(isSel ? 3 : 1);
    fill(col); circle(x, y, 13);
  }
}

function highlightModel() {
  gcnButton.style('background-color', model === 'gcn' ? '#6366f1' : '#eeeeee');
  gcnButton.style('color', model === 'gcn' ? 'white' : 'black');
  ginButton.style('background-color', model === 'gin' ? '#f59e0b' : '#eeeeee');
  ginButton.style('color', model === 'gin' ? 'white' : 'black');
}

// ---------- aggregation ----------

function embed(g) {
  let cur = g.nodes.map(() => 1);          // uniform initial feature
  let H = g.nodes.map(() => [1]);
  for (let k = 1; k <= 3; k++) {
    let nx = g.nodes.map((_, v) => {
      if (model === 'gcn') { let s = cur[v], c = 1; for (let u of g.adj[v]) { s += cur[u]; c++; } return s / c; }
      let s = cur[v]; for (let u of g.adj[v]) s += cur[u]; return s;   // GIN: sum incl self
    });
    cur = nx; H.forEach((h, v) => h.push(round4(cur[v])));
  }
  return H;
}

function recompute() {
  let idx = pairSelect ? pairSelect.elt.selectedIndex : 0;
  let d = pairDefs(idx);
  g1 = buildGraph(d[0]); g2 = buildGraph(d[1]);
  emb1 = embed(g1); emb2 = embed(g2);
  distinguishable = !sameMultisetProper(emb1, emb2);
  selNode = null;
}

function round4(x) { return Math.round(x * 10000) / 10000; }
function sameVec(a, b) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-6) return false; return true; }
// multiset comparison of node-embedding vectors
function sameMultisetProper(A, B) {
  if (A.length !== B.length) return false;
  let key = v => v.map(x => x.toFixed(3)).join(',');
  let a = A.map(key).sort(), b = B.map(key).sort();
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ---------- pairs ----------

function buildGraph(def) {
  let nodes = def.nodes.map(p => ({ rx: p[0], ry: p[1] }));
  let adj = nodes.map(() => []);
  for (let e of def.edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  return { nodes, adj, edges: def.edges };
}

function pairDefs(idx) {
  if (idx === 1) {
    let c3 = { nodes: [[0.5, 0.2], [0.22, 0.75], [0.78, 0.75]], edges: [[0, 1], [1, 2], [2, 0]] };
    let p3 = { nodes: [[0.18, 0.5], [0.5, 0.5], [0.82, 0.5]], edges: [[0, 1], [1, 2]] };
    return [c3, p3];
  }
  if (idx === 2) {
    let prism = { nodes: [[0.5, 0.1], [0.14, 0.72], [0.86, 0.72], [0.5, 0.34], [0.35, 0.58], [0.65, 0.58]],
      edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]] };
    let k33 = { nodes: [[0.25, 0.14], [0.25, 0.5], [0.25, 0.86], [0.75, 0.14], [0.75, 0.5], [0.75, 0.86]],
      edges: [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]] };
    return [prism, k33];
  }
  // idx 0: 6-cycle vs path P6
  let cyc = { nodes: ring(6), edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]] };
  let path = { nodes: [[0.08, 0.5], [0.26, 0.5], [0.44, 0.5], [0.62, 0.5], [0.8, 0.5], [0.95, 0.5]],
    edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]] };
  return [cyc, path];
}

function ring(n) { let p = []; for (let i = 0; i < n; i++) { let a = TWO_PI * i / n - HALF_PI; p.push([0.5 + 0.36 * Math.cos(a), 0.5 + 0.36 * Math.sin(a)]); } return p; }

// ---------- interaction ----------

function mousePressed() {
  let half = canvasWidth / 2;
  if (mouseY < 60 || mouseY > 250) { selNode = null; return; }
  for (let gi = 0; gi < 2; gi++) {
    let g = gi === 0 ? g1 : g2, emb = gi === 0 ? emb1 : emb2, px = gi === 0 ? 0 : half;
    for (let i = 0; i < g.nodes.length; i++) {
      let p = nodeXY(g, px, half, 68, 178, i);
      if (dist(mouseX, mouseY, p.x, p.y) <= 16) { selNode = { g: gi, vec: emb[i] }; return; }
    }
  }
  selNode = null;
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); kSlider.size(canvasWidth - 230 - margin); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
