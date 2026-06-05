// Louvain Community Detection — two-phase modularity optimization (Karate Club)
// CANVAS_HEIGHT: 500
// Learning objective (Analyze): trace how local node moves (Phase 1) aggregate into
// global communities, and how compression into super-nodes (Phase 2) enables scaling.
// Modularity Q is tracked across passes.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 360;
let defaultTextSize = 16;

let p1Button, p2Button, nextButton, resetButton, gammaSlider;

const N = 34;
let baseEdges = [], pos = [];
let origComm = [];          // community color per original node
let cur = null;             // current (super)graph
let node2orig = [];         // node2orig[superNode] = [orig nodes]
let qHist = [], gamma = 1.0, pass = 0;
let hoverNode = -1;

const PALETTE = ['#1565c0', '#2e7d32', '#c62828', '#6a1b9a', '#ef6c00', '#00838f', '#ad1457', '#558b2f', '#4e342e', '#283593'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildKarate();

  p1Button = createButton('Run Phase 1'); p1Button.position(10, drawHeight + 8); p1Button.mousePressed(() => { phase1(); syncOrigComm(); recordQ(); });
  p2Button = createButton('Run Phase 2'); p2Button.position(112, drawHeight + 8); p2Button.mousePressed(() => { phase2(); });
  nextButton = createButton('Next Pass'); nextButton.position(214, drawHeight + 8); nextButton.mousePressed(() => { phase1(); syncOrigComm(); recordQ(); phase2(); });
  resetButton = createButton('Reset'); resetButton.position(300, drawHeight + 8); resetButton.mousePressed(reset);
  gammaSlider = createSlider(0.5, 2.0, 1.0, 0.1); gammaSlider.position(sliderLeftMargin, drawHeight + 14); gammaSlider.size(canvasWidth - sliderLeftMargin - margin); gammaSlider.input(() => { gamma = gammaSlider.value(); reset(); });

  reset();
  describe('Louvain community detection. Left: the Karate Club graph colored by current ' +
    'community. Center: the compressed super-graph after Phase 2, with super-nodes sized by ' +
    'how many original nodes they contain. Right: modularity Q across passes. Run Phase 1 ' +
    '(local moves), Phase 2 (compression), or Next Pass for both.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let w3 = canvasWidth / 3;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(14);
  text('Original graph', w3 / 2, 6); text('Super-graph (compressed)', w3 + w3 / 2, 6); text('Modularity Q per pass', 2 * w3 + w3 / 2, 6);
  stroke('silver'); line(w3, 26, w3, drawHeight - 8); line(2 * w3, 26, 2 * w3, drawHeight - 8);

  hoverNode = -1;
  drawOriginal(0, 30, w3, drawHeight - 38);
  drawSuper(w3, 30, w3, drawHeight - 38);
  drawQCurve(2 * w3, 30, w3, drawHeight - 38);

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13);
  text('Resolution γ = ' + nf(gamma, 1, 1), 360, drawHeight + 40);
  text('Pass: ' + pass + '   communities: ' + (new Set(origComm)).size, 10, drawHeight + 40);
  if (hoverNode >= 0) drawTip();
}

function nodeXY(i, gx, gy, gw, gh, p) { return { x: gx + 18 + p[i].x * (gw - 36), y: gy + 14 + p[i].y * (gh - 28) }; }

function drawOriginal(gx, gy, gw, gh) {
  stroke(210); strokeWeight(1);
  for (let e of baseEdges) { let a = nodeXY(e[0], gx, gy, gw, gh, pos), b = nodeXY(e[1], gx, gy, gw, gh, pos); line(a.x, a.y, b.x, b.y); }
  for (let i = 0; i < N; i++) {
    let p = nodeXY(i, gx, gy, gw, gh, pos);
    if (dist(mouseX, mouseY, p.x, p.y) <= 9) hoverNode = i;
    stroke(40); strokeWeight(1); fill(PALETTE[origComm[i] % PALETTE.length]); circle(p.x, p.y, 15);
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(8); text(i, p.x, p.y);
  }
}

function drawSuper(gx, gy, gw, gh) {
  let n = cur.size;
  // layout super-nodes on a circle
  let sp = [];
  for (let i = 0; i < n; i++) { let a = TWO_PI * i / n - HALF_PI; sp.push({ x: 0.5 + 0.36 * Math.cos(a), y: 0.5 + 0.36 * Math.sin(a) }); }
  // edges
  for (let i = 0; i < n; i++) cur.adj[i].forEach((w, j) => { if (i < j) { let a = nodeXY(i, gx, gy, gw, gh, sp), b = nodeXY(j, gx, gy, gw, gh, sp); stroke(180); strokeWeight(constrain(w, 1, 8)); line(a.x, a.y, b.x, b.y); } });
  for (let i = 0; i < n; i++) {
    let p = nodeXY(i, gx, gy, gw, gh, sp);
    let sz = node2orig[i].length;
    stroke(40); strokeWeight(1.5); fill(PALETTE[cur.comm[i] % PALETTE.length]);
    circle(p.x, p.y, map(sz, 1, N, 12, 40));
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(10); text(sz, p.x, p.y);
  }
  noStroke(); fill('dimgray'); textAlign(CENTER, BOTTOM); textSize(11); text(n + ' super-nodes', gx + gw / 2, gy + gh + 2);
}

function drawQCurve(gx, gy, gw, gh) {
  let x0 = gx + 30, x1 = gx + gw - 10, y0 = gy + gh - 20, y1 = gy + 10;
  stroke(200); strokeWeight(1); line(x0, y0, x1, y0); line(x0, y0, x0, y1);
  // Q=0.3 threshold
  let yT = map(0.3, 0, 1, y0, y1); stroke('#c62828'); drawingContext.setLineDash([4, 4]); line(x0, yT, x1, yT); drawingContext.setLineDash([]);
  noStroke(); fill('#c62828'); textSize(10); textAlign(LEFT, CENTER); text('0.3', x0 - 24, yT);
  fill('dimgray'); textAlign(LEFT, CENTER); text('1.0', x0 - 24, y1); text('0', x0 - 14, y0);
  textAlign(CENTER, TOP); text('pass', (x0 + x1) / 2, y0 + 4);
  if (qHist.length) {
    stroke('#1565c0'); strokeWeight(2); noFill(); beginShape();
    let nn = Math.max(qHist.length - 1, 1);
    for (let k = 0; k < qHist.length; k++) vertex(map(k, 0, nn, x0, x1), map(qHist[k], 0, 1, y0, y1));
    endShape();
    for (let k = 0; k < qHist.length; k++) { noStroke(); fill('#1565c0'); circle(map(k, 0, nn, x0, x1), map(qHist[k], 0, 1, y0, y1), 5); }
    noStroke(); fill('black'); textAlign(RIGHT, TOP); textSize(12); text('Q = ' + nf(qHist[qHist.length - 1], 1, 3), x1, y1);
  }
}

function drawTip() {
  let lines = ['Node ' + hoverNode, 'community: ' + origComm[hoverNode]];
  let w = 130, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- Louvain ----------

function reset() {
  pass = 0; gamma = gammaSlider ? gammaSlider.value() : 1.0;
  // current graph = original
  cur = { size: N, adj: Array.from({ length: N }, () => new Map()), self: new Array(N).fill(0), comm: Array.from({ length: N }, (_, i) => i), m: baseEdges.length };
  for (let e of baseEdges) { addW(cur.adj, e[0], e[1], 1); addW(cur.adj, e[1], e[0], 1); }
  node2orig = Array.from({ length: N }, (_, i) => [i]);
  origComm = Array.from({ length: N }, (_, i) => i);
  qHist = [modularityOrig()];
}

function addW(adj, i, j, w) { adj[i].set(j, (adj[i].get(j) || 0) + w); }

function wdeg(i) { let d = cur.self[i] * 2; cur.adj[i].forEach(w => d += w); return d; }

function phase1() {
  let m2 = 2 * cur.m;
  let comm = cur.comm;
  let degC = {}; for (let i = 0; i < cur.size; i++) degC[comm[i]] = (degC[comm[i]] || 0) + wdeg(i);
  let improved = true, guard = 0;
  while (improved && guard < 50) {
    improved = false; guard++;
    for (let i = 0; i < cur.size; i++) {
      let di = wdeg(i), ci = comm[i];
      // weight from i to each neighbor community
      let kin = {}; cur.adj[i].forEach((w, j) => { kin[comm[j]] = (kin[comm[j]] || 0) + w; });
      // remove i from ci
      degC[ci] -= di; comm[i] = -1;
      let kinCi = kin[ci] || 0;
      let bestC = ci, bestGain = (kin[ci] || 0) - gamma * degC[ci] * di / m2;
      for (let c in kin) { let cc = +c; let gain = kin[cc] - gamma * (degC[cc] || 0) * di / m2; if (gain > bestGain) { bestGain = gain; bestC = cc; } }
      comm[i] = bestC; degC[bestC] = (degC[bestC] || 0) + di;
      if (bestC !== ci) improved = true;
    }
  }
}

function phase2() {
  // relabel communities to 0..k-1
  let map = {}, k = 0;
  for (let i = 0; i < cur.size; i++) { if (!(cur.comm[i] in map)) map[cur.comm[i]] = k++; }
  let newAdj = Array.from({ length: k }, () => new Map()), newSelf = new Array(k).fill(0), newN2o = Array.from({ length: k }, () => []);
  for (let i = 0; i < cur.size; i++) newN2o[map[cur.comm[i]]].push(...node2orig[i]);
  for (let i = 0; i < cur.size; i++) {
    let ci = map[cur.comm[i]];
    newSelf[ci] += cur.self[i];
    cur.adj[i].forEach((w, j) => { let cj = map[cur.comm[j]]; if (ci === cj) { if (i <= j) newSelf[ci] += w * (i === j ? 1 : 1) / 1; } });
  }
  // recompute cross edges and within properly
  newSelf = new Array(k).fill(0);
  for (let i = 0; i < cur.size; i++) {
    let ci = map[cur.comm[i]];
    newSelf[ci] += cur.self[i];
    cur.adj[i].forEach((w, j) => { let cj = map[cur.comm[j]]; if (i < j) { if (ci === cj) newSelf[ci] += w; else { newAdj[ci].set(cj, (newAdj[ci].get(cj) || 0) + w); newAdj[cj].set(ci, (newAdj[cj].get(ci) || 0) + w); } } });
  }
  cur = { size: k, adj: newAdj, self: newSelf, comm: Array.from({ length: k }, (_, i) => i), m: cur.m };
  node2orig = newN2o;
  pass++;
  syncOrigComm();
}

function syncOrigComm() {
  // assign each original node the community of its current super-node
  for (let s = 0; s < cur.size; s++) for (let o of node2orig[s]) origComm[o] = cur.comm[s];
}

function recordQ() { qHist.push(modularityOrig()); }

function modularityOrig() {
  let m = baseEdges.length, deg = new Array(N).fill(0);
  for (let e of baseEdges) { deg[e[0]]++; deg[e[1]]++; }
  let within = 0; for (let e of baseEdges) if (origComm[e[0]] === origComm[e[1]]) within++;
  let sumDeg = {}; for (let i = 0; i < N; i++) sumDeg[origComm[i]] = (sumDeg[origComm[i]] || 0) + deg[i];
  let expect = 0; for (let c in sumDeg) expect += gamma * (sumDeg[c] / (2 * m)) ** 2;
  return within / m - expect;
}

function buildKarate() {
  baseEdges = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],[14,33],
    [15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],[23,33],
    [24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  pos = springLayout(baseEdges);
}
function springLayout(es) {
  let p = []; for (let i = 0; i < N; i++) { let a = TWO_PI * i / N; p.push({ x: 0.5 + 0.4 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a) }); }
  for (let it = 0; it < 250; it++) {
    let fx = new Array(N).fill(0), fy = new Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), r = 0.0008 / d2; fx[i] += dx / d * r; fy[i] += dy / d * r; fx[j] -= dx / d * r; fy[j] -= dy / d * r; }
    for (let e of es) { let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, s = (d - 0.12) * 0.02; fx[e[0]] += dx / d * s; fy[e[0]] += dy / d * s; fx[e[1]] -= dx / d * s; fy[e[1]] -= dy / d * s; }
    for (let i = 0; i < N; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); gammaSlider.size(canvasWidth - sliderLeftMargin - margin); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
