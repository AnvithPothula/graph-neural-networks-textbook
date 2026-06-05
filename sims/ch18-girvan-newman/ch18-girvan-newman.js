// Girvan-Newman Community Detection (Karate Club)
// CANVAS_HEIGHT: 510
// Learning objective (Understand): edge betweenness identifies bridge edges;
// removing them one by one exposes community structure. Modularity Q is tracked
// and its maximum flagged.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 450;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let stepButton, autoButton, resetButton;

const N = 34;
let baseEdges = [], edges = [], pos = [];
let betw = {}, comp = [], compCount = 1, removed = [], iter = 0;
let Q = 0, bestQ = -1, bestIter = 0;
let autoOn = false, lastStep = 0, hoverEdge = null, hoverNode = -1;

const PALETTE = ['#1565c0', '#2e7d32', '#c62828', '#6a1b9a', '#ef6c00', '#00838f', '#ad1457', '#558b2f'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildKarate();

  stepButton = createButton('Step'); stepButton.position(10, drawHeight + 8); stepButton.mousePressed(() => { autoOn = false; autoButton.html('Auto Play'); step(); });
  autoButton = createButton('Auto Play'); autoButton.position(64, drawHeight + 8); autoButton.mousePressed(() => { autoOn = !autoOn; autoButton.html(autoOn ? 'Stop' : 'Auto Play'); lastStep = 0; });
  resetButton = createButton('Reset'); resetButton.position(150, drawHeight + 8); resetButton.mousePressed(reset);

  reset();
  describe('Girvan-Newman community detection on the Karate Club graph. Edges are colored ' +
    'by betweenness centrality (gray = low, red = high). Step removes the highest-betweenness ' +
    'edge, recomputes betweenness and components, and updates modularity Q. The partition at ' +
    'maximum Q is the detected community structure.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  if (autoOn && compCount < N && millis() - lastStep > 800) { step(); lastStep = millis(); }

  let gw = canvasWidth - 240;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18);
  text('Girvan-Newman Community Detection', gw / 2, 6);

  hoverEdge = null; hoverNode = -1;
  drawGraph(0, 34, gw, drawHeight - 42);
  drawPanel(gw, 34, 240);

  if (bestIter === iter && iter > 0) {
    noStroke(); fill(249, 168, 37, 230); rect(gw / 2 - 150, 32, 300, 22, 6);
    fill('black'); textAlign(CENTER, CENTER); textSize(13); text('Maximum modularity: Q = ' + nf(bestQ, 1, 3), gw / 2, 43);
  }
  if (hoverEdge) drawEdgeTip(); else if (hoverNode >= 0) drawNodeTip();
}

function nodeScreen(i, gx, gy, gw, gh) { return { x: gx + 22 + pos[i].x * (gw - 44), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawGraph(gx, gy, gw, gh) {
  let maxB = Math.max(...Object.values(betw), 1e-9);
  strokeWeight(2);
  for (let e of edges) {
    let a = nodeScreen(e[0], gx, gy, gw, gh), b = nodeScreen(e[1], gx, gy, gw, gh);
    let bv = betw[ekey(e[0], e[1])] || 0, frac = bv / maxB;
    if (distToSeg(mouseX, mouseY, a.x, a.y, b.x, b.y) < 5) hoverEdge = { e, bv, frac };
    stroke(lerpColor(color('#cccccc'), color('#d32f2f'), frac)); strokeWeight(1 + frac * 4);
    line(a.x, a.y, b.x, b.y);
  }
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    if (dist(mouseX, mouseY, p.x, p.y) <= 11) hoverNode = i;
    stroke(40); strokeWeight(1); fill(PALETTE[comp[i] % PALETTE.length]); circle(p.x, p.y, 18);
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(9); text(i, p.x, p.y);
  }
}

function drawPanel(px, py, pw) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, py, pw - margin, drawHeight - py - 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14);
  let tx = px + 12, ty = py + 10, lh = 22;
  text('Iteration: ' + iter, tx, ty); ty += lh;
  text('Edges removed: ' + removed.length, tx, ty); ty += lh;
  text('Components: ' + compCount, tx, ty); ty += lh;
  text('Modularity Q: ' + nf(Q, 1, 3), tx, ty); ty += lh;
  fill('#2e7d32'); text('Best Q: ' + nf(bestQ, 1, 3) + ' @ iter ' + bestIter, tx, ty); ty += lh + 6;
  fill('black'); textSize(13); text('Last removed edges:', tx, ty); ty += 18;
  textSize(12); fill('#555');
  for (let k = Math.max(0, removed.length - 6); k < removed.length; k++) { text('  (' + removed[k][0] + ' – ' + removed[k][1] + ')', tx, ty); ty += 15; }
}

function drawEdgeTip() {
  let ranks = Object.entries(betw).sort((a, b) => b[1] - a[1]).map(x => x[0]);
  let rank = ranks.indexOf(ekey(hoverEdge.e[0], hoverEdge.e[1])) + 1;
  tip(['Edge (' + hoverEdge.e[0] + ' – ' + hoverEdge.e[1] + ')', 'betweenness: ' + nf(hoverEdge.bv, 1, 1), 'rank: ' + rank + ' / ' + edges.length]);
}
function drawNodeTip() {
  let deg = edges.filter(e => e[0] === hoverNode || e[1] === hoverNode).length;
  tip(['Node ' + hoverNode, 'degree: ' + deg, 'community: ' + comp[hoverNode]]);
}
function tip(lines) {
  let w = 170, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 30, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- Girvan-Newman ----------

function ekey(a, b) { return Math.min(a, b) + '_' + Math.max(a, b); }

function step() {
  if (edges.length === 0) return;
  computeBetweenness();
  // remove max-betweenness edge
  let maxE = edges[0], maxV = -1;
  for (let e of edges) { let v = betw[ekey(e[0], e[1])] || 0; if (v > maxV) { maxV = v; maxE = e; } }
  edges = edges.filter(e => !(e[0] === maxE[0] && e[1] === maxE[1]));
  removed.push(maxE);
  iter++;
  computeComponents();
  computeModularity();
  computeBetweenness();
  if (Q > bestQ) { bestQ = Q; bestIter = iter; }
}

function adjacency(es) { let a = Array.from({ length: N }, () => []); for (let e of es) { a[e[0]].push(e[1]); a[e[1]].push(e[0]); } return a; }

function computeBetweenness() {
  betw = {}; for (let e of edges) betw[ekey(e[0], e[1])] = 0;
  let adj = adjacency(edges);
  for (let s = 0; s < N; s++) {
    // Brandes for edge betweenness
    let S = [], P = Array.from({ length: N }, () => []), sigma = new Array(N).fill(0), d = new Array(N).fill(-1);
    sigma[s] = 1; d[s] = 0; let q = [s];
    while (q.length) { let v = q.shift(); S.push(v); for (let w of adj[v]) { if (d[w] < 0) { d[w] = d[v] + 1; q.push(w); } if (d[w] === d[v] + 1) { sigma[w] += sigma[v]; P[w].push(v); } } }
    let delta = new Array(N).fill(0);
    while (S.length) { let w = S.pop(); for (let v of P[w]) { let c = (sigma[v] / sigma[w]) * (1 + delta[w]); betw[ekey(v, w)] += c; delta[v] += c; } }
  }
  for (let k in betw) betw[k] /= 2;
}

function computeComponents() {
  comp = new Array(N).fill(-1); let adj = adjacency(edges); compCount = 0;
  for (let s = 0; s < N; s++) { if (comp[s] >= 0) continue; let q = [s]; comp[s] = compCount; while (q.length) { let u = q.shift(); for (let v of adj[u]) if (comp[v] < 0) { comp[v] = compCount; q.push(v); } } compCount++; }
}

function computeModularity() {
  // Q on ORIGINAL graph with current partition
  let m = baseEdges.length; let deg = new Array(N).fill(0);
  for (let e of baseEdges) { deg[e[0]]++; deg[e[1]]++; }
  let Q2 = 0;
  for (let e of baseEdges) if (comp[e[0]] === comp[e[1]]) Q2 += 1;       // within-community edges
  Q2 = Q2 / m;
  let sumDeg = {};
  for (let i = 0; i < N; i++) sumDeg[comp[i]] = (sumDeg[comp[i]] || 0) + deg[i];
  let expect = 0; for (let c in sumDeg) expect += (sumDeg[c] / (2 * m)) ** 2;
  Q = Q2 - expect;
}

function reset() { edges = baseEdges.map(e => e.slice()); removed = []; iter = 0; Q = 0; bestQ = -1; bestIter = 0; autoOn = false; if (autoButton) autoButton.html('Auto Play'); computeComponents(); computeModularity(); computeBetweenness(); bestQ = Q; bestIter = 0; }

function distToSeg(px, py, x1, y1, x2, y2) { let dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy; let t = l2 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l2)) : 0; return dist(px, py, x1 + t * dx, y1 + t * dy); }

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

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
