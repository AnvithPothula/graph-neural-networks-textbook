// Motif Significance (Z-score) Profile
// CANVAS_HEIGHT: 520
// Learning objective (Analyze): a subgraph is a motif when it appears far more often
// than in degree-preserving random graphs (Z > 2), an anti-motif when far less (Z < -2).
// Significance differs from raw frequency.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let graphSelect, ensSlider, recomputeButton;

const N = 34;
let adj = [], pos = [], ensembleSize = 200;
const MOTIFS = ['wedge', 'triangle', 'path4', 'star4', 'cycle4', 'paw', 'diamond', 'K4'];
let realCounts = {}, zscores = {}, randMean = {}, randStd = {}, hoverBar = -1, computing = false;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  graphSelect = createSelect(); graphSelect.position(10, drawHeight + 8);
  ['Karate Club', 'Random (Erdős–Rényi)', 'Scale-free (Barabási–Albert)', 'Ring lattice'].forEach(o => graphSelect.option(o));
  graphSelect.changed(() => { loadGraph(); recompute(); });
  recomputeButton = createButton('Recompute ensemble'); recomputeButton.position(310, drawHeight + 8); recomputeButton.mousePressed(recompute);
  ensSlider = createSlider(100, 1000, 200, 100); ensSlider.position(sliderLeftMargin, drawHeight + 45); ensSlider.size(canvasWidth - sliderLeftMargin - margin); ensSlider.input(() => ensembleSize = ensSlider.value());

  loadGraph(); recompute();
  describe('Motif significance profile. The left panel shows the chosen host graph; the ' +
    'right panel shows Z-scores for eight connected subgraphs on 3 and 4 nodes, comparing the ' +
    'real counts against a degree-preserving random ensemble. Green bars (Z > 2) are motifs, ' +
    'red bars (Z < −2) are anti-motifs.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let gw = canvasWidth * 0.4;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(15);
  text('Host graph: ' + graphSelect.value(), gw / 2, 6);
  text('Motif significance (Z-score)', gw + (canvasWidth - gw) / 2, 6);
  stroke('silver'); line(gw, 26, gw, drawHeight - 8);

  drawGraph(0, 30, gw, drawHeight - 80);
  drawDegDist(0, drawHeight - 76, gw, 64);
  hoverBar = -1;
  drawBars(gw, 30, canvasWidth - gw, drawHeight - 58);
  if (hoverBar >= 0) drawTip();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13); text('Ensemble size = ' + ensembleSize, 10, drawHeight + 52);
}

function nodeXY(i, gx, gy, gw, gh) { return { x: gx + 22 + pos[i].x * (gw - 44), y: gy + 14 + pos[i].y * (gh - 28) }; }

function drawGraph(gx, gy, gw, gh) {
  stroke(195); strokeWeight(1);
  for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) { let a = nodeXY(i, gx, gy, gw, gh), b = nodeXY(j, gx, gy, gw, gh); line(a.x, a.y, b.x, b.y); }
  for (let i = 0; i < N; i++) { let p = nodeXY(i, gx, gy, gw, gh); stroke(40); strokeWeight(1); fill('#3949ab'); circle(p.x, p.y, 12); }
}

function drawDegDist(gx, gy, gw, gh) {
  let deg = adj.map(a => a.length), maxD = Math.max(...deg, 1);
  let hist = new Array(maxD + 1).fill(0); for (let d of deg) hist[d]++;
  noStroke(); fill('dimgray'); textAlign(LEFT, TOP); textSize(10); text('degree distribution', gx + 6, gy);
  let bw = (gw - 20) / (maxD + 1);
  for (let k = 0; k <= maxD; k++) { let h = (hist[k] / Math.max(...hist)) * (gh - 18); fill('#90caf9'); stroke('silver'); rect(gx + 8 + k * bw, gy + gh - 2 - h, bw - 1, h); }
}

function drawBars(gx, gy, gw, gh) {
  let bw = (gw - 40) / MOTIFS.length;
  let zmax = 6;
  let y0 = gy + gh / 2;   // zero line in middle
  stroke(180); strokeWeight(1); line(gx + 20, y0, gx + gw - 16, y0);
  // threshold lines
  stroke('#aaa'); drawingContext.setLineDash([3, 3]);
  let yp = map(2, -zmax, zmax, gy + gh - 16, gy + 16), yn = map(-2, -zmax, zmax, gy + gh - 16, gy + 16);
  line(gx + 20, yp, gx + gw - 16, yp); line(gx + 20, yn, gx + gw - 16, yn); drawingContext.setLineDash([]);
  noStroke(); fill('#888'); textSize(9); textAlign(LEFT, CENTER); text('Z=+2', gx + gw - 44, yp); text('Z=−2', gx + gw - 44, yn);
  for (let i = 0; i < MOTIFS.length; i++) {
    let z = zscores[MOTIFS[i]] || 0; z = constrain(z, -zmax, zmax);
    let x = gx + 24 + i * bw;
    let yv = map(z, -zmax, zmax, gy + gh - 16, gy + 16);
    if (mouseX > x && mouseX < x + bw - 6 && mouseY > Math.min(y0, yv) && mouseY < Math.max(y0, yv)) hoverBar = i;
    let col = z > 2 ? '#2e7d32' : z < -2 ? '#c62828' : '#9e9e9e';
    fill(col); stroke(60); strokeWeight(1); rect(x, Math.min(y0, yv), bw - 6, Math.abs(yv - y0));
    noStroke(); fill('black'); textAlign(CENTER, TOP); textSize(9); text(MOTIFS[i], x + (bw - 6) / 2, gy + gh - 12);
    drawMotifIcon(x + (bw - 6) / 2, gy + gh + 2, MOTIFS[i]);
  }
}

function drawMotifIcon(cx, cy, m) {
  push(); stroke(80); strokeWeight(1); fill('#3949ab'); let r = 2.2;
  let pts = motifPts(m).map(p => ({ x: cx + p[0] * 9, y: cy + p[1] * 9 }));
  let me = motifEdges(m);
  for (let e of me) line(pts[e[0]].x, pts[e[0]].y, pts[e[1]].x, pts[e[1]].y);
  for (let p of pts) circle(p.x, p.y, r * 2);
  pop();
}
function motifPts(m) {
  if (m === 'wedge') return [[-1, 0.6], [0, -0.6], [1, 0.6]];
  if (m === 'triangle') return [[-1, 0.6], [0, -0.7], [1, 0.6]];
  if (m === 'path4') return [[-1.2, 0], [-0.4, 0], [0.4, 0], [1.2, 0]];
  if (m === 'star4') return [[0, 0], [-1, -0.6], [1, -0.6], [0, 1]];
  if (m === 'cycle4') return [[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]];
  if (m === 'paw') return [[-1, -0.6], [0, -0.6], [-0.5, 0.7], [1.1, -0.6]];
  if (m === 'diamond') return [[-1, 0], [0, -0.9], [0, 0.9], [1, 0]];
  return [[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]];
}
function motifEdges(m) {
  if (m === 'wedge') return [[0, 1], [1, 2]];
  if (m === 'triangle') return [[0, 1], [1, 2], [2, 0]];
  if (m === 'path4') return [[0, 1], [1, 2], [2, 3]];
  if (m === 'star4') return [[0, 1], [0, 2], [0, 3]];
  if (m === 'cycle4') return [[0, 1], [1, 2], [2, 3], [3, 0]];
  if (m === 'paw') return [[0, 1], [1, 2], [2, 0], [1, 3]];
  if (m === 'diamond') return [[0, 1], [1, 3], [3, 2], [2, 0], [1, 2]];
  return [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]];
}

function drawTip() {
  let m = MOTIFS[hoverBar];
  let lines = [m, 'real count: ' + (realCounts[m] || 0), 'random mean: ' + nf(randMean[m] || 0, 1, 1),
    'random std: ' + nf(randStd[m] || 0, 1, 2), 'Z-score: ' + nf(zscores[m] || 0, 1, 2)];
  let w = 170, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 30, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- motif counting (ESU) ----------

function countMotifs(adjList) {
  let adjS = adjList.map(a => new Set(a));
  let c = {}; MOTIFS.forEach(m => c[m] = 0);
  // size 3 and 4 connected subgraphs via ESU
  for (let v = 0; v < N; v++) {
    let ext = adjList[v].filter(u => u > v);
    esu([v], ext.slice(), v, adjList, adjS, c);
  }
  return c;
}
function esu(sub, ext, v, adjList, adjS, c) {
  if (sub.length === 4) { classify(sub, adjS, c); return; }
  if (sub.length === 3) { classify(sub, adjS, c); }   // also count size-3
  while (ext.length) {
    let w = ext.pop();
    // exclusive neighbors of w (index>v, not in sub, not neighbor of existing sub)
    let nx = ext.slice();
    for (let u of adjList[w]) if (u > v && !sub.includes(u) && !ext.includes(u)) { let excl = true; for (let s of sub) if (adjS[s].has(u)) { excl = false; break; } if (excl) nx.push(u); }
    esu(sub.concat([w]), nx, v, adjList, adjS, c);
  }
}
function classify(sub, adjS, c) {
  let k = sub.length, m = 0, deg = new Array(k).fill(0);
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) if (adjS[sub[i]].has(sub[j])) { m++; deg[i]++; deg[j]++; }
  let ds = deg.slice().sort((a, b) => a - b).join('');
  if (k === 3) { if (m === 2) c.wedge++; else if (m === 3) c.triangle++; return; }
  if (m === 3) { if (ds === '1113') c.star4++; else c.path4++; }
  else if (m === 4) { if (ds === '2222') c.cycle4++; else c.paw++; }
  else if (m === 5) c.diamond++;
  else if (m === 6) c.K4++;
}

function recompute() {
  realCounts = countMotifs(adj);
  // ensemble of degree-preserving rewirings
  let acc = {}; MOTIFS.forEach(m => acc[m] = []);
  let edges0 = []; for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) edges0.push([i, j]);
  let rng = mulberry32(13);
  for (let s = 0; s < ensembleSize; s++) {
    let es = edges0.map(e => e.slice());
    rewire(es, rng);
    let ra = Array.from({ length: N }, () => []); for (let e of es) { ra[e[0]].push(e[1]); ra[e[1]].push(e[0]); }
    let cc = countMotifs(ra);
    MOTIFS.forEach(m => acc[m].push(cc[m]));
  }
  randMean = {}; randStd = {}; zscores = {};
  for (let m of MOTIFS) {
    let arr = acc[m], mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    let varr = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
    let std = Math.sqrt(varr);
    randMean[m] = mean; randStd[m] = std;
    zscores[m] = std > 1e-6 ? (realCounts[m] - mean) / std : 0;
  }
}

function rewire(es, rng) {
  let set = new Set(es.map(e => e[0] + '_' + e[1]));
  let swaps = es.length * 3;
  for (let s = 0; s < swaps; s++) {
    let i = Math.floor(rng() * es.length), j = Math.floor(rng() * es.length);
    if (i === j) continue;
    let [a, b] = es[i], [cc, d] = es[j];
    if (a === cc || a === d || b === cc || b === d) continue;
    let n1 = [Math.min(a, d), Math.max(a, d)], n2 = [Math.min(cc, b), Math.max(cc, b)];
    let k1 = n1[0] + '_' + n1[1], k2 = n2[0] + '_' + n2[1];
    if (set.has(k1) || set.has(k2)) continue;
    set.delete(es[i][0] + '_' + es[i][1]); set.delete(es[j][0] + '_' + es[j][1]);
    es[i] = n1; es[j] = n2; set.add(k1); set.add(k2);
  }
}

// ---------- graphs ----------

function loadGraph() {
  let name = graphSelect ? graphSelect.value() : 'Karate Club';
  let edges = [];
  if (name.startsWith('Karate')) {
    edges = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
      [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
      [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],[14,33],
      [15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],[23,33],
      [24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  } else if (name.startsWith('Random')) {
    let rng = mulberry32(5), p = 0.14, set = new Set();
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (rng() < p) { edges.push([i, j]); set.add(i + '_' + j); }
    for (let i = 0; i < N - 1; i++) if (!set.has(i + '_' + (i + 1))) edges.push([i, i + 1]);
  } else if (name.startsWith('Scale')) {
    let rng = mulberry32(9), rep = [0, 1, 2]; edges.push([0, 1], [1, 2], [0, 2]);
    for (let v = 3; v < N; v++) { let chosen = new Set(); while (chosen.size < 2) chosen.add(rep[Math.floor(rng() * rep.length)]); for (let t of chosen) { edges.push([Math.min(v, t), Math.max(v, t)]); rep.push(t); rep.push(v); } }
  } else {
    for (let i = 0; i < N; i++) { edges.push([i, (i + 1) % N]); edges.push([Math.min(i, (i + 2) % N), Math.max(i, (i + 2) % N)]); }
  }
  adj = Array.from({ length: N }, () => []);
  let seen = new Set();
  for (let e of edges) { let k = Math.min(e[0], e[1]) + '_' + Math.max(e[0], e[1]); if (seen.has(k) || e[0] === e[1]) continue; seen.add(k); adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  pos = springLayout(edges);
}

function springLayout(es) {
  let p = []; for (let i = 0; i < N; i++) { let a = TWO_PI * i / N; p.push({ x: 0.5 + 0.4 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a) }); }
  for (let it = 0; it < 200; it++) {
    let fx = new Array(N).fill(0), fy = new Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), r = 0.0008 / d2; fx[i] += dx / d * r; fy[i] += dy / d * r; fx[j] -= dx / d * r; fy[j] -= dy / d * r; }
    for (let e of es) { let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, s = (d - 0.12) * 0.02; fx[e[0]] += dx / d * s; fy[e[0]] += dy / d * s; fx[e[1]] -= dx / d * s; fy[e[1]] -= dy / d * s; }
    for (let i = 0; i < N; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); ensSlider.size(canvasWidth - sliderLeftMargin - margin); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
