// Spectral vs Spatial Explorer for graph convolution
// CANVAS_HEIGHT: 530
// Learning objective (Analyze, Bloom L4): compare the spatial view (k-hop neighborhood
// aggregation) with the spectral view (Laplacian eigenvectors). Click eigenvectors to
// see them on the graph; low-pass filtering reproduces the smoothing of aggregation.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 430;
let controlHeight = 100;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 170;
let defaultTextSize = 16;

let propagateButton, lowpassCheckbox, kSlider, threshSlider;

const N = 12;
let adj = [], deg = [], pos = [];
let eig = [];            // [{val, vec}] sorted ascending
let mode = 'spatial';    // 'spatial' | 'spectral' | 'lowpass'
let K = 2, selEig = -1, thresh = 0.6, lowpass = false;
let seeds = [0, 11];
let sig = [];            // current node signal for coloring
let hoverNode = -1, hoverBar = -1;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildGraph();
  eig = jacobiEigen(normalizedLaplacian());
  thresh = eig[Math.floor(N / 3)].val;

  propagateButton = createButton('Propagate K hops');
  propagateButton.position(10, drawHeight + 8);
  propagateButton.mousePressed(propagate);

  lowpassCheckbox = createCheckbox(' Low-pass filter', false);
  lowpassCheckbox.position(150, drawHeight + 10);
  lowpassCheckbox.changed(() => { lowpass = lowpassCheckbox.checked(); if (lowpass) { mode = 'lowpass'; applyLowpass(); } else { mode = 'spatial'; propagate(); } });

  kSlider = createSlider(1, 3, 2, 1);
  kSlider.position(sliderLeftMargin, drawHeight + 42);
  kSlider.input(() => { K = kSlider.value(); if (mode === 'spatial') propagate(); });

  threshSlider = createSlider(0, 2, thresh, 0.05);
  threshSlider.position(sliderLeftMargin, drawHeight + 74);
  threshSlider.input(() => { thresh = threshSlider.value(); if (lowpass) applyLowpass(); });
  sizeSliders();

  propagate();
  describe('Spatial vs spectral views of graph convolution on a 12-node, two-cluster ' +
    'graph. Left: k-hop neighborhood aggregation from two seed nodes. Right: the Laplacian ' +
    'eigenvalue spectrum — click a bar to paint its eigenvector on the graph. Toggle the ' +
    'low-pass filter to keep only low-frequency eigenvectors and watch aggregation-style ' +
    'smoothing emerge.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let gw = canvasWidth * 0.52;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(15);
  text('Spatial: k-hop aggregation', gw / 2, 6);
  text('Spectral: Laplacian eigenvectors', gw + (canvasWidth - gw) / 2, 6);

  drawGraph(0, 30, gw, drawHeight - 38);
  drawSpectrum(gw + 8, 30, canvasWidth - gw - 8 - margin, drawHeight - 38);
  if (hoverNode >= 0) drawNodeTip();
  if (hoverBar >= 0) drawBarTip();

  drawControlLabels();
}

function nodeScreen(i, gx, gy, gw, gh) { return { x: gx + 28 + pos[i].x * (gw - 56), y: gy + 24 + pos[i].y * (gh - 48) }; }

function drawGraph(gx, gy, gw, gh) {
  hoverNode = -1;
  // receptive-field rings in spatial mode
  let hop = null;
  if (mode === 'spatial') { hop = new Array(N).fill(99); for (let s of seeds) { let d = bfs(s); for (let i = 0; i < N; i++) hop[i] = Math.min(hop[i], d[i]); } }

  stroke(180); strokeWeight(1.5);
  for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) {
    let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh); line(a.x, a.y, b.x, b.y);
  }
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    if (dist(mouseX, mouseY, p.x, p.y) <= 16) hoverNode = i;
    if (hop && hop[i] >= 1 && hop[i] <= K) {
      let rc = hop[i] === 1 ? color(33, 150, 243, 80) : hop[i] === 2 ? color(0, 150, 136, 70) : color(76, 175, 80, 60);
      noStroke(); fill(rc); circle(p.x, p.y, 40);
    }
    // color by current signal/eigenvector (diverging) or spatial (sequential)
    let col;
    if (mode === 'spectral') col = divergingColor(sig[i]);
    else col = lerpColor(color('#e3f2fd'), color('#1a237e'), constrain(sig[i], 0, 1));
    let isSeed = seeds.includes(i) && mode === 'spatial';
    if (isSeed) { stroke('#e65100'); strokeWeight(4); } else { stroke(60); strokeWeight(1.5); }
    fill(col); circle(p.x, p.y, 30);
    noStroke(); fill('black'); textAlign(CENTER, CENTER); textSize(10); text(i, p.x, p.y - 1);
  }
  // caption
  noStroke(); fill('dimgray'); textAlign(LEFT, BOTTOM); textSize(11);
  let cap = mode === 'spectral' ? ('eigenvector ' + selEig + '  (λ=' + nf(eig[selEig].val, 1, 2) + ')')
    : mode === 'lowpass' ? ('low-pass: keep λ ≤ ' + nf(thresh, 1, 2)) : ('mean aggregation, K=' + K + ' from 2 seeds');
  text(cap, gx + 6, gy + gh + 4);
}

function drawSpectrum(px, py, pw, ph) {
  hoverBar = -1;
  noStroke(); fill('dimgray'); textAlign(LEFT, TOP); textSize(11);
  text('click a bar to view its eigenvector →', px, py - 2);
  let n = eig.length, gap = 4;
  let bw = (pw - (n - 1) * gap - 8) / n;
  let baseY = py + ph - 28, maxH = ph - 60;
  let lmax = eig[n - 1].val || 2;
  for (let k = 0; k < n; k++) {
    let x = px + 4 + k * (bw + gap);
    let h = (eig[k].val / lmax) * maxH + 3;
    if (mouseX >= x && mouseX <= x + bw && mouseY >= py + 10 && mouseY <= baseY) hoverBar = k;
    // color low(blue)->high(red) by eigenvalue
    let frac = eig[k].val / 2;
    stroke(k === selEig ? 'black' : 'silver'); strokeWeight(k === selEig ? 2 : 1);
    fill(lerpColor(color('#1565c0'), color('#c62828'), frac));
    rect(x, baseY - h, bw, h);
    noStroke(); fill('dimgray'); textAlign(CENTER, TOP); textSize(9); text(k, x + bw / 2, baseY + 2);
    // low-pass threshold cutoff
  }
  // threshold marker
  if (lowpass) {
    let lmax2 = 2;
    // find x position between bars whose eigenvalue crosses thresh
    let cutX = px + 4;
    for (let k = 0; k < n; k++) { if (eig[k].val <= thresh) cutX = px + 4 + (k + 1) * (bw + gap); }
    stroke('#e65100'); strokeWeight(2); drawingContext.setLineDash([4, 3]);
    line(cutX, py + 10, cutX, baseY); drawingContext.setLineDash([]);
    noStroke(); fill('#e65100'); textSize(10);
    let lblTxt = 'λ ≤ ' + nf(thresh, 1, 2) + ' kept';
    let rightEdge = px + pw;
    // if the label would overflow the panel's right edge, draw it to the LEFT of the cut line
    if (cutX + 4 + textWidth(lblTxt) > rightEdge) { textAlign(RIGHT, TOP); text(lblTxt, cutX - 4, py + 10); }
    else { textAlign(LEFT, TOP); text(lblTxt, cutX + 4, py + 10); }
  }
  noStroke(); fill('dimgray'); textAlign(CENTER, TOP); textSize(11);
  text('eigenvalue index (low freq → high freq)', px + pw / 2, baseY + 14);
}

function drawNodeTip() {
  let i = hoverNode;
  let lines = ['Node ' + i, 'value: ' + nf(sig[i], 1, 3), 'degree: ' + deg[i]];
  tip(lines, mouseX, mouseY);
}
function drawBarTip() { tip(['Eigenvector ' + hoverBar, 'λ = ' + nf(eig[hoverBar].val, 1, 3)], mouseX, mouseY); }
function tip(lines, mx, my) {
  let w = 150, h = lines.length * 17 + 12;
  let tx = constrain(mx + 14, 4, canvasWidth - w - 4), ty = constrain(my, 30, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

function drawControlLabels() {
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('Depth K = ' + K, 10, drawHeight + 49);
  text('λ thresh = ' + nf(thresh, 1, 2), 10, drawHeight + 81);
}

// ---------- spatial & spectral ops ----------

function propagate() {
  mode = 'spatial'; selEig = -1;
  let s = new Array(N).fill(0); for (let v of seeds) s[v] = 1;
  for (let r = 0; r < K; r++) {
    let ns = new Array(N);
    for (let v = 0; v < N; v++) { let sum = s[v], c = 1; for (let u of adj[v]) { sum += s[u]; c++; } ns[v] = sum / c; }
    s = ns;
  }
  // normalize to [0,1] for coloring
  let mx = Math.max(...s), mn = Math.min(...s);
  sig = s.map(v => (v - mn) / (mx - mn + 1e-9));
}

function selectEig(k) {
  mode = 'spectral'; selEig = k; lowpass = false; if (lowpassCheckbox) lowpassCheckbox.checked(false);
  sig = eig[k].vec.slice();   // diverging color uses raw components
}

function applyLowpass() {
  mode = 'lowpass'; selEig = -1;
  let s0 = new Array(N).fill(0); for (let v of seeds) s0[v] = 1;
  // reconstruct using eigenvectors with eigenvalue <= thresh
  let recon = new Array(N).fill(0);
  for (let k = 0; k < N; k++) {
    if (eig[k].val > thresh) continue;
    let coeff = 0; for (let i = 0; i < N; i++) coeff += s0[i] * eig[k].vec[i];
    for (let i = 0; i < N; i++) recon[i] += coeff * eig[k].vec[i];
  }
  let mx = Math.max(...recon), mn = Math.min(...recon);
  sig = recon.map(v => (v - mn) / (mx - mn + 1e-9));
}

function divergingColor(v) {
  // v is an eigenvector component; scale by max abs over current sig
  let m = Math.max(...sig.map(Math.abs), 1e-9);
  let t = constrain(v / m, -1, 1);
  if (t >= 0) return lerpColor(color('#f7f7f7'), color('#b2182b'), t);
  return lerpColor(color('#f7f7f7'), color('#2166ac'), -t);
}

// ---------- graph + linear algebra ----------

function buildGraph() {
  // two clusters of 6, one bridge
  let edges = [[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,5],
               [6,7],[6,8],[7,8],[7,9],[8,10],[9,10],[9,11],[10,11],
               [5,6]];
  adj = Array.from({ length: N }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  deg = adj.map(a => a.length);
  // positions: two clusters
  pos = [];
  let A = [0,1,2,3,4,5], B = [6,7,8,9,10,11];
  A.forEach((id, k) => { let a = TWO_PI * k / 6 - HALF_PI; pos[id] = { x: 0.27 + 0.17 * Math.cos(a), y: 0.5 + 0.32 * Math.sin(a) }; });
  B.forEach((id, k) => { let a = TWO_PI * k / 6 - HALF_PI; pos[id] = { x: 0.73 + 0.17 * Math.cos(a), y: 0.5 + 0.32 * Math.sin(a) }; });
}

function normalizedLaplacian() {
  let L = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    L[i][i] = deg[i] > 0 ? 1 : 0;
    for (let j of adj[i]) L[i][j] = -1 / Math.sqrt(deg[i] * deg[j]);
  }
  return L;
}

function jacobiEigen(Ain) {
  let n = Ain.length;
  let A = Ain.map(r => r.slice());
  let V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-12) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-14) continue;
      let phi = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
      let c = Math.cos(phi), s = Math.sin(phi);
      for (let k = 0; k < n; k++) { let akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { let apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { let vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  let e = [];
  for (let i = 0; i < n; i++) e.push({ val: A[i][i], vec: V.map(r => r[i]) });
  e.sort((a, b) => a.val - b.val);
  return e;
}

function bfs(s) { let d = new Array(N).fill(-1); d[s] = 0; let q = [s]; while (q.length) { let u = q.shift(); for (let v of adj[u]) if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); } } return d.map(x => x < 0 ? 99 : x); }

// ---------- interaction ----------

function mousePressed() {
  let gw = canvasWidth * 0.52;
  // bar click in spectrum
  if (mouseX > gw && hoverBar >= 0) { selectEig(hoverBar); return; }
}

function sizeSliders() { let w = canvasWidth - sliderLeftMargin - margin; [kSlider, threshSlider].forEach(s => s && s.size(w)); }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); sizeSliders(); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
