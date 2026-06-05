// Graph Transformer Attention Heatmap (local MPNN vs global self-attention)
// CANVAS_HEIGHT: 520
// Learning objective (Analyze): compare strictly-local MPNN attention with global
// multi-head self-attention on a molecular graph, and watch attention spread from
// local to long-range as GPS layers deepen. Synthetic attention (softmax over
// graph distance + noise) mimics a trained model without one.
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

let graphSelect, mpnnButton, globalButton, layerSlider;

let pos = [], adj = [], distM = [];
let N = 6, layer = 1, attnMode = 'mpnn';   // 'mpnn' | 'global'
let qmag = [];           // synthetic query magnitudes
let A = [];              // attention matrix A[i][j] = i attends to j
let hoverNode = -1;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  graphSelect = createSelect();
  graphSelect.position(10, drawHeight + 8);
  graphSelect.option('Benzene (6-ring)');
  graphSelect.option('Naphthalene (fused bicyclic, 10)');
  graphSelect.option('Random ER graph (14)');
  graphSelect.changed(loadGraph);

  mpnnButton = createButton('Local MPNN');
  mpnnButton.position(234, drawHeight + 8);
  mpnnButton.mousePressed(() => { attnMode = 'mpnn'; computeAttention(); });
  globalButton = createButton('Global MHA');
  globalButton.position(326, drawHeight + 8);
  globalButton.mousePressed(() => { attnMode = 'global'; computeAttention(); });

  layerSlider = createSlider(1, 5, 1, 1);
  layerSlider.position(sliderLeftMargin, drawHeight + 45);
  layerSlider.size(canvasWidth - sliderLeftMargin - margin);
  layerSlider.input(() => { layer = layerSlider.value(); computeAttention(); });

  loadGraph();
  describe('Graph transformer attention on a molecular graph. Toggle between strictly ' +
    'local MPNN attention (only along bonds) and global multi-head self-attention (curved ' +
    'arcs connect distant atoms). The layer slider shows attention spreading from local to ' +
    'long-range with depth. Hover an atom to see its top attended neighbors.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let gw = canvasWidth * 0.66;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(19);
  text('Graph Transformer Attention', gw / 2, 6);
  textSize(13); fill('#555'); textAlign(CENTER, TOP);
  text((attnMode === 'mpnn' ? 'Local MPNN' : 'Global MHA') + '  ·  layer ' + layer + ' / 5', gw / 2, 28);

  drawGraph(0, 48, gw, drawHeight - 56);
  drawPanel(gw, 48, canvasWidth - gw - margin);
  highlightButtons();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('GPS layer = ' + layer, 10, drawHeight + 52);
}

function nodeScreen(i, gx, gy, gw, gh) { return { x: gx + 30 + pos[i].x * (gw - 60), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawGraph(gx, gy, gw, gh) {
  hoverNode = -1;
  // received attention (column sums) for node brightness
  let recv = new Array(N).fill(0);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) recv[j] += A[i][j];
  let maxRecv = Math.max(...recv, 1e-9);

  // attention edges
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (i === j) continue;
    let w = Math.max(A[i][j], A[j][i]);
    if (w < 0.02) continue;
    let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh);
    let bonded = adj[i].includes(j);
    if (attnMode === 'mpnn' && !bonded) continue;
    let op = constrain(w * 3, 0.05, 1);
    if (bonded) {
      stroke(red(color('#d32f2f')), green(color('#d32f2f')), blue(color('#d32f2f')), op * 255);
      strokeWeight(1 + w * 12); noFill(); line(a.x, a.y, b.x, b.y);
    } else {
      // global non-adjacent: curved arc
      stroke(red(color('#6a1b9a')), green(color('#6a1b9a')), blue(color('#6a1b9a')), op * 200);
      strokeWeight(1 + w * 8); noFill();
      let mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 - 40;
      beginShape(); vertex(a.x, a.y); quadraticVertex(mx, my, b.x, b.y); endShape();
    }
  }
  // bonds (faint baseline)
  stroke(150); strokeWeight(1);
  for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) { let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh); line(a.x, a.y, b.x, b.y); }

  // nodes (brightness = received attention)
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    if (dist(mouseX, mouseY, p.x, p.y) <= 15) hoverNode = i;
    let frac = recv[i] / maxRecv;
    stroke(i === hoverNode ? 'black' : 60); strokeWeight(i === hoverNode ? 3 : 1.5);
    fill(lerpColor(color('#e8eaf6'), color('#283593'), frac));
    circle(p.x, p.y, 26);
    noStroke(); fill(frac > 0.5 ? 'white' : 'black'); textAlign(CENTER, CENTER); textSize(10); text(i, p.x, p.y);
  }
  noStroke(); fill('dimgray'); textAlign(LEFT, BOTTOM); textSize(11);
  text(attnMode === 'mpnn' ? 'red = attention along bonds (local)' : 'red = bond attention · purple arcs = long-range', gx + 6, gy + gh + 2);
}

function drawPanel(px, py, pw) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, py, pw, drawHeight - py - 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14);
  let tx = px + 10, ty = py + 8;
  if (hoverNode < 0) { text('Hover an atom to', tx, ty); ty += 18; text('see its attention.', tx, ty); ty += 26; }
  else {
    let i = hoverNode;
    text('Atom ' + i, tx, ty); ty += 20;
    textSize(12); text('query magnitude: ' + nf(qmag[i], 1, 2), tx, ty); ty += 20;
    text('Top attended:', tx, ty); ty += 16;
    let row = A[i].map((w, j) => [w, j]).filter(x => x[1] !== i).sort((a, b) => b[0] - a[0]).slice(0, 5);
    let valX = px + pw - 8;          // right edge for value labels (inside panel)
    let barX = tx + 52;              // bar start
    let barMaxW = (valX - 30) - barX; // leave room for the value label
    for (let [w, j] of row) {
      if (w < 0.005) continue;
      noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12); text('atom ' + j, tx, ty);
      stroke('silver'); strokeWeight(1); fill('#283593'); rect(barX, ty, Math.max(1, w * barMaxW), 10, 2);
      noStroke(); fill('dimgray'); textAlign(RIGHT, TOP); text(nf(w, 1, 2), valX, ty - 1);
      ty += 15;
    }
    textAlign(LEFT, TOP);
    ty += 8;
  }
  noStroke(); fill('dimgray'); textSize(11); textAlign(LEFT, TOP);
  text('node brightness =', tx, ty); ty += 14; text('attention received', tx, ty);
}

function highlightButtons() {
  mpnnButton.style('background-color', attnMode === 'mpnn' ? '#283593' : '#eeeeee');
  mpnnButton.style('color', attnMode === 'mpnn' ? 'white' : 'black');
  globalButton.style('background-color', attnMode === 'global' ? '#6a1b9a' : '#eeeeee');
  globalButton.style('color', attnMode === 'global' ? 'white' : 'black');
}

// ---------- synthetic attention ----------

function computeAttention() {
  // beta: sharp/local at layer 1, distributed at layer 5
  let beta = 3.2 / layer;
  A = [];
  let rng = mulberry32(1234 + layer * 7 + (attnMode === 'global' ? 100 : 0));
  for (let i = 0; i < N; i++) {
    let cands = attnMode === 'mpnn' ? [i, ...adj[i]] : [...Array(N).keys()];
    let scores = cands.map(j => {
      let d = (j === i) ? 0 : distM[i][j];
      return { j, s: -beta * d + (rng() - 0.5) * 0.8 + 0.3 * qmag[j] };
    });
    let mx = Math.max(...scores.map(x => x.s));
    let sum = 0; scores.forEach(x => { x.e = Math.exp(x.s - mx); sum += x.e; });
    let row = new Array(N).fill(0);
    scores.forEach(x => row[x.j] = x.e / sum);
    A.push(row);
  }
}

// ---------- graphs ----------

function loadGraph() {
  let name = graphSelect ? graphSelect.value() : 'Benzene (6-ring)';
  let edges = [];
  if (name.startsWith('Benzene')) {
    N = 6; pos = [];
    for (let i = 0; i < 6; i++) { let a = TWO_PI * i / 6 - HALF_PI; pos.push({ x: 0.5 + 0.32 * Math.cos(a), y: 0.5 + 0.36 * Math.sin(a) }); }
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]];
  } else if (name.startsWith('Naphthalene')) {
    N = 10;
    pos = [
      { x: 0.153, y: 0.585 }, { x: 0.30, y: 0.70 }, { x: 0.447, y: 0.585 }, { x: 0.447, y: 0.415 },
      { x: 0.30, y: 0.30 }, { x: 0.153, y: 0.415 }, { x: 0.741, y: 0.585 }, { x: 0.594, y: 0.70 },
      { x: 0.594, y: 0.30 }, { x: 0.741, y: 0.415 }
    ];
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [2, 7], [7, 6], [6, 9], [9, 8], [8, 3]];
  } else {
    N = 14; let rng = mulberry32(77); pos = [];
    for (let i = 0; i < N; i++) { let a = TWO_PI * i / N; pos.push({ x: 0.5 + 0.36 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a) }); }
    let set = new Set();
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (rng() < 0.22) { edges.push([i, j]); set.add(i + '-' + j); }
    // ensure connected-ish: chain
    for (let i = 0; i < N - 1; i++) if (!set.has(i + '-' + (i + 1))) edges.push([i, i + 1]);
  }
  adj = Array.from({ length: N }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  distM = allPairsBFS();
  let rng2 = mulberry32(42);
  qmag = Array.from({ length: N }, () => 0.3 + rng2() * 0.7);
  layer = layerSlider ? layerSlider.value() : 1;
  computeAttention();
}

function allPairsBFS() {
  let D = [];
  for (let s = 0; s < N; s++) {
    let d = new Array(N).fill(99); d[s] = 0; let q = [s];
    while (q.length) { let u = q.shift(); for (let v of adj[u]) if (d[v] === 99) { d[v] = d[u] + 1; q.push(v); } }
    D.push(d);
  }
  return D;
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// ---------- responsive ----------

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); layerSlider.size(canvasWidth - sliderLeftMargin - margin); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
