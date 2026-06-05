// SPMiner Order Embedding Space
// CANVAS_HEIGHT: 520
// Learning objective (Understand): order embeddings encode the subgraph partial order
// geometrically — a neighborhood contains the query iff its embedding dominates the
// query coordinate-wise. Frequency estimation = counting dominated points.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 460;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let epsSlider, coneToggle;

const N = 34;
let emb = [], nbSize = [];        // 2D order embedding per Karate node neighborhood
let queries = [], activeQ = 0, eps = 0.03, showCone = true, hoverPt = -1;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildEmbeddings();

  epsSlider = createSlider(0.001, 0.1, 0.03, 0.001); epsSlider.position(sliderLeftMargin, drawHeight + 18); epsSlider.size(240); epsSlider.input(() => eps = epsSlider.value());
  coneToggle = createButton('Toggle dominance cone'); coneToggle.position(430, drawHeight + 14); coneToggle.mousePressed(() => showCone = !showCone);

  describe('SPMiner order-embedding space for the Karate Club graph. Each point is a node ' +
    'neighborhood embedded in 2D so that containment becomes coordinate-wise dominance. Pick a ' +
    'query subgraph (left): neighborhoods whose embedding dominates the query (inside its cone) ' +
    'are estimated matches (green). The match count estimates subgraph frequency.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18); text('Order Embedding Space', canvasWidth / 2, 6);

  let lw = 190, rw = 200, cw = canvasWidth - lw - rw;
  drawQueryList(0, 30, lw, drawHeight - 38);
  hoverPt = -1;
  let matches = drawScatter(lw, 30, cw, drawHeight - 38);
  drawStats(lw + cw, 30, rw, matches);

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13); text('ε = ' + nf(eps, 1, 3), 10, drawHeight + 25);
  if (hoverPt >= 0) drawTip();
}

function drawQueryList(gx, gy, gw, gh) {
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(13); text('Query subgraph:', gx + 8, gy);
  let y = gy + 22;
  for (let i = 0; i < queries.length; i++) {
    let sel = i === activeQ;
    fill(sel ? '#ffe082' : 'white'); stroke('#ccc'); strokeWeight(1); rect(gx + 8, y, gw - 16, 32, 4);
    if (mouseX > gx + 8 && mouseX < gx + gw - 8 && mouseY > y && mouseY < y + 32 && mouseIsPressedEdge) activeQ = i;
    // name on top line, true-freq on its own line below so long names never collide
    noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12); text(queries[i].name, gx + 16, y + 5);
    fill('#888'); textSize(10); textAlign(LEFT, TOP); text('true freq: ' + queries[i].trueFreq, gx + 16, y + 19);
    y += 38;
  }
  noStroke(); fill('dimgray'); textSize(10); textAlign(LEFT, TOP); text('click to select', gx + 8, y + 2);
}

function scr(e, gx, gy, gw, gh) { return { x: gx + 36 + e[0] * (gw - 56), y: gy + gh - 28 - e[1] * (gh - 50) }; }

function drawScatter(gx, gy, gw, gh, ) {
  // axes
  stroke(210); strokeWeight(1); line(gx + 32, gy + gh - 26, gx + gw - 16, gy + gh - 26); line(gx + 32, gy + 12, gx + 32, gy + gh - 26);
  noStroke(); fill('dimgray'); textSize(10); textAlign(CENTER, TOP); text('embedding dim 1', gx + gw / 2, gy + gh - 14);
  push(); translate(gx + 14, gy + gh / 2); rotate(-HALF_PI); textAlign(CENTER, CENTER); text('embedding dim 2', 0, 0); pop();

  let q = queries[activeQ].emb;
  let qp = scr(q, gx, gy, gw, gh);
  // dominance cone (toward +,+)
  if (showCone) { noStroke(); fill(46, 125, 50, 28); rect(qp.x, gy + 12, (gx + gw - 16) - qp.x, qp.y - (gy + 12)); }

  let matches = 0;
  for (let i = 0; i < N; i++) {
    let p = scr(emb[i], gx, gy, gw, gh);
    let dominates = emb[i][0] >= q[0] - eps && emb[i][1] >= q[1] - eps;
    if (dominates) matches++;
    if (dist(mouseX, mouseY, p.x, p.y) <= 7) hoverPt = i;
    stroke(40); strokeWeight(1);
    fill(dominates ? '#2e7d32' : lerpColor(color('#90caf9'), color('#ff9800'), nbSize[i] / Math.max(...nbSize)));
    circle(p.x, p.y, dominates ? 11 : 9);
  }
  // query star
  noStroke(); fill('#f9a825'); stroke('#b8860b'); strokeWeight(1.5); star(qp.x, qp.y, 7, 14, 5);
  return matches;
}

function star(x, y, r1, r2, n) { beginShape(); for (let i = 0; i < n * 2; i++) { let a = PI * i / n - HALF_PI; let r = i % 2 ? r1 : r2; vertex(x + r * cos(a), y + r * sin(a)); } endShape(CLOSE); }

function drawStats(gx, gy, gw, matches) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(gx + 8, gy, gw - 8 - margin, drawHeight - gy - 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(13);
  let tx = gx + 18, ty = gy + 12, lh = 22;
  let q = queries[activeQ];
  text(q.name, tx, ty); ty += lh + 4;
  let estFreq = matches;
  fill('#2e7d32'); text('Estimated matches: ' + estFreq, tx, ty); ty += lh;
  fill('black'); text('True (VF2): ' + q.trueFreq, tx, ty); ty += lh;
  let prec = q.trueFreq > 0 ? Math.min(1, q.trueFreq / Math.max(estFreq, 1)) : 0;
  text('Precision proxy: ' + nf(estFreq > 0 ? Math.min(estFreq, q.trueFreq) / estFreq : 0, 1, 2), tx, ty); ty += lh + 8;
  fill('#555'); textSize(11);
  text('Green points dominate the', tx, ty); ty += 14; text('query (contain it). Raising ε', tx, ty); ty += 14; text('relaxes the match boundary.', tx, ty);
}

function drawTip() {
  let i = hoverPt, q = queries[activeQ].emb;
  let viol = Math.max(0, q[0] - emb[i][0]) + Math.max(0, q[1] - emb[i][1]);
  let lines = ['Node ' + i + ' neighborhood', 'size: ' + nbSize[i], 'emb (' + nf(emb[i][0], 1, 2) + ', ' + nf(emb[i][1], 1, 2) + ')',
    'order violation: ' + nf(viol, 1, 3), viol <= eps ? 'MATCH ✓' : 'no match'];
  let w = 190, h = lines.length * 16 + 10;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 30, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 5 + k * 16);
}

// ---------- embeddings ----------

function buildEmbeddings() {
  let adj = buildKarate();
  // 2D order embedding: dim1 ~ degree (size), dim2 ~ local triangle density.
  // Larger/denser neighborhoods dominate smaller ones (order property).
  nbSize = []; emb = [];
  let degs = adj.map(a => a.length);
  let maxDeg = Math.max(...degs);
  for (let v = 0; v < N; v++) {
    let nb = adj[v];
    nbSize.push(nb.length + 1);
    // triangles around v
    let tri = 0; for (let i = 0; i < nb.length; i++) for (let j = i + 1; j < nb.length; j++) if (adj[nb[i]].includes(nb[j])) tri++;
    let maxTri = nb.length * (nb.length - 1) / 2 || 1;
    emb.push([degs[v] / maxDeg, 0.15 + 0.85 * tri / Math.max(maxTri, 1)]);
  }
  // queries: increasing in the order. emb is a LOWER bound a neighborhood must dominate.
  queries = [
    { name: 'single edge', emb: [0.10, 0.10], trueFreq: 34 },
    { name: 'path P3 (wedge)', emb: [0.25, 0.10], trueFreq: 28 },
    { name: 'triangle', emb: [0.25, 0.45], trueFreq: 14 },
    { name: '4-cycle', emb: [0.45, 0.30], trueFreq: 9 },
    { name: 'diamond', emb: [0.55, 0.55], trueFreq: 5 },
    { name: '4-clique', emb: [0.70, 0.80], trueFreq: 2 },
  ];
}

function buildKarate() {
  let e = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],[14,33],
    [15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],[23,33],
    [24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  let adj = Array.from({ length: N }, () => []);
  for (let ed of e) { adj[ed[0]].push(ed[1]); adj[ed[1]].push(ed[0]); }
  return adj;
}

let mouseIsPressedEdge = false;
function mousePressed() { mouseIsPressedEdge = true; }
function mouseReleased() { mouseIsPressedEdge = false; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
