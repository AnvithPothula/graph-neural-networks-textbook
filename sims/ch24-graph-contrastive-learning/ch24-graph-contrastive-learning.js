// Graph Contrastive Learning Pipeline
// CANVAS_HEIGHT: 520
// Learning objective (Understand): two augmented views of a graph produce two embeddings
// per node; the NT-Xent contrastive loss pulls matching views together and pushes others
// apart. Stronger augmentation moves the views apart and raises the loss.
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

let edgeSlider, maskSlider, tempSlider, resampleButton;
const NN = 10;
let pos = [], edges = [], cls = [], base = [], v1 = [], v2 = [], dropped = [], masked = [];
let edgeDrop = 0.15, featMask = 0.2, temp = 0.5, selNode = -1, hoverP = -1;
let rng = mulberry32(7);
const BG = 'aliceblue';

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildGraph(); resample();

  edgeSlider = createSlider(0, 40, 15, 1); edgeSlider.position(sliderLeftMargin, drawHeight + 8); edgeSlider.size(180); edgeSlider.input(() => { edgeDrop = edgeSlider.value() / 100; resample(); });
  maskSlider = createSlider(0, 40, 20, 1); maskSlider.position(sliderLeftMargin, drawHeight + 32); maskSlider.size(180); maskSlider.input(() => { featMask = maskSlider.value() / 100; resample(); });
  tempSlider = createSlider(0.1, 1.0, 0.5, 0.05); tempSlider.position(sliderLeftMargin + 430, drawHeight + 8); tempSlider.size(150); tempSlider.input(() => temp = tempSlider.value());
  resampleButton = createButton('Resample Views'); resampleButton.position(sliderLeftMargin + 430, drawHeight + 40); resampleButton.mousePressed(resample);

  describe('Graph contrastive learning. The original graph (left) is augmented two ways: ' +
    'View 1 drops edges (red), View 2 masks node features (gray). A shared encoder maps each ' +
    'view of each node into embedding space (right), where matching pairs are linked by dashed ' +
    'lines. The NT-Xent loss falls when matching views stay close and others stay far; raising ' +
    'augmentation strength pushes the pairs apart and raises the loss.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill(BG); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let w3 = canvasWidth / 3;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(13);
  text('Original graph', w3 / 2, 6); text('Two augmented views', w3 + w3 / 2, 6); text('Embedding space', 2 * w3 + w3 / 2, 6);
  stroke('#bbb'); line(w3, 24, w3, drawHeight - 8); line(2 * w3, 24, 2 * w3, drawHeight - 8);

  hoverP = -1;
  drawOriginal(0, 28, w3, drawHeight - 36);
  drawViews(w3, 28, w3, drawHeight - 36);
  drawEmbed(2 * w3, 28, w3, drawHeight - 36);

  // control labels
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(12);
  text('Edge dropout ' + Math.round(edgeDrop * 100) + '%', 10, drawHeight + 16);
  text('Feature mask ' + Math.round(featMask * 100) + '%', 10, drawHeight + 40);
  text('Temperature τ ' + nf(temp, 1, 2), sliderLeftMargin + 250, drawHeight + 16);
  if (hoverP >= 0) drawTip();
}

function nodeXY(i, gx, gy, gw, gh) { return { x: gx + 22 + pos[i].x * (gw - 44), y: gy + 14 + pos[i].y * (gh - 28) }; }

function drawOriginal(gx, gy, gw, gh) {
  stroke(60, 60, 60, 120); strokeWeight(1.2);
  for (let e of edges) { let a = nodeXY(e[0], gx, gy, gw, gh), b = nodeXY(e[1], gx, gy, gw, gh); line(a.x, a.y, b.x, b.y); }
  for (let i = 0; i < NN; i++) { let p = nodeXY(i, gx, gy, gw, gh); if (dist(mouseX, mouseY, p.x, p.y) <= 11) hoverP = i; stroke(i === selNode ? 'black' : '#888'); strokeWeight(i === selNode ? 3 : 1); fill(cls[i] ? '#ff9800' : '#4fc3f7'); circle(p.x, p.y, 16); noStroke(); fill('#0d1117'); textAlign(CENTER, CENTER); textSize(8); text(i + 1, p.x, p.y); }
}

function drawViews(gx, gy, gw, gh) {
  let hh = (gh - 20) / 2;
  // View 1 (top): edge dropout
  noStroke(); fill('#555'); textAlign(LEFT, TOP); textSize(10); text('View 1: Edge dropout ' + Math.round(edgeDrop * 100) + '%', gx + 6, gy);
  drawViewGraph(gx, gy + 14, gw, hh - 14, 1);
  // View 2 (bottom): feature masking
  noStroke(); fill('#555'); text('View 2: Feature masking ' + Math.round(featMask * 100) + '%', gx + 6, gy + hh + 4);
  drawViewGraph(gx, gy + hh + 18, gw, hh - 14, 2);
}
function drawViewGraph(gx, gy, gw, gh, view) {
  for (let ei = 0; ei < edges.length; ei++) {
    let e = edges[ei], a = nodeXY(e[0], gx, gy, gw, gh), b = nodeXY(e[1], gx, gy, gw, gh);
    if (view === 1 && dropped[ei]) { stroke('#e53935'); strokeWeight(1); drawingContext.setLineDash([3, 3]); line(a.x, a.y, b.x, b.y); drawingContext.setLineDash([]); }
    else { stroke(60, 60, 60, view === 1 ? 120 : 90); strokeWeight(1); line(a.x, a.y, b.x, b.y); }
  }
  for (let i = 0; i < NN; i++) { let p = nodeXY(i, gx, gy, gw, gh); let m = view === 2 && masked[i]; stroke('#888'); strokeWeight(1); fill(m ? '#9e9e9e' : (cls[i] ? '#ff9800' : '#4fc3f7')); circle(p.x, p.y, 11); }
}

function drawEmbed(gx, gy, gw, gh) {
  // axes box
  let x0 = gx + 16, x1 = gx + gw - 12, y0 = gy + gh - 36, y1 = gy + 14;
  let colr = ['#ef5350', '#42a5f5', '#66bb6a', '#ffa726', '#ab47bc', '#26c6da', '#ec407a', '#9ccc65', '#5c6bc0', '#ffca28'];
  for (let i = 0; i < NN; i++) {
    let a = embPt(v1[i], x0, x1, y0, y1), b = embPt(v2[i], x0, x1, y0, y1);
    let sel = (i === selNode);
    stroke(sel ? 'black' : color(red(color(colr[i])), green(color(colr[i])), blue(color(colr[i])), 150)); strokeWeight(sel ? 2 : 1); drawingContext.setLineDash([4, 3]); line(a.x, a.y, b.x, b.y); drawingContext.setLineDash([]);
    noStroke(); fill(colr[i]); circle(a.x, a.y, sel ? 11 : 8); circle(b.x, b.y, sel ? 11 : 8);
    if (dist(mouseX, mouseY, a.x, a.y) < 7 || dist(mouseX, mouseY, b.x, b.y) < 7) hoverP = i;
  }
  // loss
  let loss = ntxent();
  noStroke(); fill('black'); textAlign(CENTER, BOTTOM); textSize(13); text('NT-Xent loss: ' + nf(loss, 1, 3), gx + gw / 2, gy + gh - 2);
  fill('#555'); textSize(9); textAlign(CENTER, BOTTOM); text('dashed = same node, two views', gx + gw / 2, gy + gh - 18);
}
function embPt(v, x0, x1, y0, y1) { return { x: map(v[0], -1.6, 1.6, x0, x1), y: map(v[1], -1.6, 1.6, y0, y1) }; }

function drawTip() {
  let lines = ['Node ' + (hoverP + 1), 'class: ' + (cls[hoverP] ? 'B (orange)' : 'A (blue)'), 'view distance: ' + nf(Math.hypot(v1[hoverP][0] - v2[hoverP][0], v1[hoverP][1] - v2[hoverP][1]), 1, 2)];
  let w = 160, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- model ----------

function ntxent() {
  // similarity = cosine; loss averaged over positives
  let L = 0;
  for (let i = 0; i < NN; i++) {
    let num = Math.exp(cosim(v1[i], v2[i]) / temp);
    let den = 0; for (let j = 0; j < NN; j++) den += Math.exp(cosim(v1[i], v2[j]) / temp);
    L += -Math.log(num / den);
  }
  return L / NN;
}
function cosim(a, b) { let d = a[0] * b[0] + a[1] * b[1], na = Math.hypot(a[0], a[1]) + 1e-9, nb = Math.hypot(b[0], b[1]) + 1e-9; return d / (na * nb); }

function resample() {
  rng = mulberry32(Math.floor(1000 * (edgeDrop + featMask) + frameCount));
  dropped = edges.map(() => rng() < edgeDrop);
  masked = pos.map(() => rng() < featMask);
  // view embeddings = base + augmentation noise (stronger aug → more noise)
  let amp1 = 0.15 + edgeDrop * 2.2, amp2 = 0.15 + featMask * 2.2;
  v1 = base.map(b => [b[0] + gauss() * amp1, b[1] + gauss() * amp1]);
  v2 = base.map((b, i) => [b[0] + gauss() * amp2 + (masked[i] ? gauss() * 0.6 : 0), b[1] + gauss() * amp2 + (masked[i] ? gauss() * 0.6 : 0)]);
}
function gauss() { return (rng() + rng() + rng() - 1.5) * 0.9; }

function buildGraph() {
  let r0 = mulberry32(3);
  pos = []; for (let i = 0; i < NN; i++) { let a = TWO_PI * i / NN; pos.push({ x: 0.5 + 0.38 * Math.cos(a), y: 0.5 + 0.38 * Math.sin(a) }); }
  cls = []; for (let i = 0; i < NN; i++) cls.push(i < 5 ? 0 : 1);
  edges = []; let set = new Set();
  for (let i = 0; i < NN; i++) { let j = (i + 1) % NN; edges.push([i, j]); set.add(i + '_' + j); }
  while (edges.length < 15) { let i = Math.floor(r0() * NN), j = Math.floor(r0() * NN); if (i === j) continue; let k = Math.min(i, j) + '_' + Math.max(i, j); if (set.has(k)) continue; set.add(k); edges.push([Math.min(i, j), Math.max(i, j)]); }
  // base embeddings clustered by class
  base = []; for (let i = 0; i < NN; i++) { let cx = cls[i] ? 0.7 : -0.7; base.push([cx + (r0() - 0.5) * 0.8, (r0() - 0.5) * 1.4]); }
}

function mousePressed() {
  let w3 = canvasWidth / 3;
  if (mouseX < w3 && mouseY > 24 && mouseY < drawHeight) { for (let i = 0; i < NN; i++) { let p = nodeXY(i, 0, 28, w3, drawHeight - 36); if (dist(mouseX, mouseY, p.x, p.y) <= 11) { selNode = (selNode === i ? -1 : i); return; } } }
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
