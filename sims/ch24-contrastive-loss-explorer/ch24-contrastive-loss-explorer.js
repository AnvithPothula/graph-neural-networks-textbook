// Contrastive Loss Explorer — augmentation strength, temperature, and the loss landscape
// CANVAS_HEIGHT: 480
// Learning objective (Apply): manipulate augmentation strength and temperature to see how
// they move the operating point on the NT-Xent loss landscape.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 400;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let strSlider, tempSlider, batchSelect, computeButton;
const NN = 8;
let pos = [], edges = [], dropped = [], masked = [];
let strength = 0.2, temp = 0.5, batch = 64, hoverHeat = null, animT = -1;
let rng = mulberry32(5);

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildGraph(); applyAug();

  strSlider = createSlider(0, 50, 20, 1); strSlider.position(sliderLeftMargin, drawHeight + 8); strSlider.size(180); strSlider.input(() => { strength = strSlider.value() / 100; applyAug(); });
  tempSlider = createSlider(0.1, 1.0, 0.5, 0.05); tempSlider.position(sliderLeftMargin, drawHeight + 38); tempSlider.size(180); tempSlider.input(() => temp = tempSlider.value());
  batchSelect = createSelect(); batchSelect.position(sliderLeftMargin + 250, drawHeight + 8); [16, 64, 256].forEach(b => batchSelect.option(b)); batchSelect.value(64); batchSelect.changed(() => batch = parseInt(batchSelect.value()));
  computeButton = createButton('Compute Loss'); computeButton.position(sliderLeftMargin + 250, drawHeight + 38); computeButton.mousePressed(() => animT = millis());

  describe('Contrastive loss explorer. Left: an 8-node graph shown as two overlaid augmented ' +
    'views (blue and orange); the augmentation-strength slider drops edges (dotted) and masks ' +
    'features (gray). Right: the NT-Xent loss as a heatmap over view similarity (x) and ' +
    'temperature τ (y), with a crosshair at the current operating point. Stronger augmentation ' +
    'lowers similarity (moves left); τ sets how sharp the contrast is.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('#f0f4ff'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let half = canvasWidth / 2;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(15);
  text('Augmented views', half / 2, 6); text('NT-Xent loss landscape', half + half / 2, 6);
  stroke('silver'); line(half, 24, half, drawHeight - 8);

  hoverHeat = null;
  drawViews(0, 28, half, drawHeight - 36);
  drawHeatmap(half, 28, half, drawHeight - 36);
  if (hoverHeat) drawHeatTip();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(12);
  text('Aug strength ' + Math.round(strength * 100) + '%', 10, drawHeight + 16);
  text('Temperature τ ' + nf(temp, 1, 2), 10, drawHeight + 46);
  text('Batch (negatives): ' + batch, sliderLeftMargin + 250 - 0, drawHeight + 64);
}

function nodeXY(i, gx, gy, gw, gh, dx, dy) { return { x: gx + 30 + pos[i].x * (gw - 60) + dx, y: gy + 16 + pos[i].y * (gh - 32) + dy }; }

function drawViews(gx, gy, gw, gh) {
  // two overlaid translucent layers, offset slightly by strength
  let off = 3 + strength * 14;
  // view 1 (blue) offset up-left, view 2 (orange) offset down-right
  for (let layer = 0; layer < 2; layer++) {
    let dx = layer === 0 ? -off : off, dy = layer === 0 ? -off : off;
    let col = layer === 0 ? color(66, 165, 245, 150) : color(255, 152, 0, 150);
    for (let ei = 0; ei < edges.length; ei++) {
      let e = edges[ei], a = nodeXY(e[0], gx, gy, gw, gh, dx, dy), b = nodeXY(e[1], gx, gy, gw, gh, dx, dy);
      stroke(col);
      if (dropped[ei]) { strokeWeight(1); drawingContext.setLineDash([3, 3]); line(a.x, a.y, b.x, b.y); drawingContext.setLineDash([]); }
      else { strokeWeight(1.5); line(a.x, a.y, b.x, b.y); }
    }
    for (let i = 0; i < NN; i++) { let p = nodeXY(i, gx, gy, gw, gh, dx, dy); noStroke(); fill(col); circle(p.x, p.y, 16); if (masked[i]) { fill(120); circle(p.x, p.y, 7); } }
  }
  noStroke(); fill('#42a5f5'); textAlign(LEFT, CENTER); textSize(11); text('● View 1', gx + 8, gy + gh - 8); fill('#ff9800'); text('● View 2', gx + 70, gy + gh - 8); fill('#777'); text('dotted = dropped edge, gray center = masked', gx + 140, gy + gh - 8);
}

function lossAt(sim, tau) { return Math.log(1 + batch * Math.exp(-sim / tau)); }

function drawHeatmap(gx, gy, gw, gh) {
  let x0 = gx + 34, x1 = gx + gw - 14, y0 = gy + gh - 28, y1 = gy + 14;
  let cols = 40, rows = 30, cw = (x1 - x0) / cols, ch = (y0 - y1) / rows;
  let lmax = lossAt(-1, 0.1), lmin = lossAt(1, 1.0);
  for (let cx = 0; cx < cols; cx++) for (let cy = 0; cy < rows; cy++) {
    let sim = map(cx + 0.5, 0, cols, -1, 1), tau = map(cy + 0.5, 0, rows, 1.0, 0.1);
    let L = lossAt(sim, tau), f = constrain((L - lmin) / (lmax - lmin), 0, 1);
    noStroke(); fill(lerpColor(color('#1565c0'), color('#c62828'), f)); rect(x0 + cx * cw, y1 + cy * ch, cw + 1, ch + 1);
  }
  // axes
  noStroke(); fill('#333'); textAlign(CENTER, TOP); textSize(10); text('cosine similarity (views)', (x0 + x1) / 2, y0 + 6);
  textAlign(LEFT, CENTER); text('-1', x0, y0 + 12); textAlign(RIGHT, CENTER); text('+1', x1, y0 + 12);
  push(); translate(gx + 14, (y0 + y1) / 2); rotate(-HALF_PI); textAlign(CENTER, CENTER); text('temperature τ', 0, 0); pop();
  // crosshair at current operating point
  let curSim = constrain(1 - strength * 2.4, -1, 1);
  let px = map(curSim, -1, 1, x0, x1), py = map(temp, 1.0, 0.1, y1, y0);
  let pulse = animT >= 0 && millis() - animT < 900 ? 1 + 0.4 * Math.sin((millis() - animT) * 0.02) : 1;
  stroke('white'); strokeWeight(1.5); line(px - 10 * pulse, py, px + 10 * pulse, py); line(px, py - 10 * pulse, px, py + 10 * pulse);
  noStroke(); fill('white'); circle(px, py, 5);
  // current loss
  let L = lossAt(curSim, temp);
  noStroke(); fill('black'); textAlign(RIGHT, TOP); textSize(12); text('loss ≈ ' + nf(L, 1, 2), x1, y1 - 2);
  // hover
  if (mouseX > x0 && mouseX < x1 && mouseY > y1 && mouseY < y0) { let s = map(mouseX, x0, x1, -1, 1), t = map(mouseY, y1, y0, 1.0, 0.1); hoverHeat = { s, t, L: lossAt(s, t) }; }
}

function drawHeatTip() {
  let learn = hoverHeat.t > 0.6 ? 'High τ → soft distribution, slow learning' : hoverHeat.t < 0.3 ? 'Low τ → sharp distribution, fast but unstable' : 'Moderate τ → balanced contrast';
  let lines = ['sim ' + nf(hoverHeat.s, 1, 2) + ', τ ' + nf(hoverHeat.t, 1, 2), 'loss ' + nf(hoverHeat.L, 1, 2), learn];
  let w = 250, h = lines.length * 16 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 250); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(11.5);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 16, w - 16);
}

function applyAug() { rng = mulberry32(Math.floor(strength * 500) + 1); dropped = edges.map(() => rng() < strength); masked = pos.map(() => rng() < strength); }

function buildGraph() {
  let r0 = mulberry32(9);
  pos = []; for (let i = 0; i < NN; i++) { let a = TWO_PI * i / NN; pos.push({ x: 0.5 + 0.32 * Math.cos(a), y: 0.5 + 0.32 * Math.sin(a) }); }
  edges = []; let set = new Set();
  for (let i = 0; i < NN; i++) { let j = (i + 1) % NN; edges.push([i, j]); set.add(i + '_' + j); }
  while (edges.length < 11) { let i = Math.floor(r0() * NN), j = Math.floor(r0() * NN); if (i === j) continue; let k = Math.min(i, j) + '_' + Math.max(i, j); if (set.has(k)) continue; set.add(k); edges.push([Math.min(i, j), Math.max(i, j)]); }
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
