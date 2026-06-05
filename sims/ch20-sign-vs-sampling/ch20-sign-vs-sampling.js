// SIGN vs Neighbor Sampling — architecture comparison
// CANVAS_HEIGHT: 570
// Learning objective (Apply): compare neighbor sampling's in-loop O(K^L) graph ops
// against SIGN's one-time precomputation (A^k X) and O(d)-per-node training.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 470;
let controlHeight = 100;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let kSlider, lSlider, memButton;
let fanK = 4, layersL = 2, showMem = false, hoverInfo = null;
const HOP_COL = ['#5c6bc0', '#26a69a', '#ffa726', '#ef5350'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  memButton = createButton('Show Memory Usage'); memButton.position(10, drawHeight + 8); memButton.mousePressed(() => showMem = !showMem);
  kSlider = createSlider(2, 20, 4, 1); kSlider.position(sliderLeftMargin, drawHeight + 42); kSlider.size(canvasWidth - sliderLeftMargin - margin); kSlider.input(() => fanK = kSlider.value());
  lSlider = createSlider(1, 3, 2, 1); lSlider.position(sliderLeftMargin, drawHeight + 72); lSlider.size(canvasWidth - sliderLeftMargin - margin); lSlider.input(() => layersL = lSlider.value());

  describe('Side-by-side comparison of neighbor sampling and SIGN. Left: a target node and ' +
    'L layers of K sampled neighbors, with in-loop cost O(K^L) per step that grows as you raise ' +
    'fan-out K and depth L. Right: SIGN precomputes A^k·X once offline, so each training step is ' +
    'O(d) per node with no graph ops. Toggle memory usage to compare footprints.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let half = canvasWidth / 2;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(15);
  text('Neighbor Sampling', half / 2, 6); text('SIGN Pre-computation', half + half / 2, 6);
  stroke('silver'); line(half, 26, half, drawHeight - 8);

  hoverInfo = null;
  drawSampling(0, 30, half, drawHeight - 78);
  drawSIGN(half, 30, half, drawHeight - 78);
  if (hoverInfo) drawTip();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13);
  text('Fan-out K = ' + fanK, 10, drawHeight + 49);
  text('Layers L = ' + layersL, 10, drawHeight + 79);
}

// ---------- left: neighbor sampling ----------

function drawSampling(gx, gy, gw, gh) {
  let capPerParent = 4;   // display cap to avoid clutter
  // columns: target at right, hop-L at left
  let colX = [];
  for (let l = 0; l <= layersL; l++) colX.push(gx + gw - 60 - l * (gw - 120) / layersL);
  // build tree of displayed nodes per layer
  let layers = [[{ x: colX[0], y: gy + gh / 2, hop: 0 }]];
  for (let l = 1; l <= layersL; l++) {
    let prev = layers[l - 1], cur = [];
    let shown = Math.min(fanK, capPerParent);
    let total = prev.length * shown;
    let i = 0;
    for (let par of prev) {
      for (let s = 0; s < shown; s++) {
        let y = gy + 24 + (gh - 48) * (i + 0.5) / total;
        cur.push({ x: colX[l], y, hop: l, parent: par });
        i++;
      }
    }
    layers.push(cur);
  }
  // edges (messages flow inward: from hop l to hop l-1)
  for (let l = 1; l <= layersL; l++) for (let nd of layers[l]) { stroke(HOP_COL[Math.min(l, 3)]); strokeWeight(1.2); drawArrowSeg(nd.x, nd.y, nd.parent.x, nd.parent.y, 10); }
  // nodes
  for (let l = layersL; l >= 0; l--) for (let nd of layers[l]) {
    let r = l === 0 ? 14 : 8;
    if (dist(mouseX, mouseY, nd.x, nd.y) <= r + 2) hoverInfo = { lines: [l === 0 ? 'Target node' : l + '-hop sampled neighbor', l === 0 ? 'the node being classified' : 'one of K=' + fanK + ' sampled at hop ' + l] };
    stroke(40); strokeWeight(1); fill(l === 0 ? '#ff9800' : HOP_COL[Math.min(l, 3)]); circle(nd.x, nd.y, r * 2);
  }
  // "+more" note if capped
  if (fanK > capPerParent) { noStroke(); fill('#888'); textAlign(CENTER, TOP); textSize(10); text('(showing ' + capPerParent + ' of K=' + fanK + ' per node)', gx + gw / 2, gy + gh - 30); }
  // cost label
  let cost = Math.pow(fanK, layersL);
  noStroke(); fill('#c62828'); textAlign(CENTER, BOTTOM); textSize(13); textStyle(BOLD);
  text('In-loop graph ops: O(Kᴸ) ≈ ' + cost + ' / step', gx + gw / 2, gy + gh - 4); textStyle(NORMAL);
  if (showMem) memBar(gx + 20, gy + gh + 10, gw - 40, cost, Math.pow(20, 3), '#c62828', 'subgraph');
}

// ---------- right: SIGN ----------

function drawSIGN(gx, gy, gw, gh) {
  let cx = gx + gw / 2;
  // offline block
  noStroke(); fill('#eceff1'); stroke('#b0bec5'); strokeWeight(1); rect(gx + 20, gy + 16, gw - 40, 150, 6);
  noStroke(); fill('#37474f'); textAlign(LEFT, TOP); textSize(12); textStyle(BOLD); text('Offline (once):', gx + 28, gy + 22); textStyle(NORMAL);
  // stacked matrices A^k X
  let mw = (gw - 80) / (layersL + 1);
  for (let k = 0; k <= layersL; k++) {
    let mx = gx + 32 + k * mw, my = gy + 44, mh = 100;
    if (mouseX > mx && mouseX < mx + mw - 8 && mouseY > my && mouseY < my + mh) hoverInfo = { lines: ['Aᵏ·X  (k=' + k + ')', k === 0 ? 'raw node features X' : 'features diffused over ' + k + '-hop neighborhoods', 'precomputed once, reused every epoch'] };
    stroke(60); strokeWeight(1); fill(HOP_COL[Math.min(k, 3)]); rect(mx, my, mw - 8, mh, 3);
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(11); text('A' + supK(k) + 'X', mx + (mw - 8) / 2, my + mh / 2);
  }
  // arrow to online block
  stroke('#555'); strokeWeight(1.5); drawArrowSeg(cx, gy + 168, cx, gy + 196, 0);
  noStroke(); fill('#555'); textAlign(CENTER, TOP); textSize(11); text('pre-computed once →', cx, gy + 170);
  // online block: MLP
  fill('#e8f5e9'); stroke('#a5d6a7'); strokeWeight(1); rect(gx + 20, gy + 200, gw - 40, gh - 220, 6);
  noStroke(); fill('#1b5e20'); textAlign(LEFT, TOP); textSize(12); textStyle(BOLD); text('Online training:', gx + 28, gy + 206); textStyle(NORMAL);
  // MLP nodes: inputs (L+1) -> hidden -> output
  let ix = gx + 60, ox = gx + gw - 70, midx = (ix + ox) / 2, mty = gy + 200, mbh = gh - 220;
  let inY = []; for (let k = 0; k <= layersL; k++) inY.push(mty + 36 + (mbh - 60) * k / Math.max(layersL, 1));
  let hidY = [mty + mbh * 0.4, mty + mbh * 0.65];
  for (let y of inY) for (let hy of hidY) { stroke('#aaa'); strokeWeight(0.7); line(ix, y, midx, hy); }
  for (let hy of hidY) { stroke('#aaa'); line(midx, hy, ox, mty + mbh / 2); }
  for (let k = 0; k <= layersL; k++) { noStroke(); fill(HOP_COL[Math.min(k, 3)]); circle(ix, inY[k], 14); }
  for (let hy of hidY) { noStroke(); fill('#66bb6a'); circle(midx, hy, 14); }
  noStroke(); fill('#2e7d32'); circle(ox, mty + mbh / 2, 16);
  fill('#555'); textAlign(CENTER, TOP); textSize(9); text('concat ' + (layersL + 1) + ' vectors', ix, mty + 16); text('MLP', midx, mty + 18); text('ŷ', ox, mty + mbh / 2 - 26);
  // cost
  fill('#2e7d32'); textAlign(CENTER, BOTTOM); textSize(13); textStyle(BOLD); text('No graph ops in loop: O(d) / node', gx + gw / 2, gy + gh - 4); textStyle(NORMAL);
  if (showMem) memBar(gx + 20, gy + gh + 10, gw - 40, (layersL + 1), 4, '#2e7d32', 'precomputed');
}

function memBar(x, y, w, val, maxVal, col, label) {
  noStroke(); fill('#eee'); rect(x, y, w, 12, 3);
  fill(col); rect(x, y, w * constrain(val / maxVal, 0.02, 1), 12, 3);
  fill('#333'); textAlign(LEFT, BOTTOM); textSize(10); text('memory (' + label + ')', x, y - 1);
}

function supK(k) { return ['⁰', '¹', '²', '³', '⁴'][k] || ('^' + k); }

function drawArrowSeg(x1, y1, x2, y2, rad) {
  let ang = atan2(y2 - y1, x2 - x1), ex = x2 - cos(ang) * rad, ey = y2 - sin(ang) * rad;
  line(x1, y1, ex, ey); push(); translate(ex, ey); rotate(ang); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -7, 2.5, -7, -2.5); pop();
}

function drawTip() {
  let w = 230, h = hoverInfo.lines.length * 16 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 30, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < hoverInfo.lines.length; k++) { if (k === 0) textStyle(BOLD); else textStyle(NORMAL); text(hoverInfo.lines[k], tx + 8, ty + 6 + k * 16); }
  textStyle(NORMAL);
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); kSlider.size(canvasWidth - sliderLeftMargin - margin); lSlider.size(canvasWidth - sliderLeftMargin - margin); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
