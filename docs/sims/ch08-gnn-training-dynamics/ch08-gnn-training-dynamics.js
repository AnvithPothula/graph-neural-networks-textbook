// GNN Training Dynamics - depth, over-smoothing, and early stopping
// CANVAS_HEIGHT: 530
// Learning objective (Understand/Analyze): see how training loss and validation
// accuracy diverge with GCN depth, making the over-smoothing feedback loop tangible.
// Residual connections and dropout mitigate the collapse.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 420;
let controlHeight = 110;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let runButton, resetButton, skipCheckbox, depthSlider, dropoutSlider;

let depth = 4, dropout = 0.3, skip = false;
let lossArr = [], accArr = [], bestEpoch = 0;
let curEpoch = 0, running = false;
let nodes20 = [], hoverChart = null;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  runButton = createButton('Run');
  runButton.position(10, drawHeight + 8);
  runButton.mousePressed(toggleRun);
  resetButton = createButton('Reset');
  resetButton.position(64, drawHeight + 8);
  resetButton.mousePressed(() => { curEpoch = 0; running = false; runButton.html('Run'); });
  skipCheckbox = createCheckbox(' Residual (skip) connections', false);
  skipCheckbox.position(128, drawHeight + 10);
  skipCheckbox.changed(() => { skip = skipCheckbox.checked(); simulate(); });

  depthSlider = createSlider(1, 16, 4, 1);
  depthSlider.position(sliderLeftMargin, drawHeight + 42);
  depthSlider.input(() => { depth = depthSlider.value(); simulate(); });
  dropoutSlider = createSlider(0, 0.8, 0.3, 0.05);
  dropoutSlider.position(sliderLeftMargin, drawHeight + 76);
  dropoutSlider.input(() => { dropout = dropoutSlider.value(); simulate(); });
  sizeSliders();

  initNodes();
  simulate();
  curEpoch = 300;   // show the full curves on load; Run restarts the animation
  describe('GNN training dynamics. The left chart overlays training loss (red, left ' +
    'axis) and validation accuracy (blue, right axis) over 300 epochs; a dashed line marks ' +
    'the best checkpoint and an early-stopping annotation. Increase depth to watch ' +
    'over-smoothing crush accuracy, then enable residual connections or dropout to recover. ' +
    'The right panel shows 20 nodes turning to their correct class as accuracy rises.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  if (running) { curEpoch += 2; if (curEpoch >= 300) { curEpoch = 300; running = false; runButton.html('Run'); } }

  let chartW = canvasWidth * 0.66;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18);
  text('GNN Training Dynamics', chartW / 2, 6);

  drawChart(40, 40, chartW - 92, drawHeight - 40 - 40);
  drawGraphPanel(chartW + 10, 40, canvasWidth - chartW - 10 - margin, drawHeight - 48);
  drawControlLabels();
  if (hoverChart) drawChartTooltip();
}

function drawChart(gx, gy, gw, gh) {
  hoverChart = null;
  let x0 = gx, x1 = gx + gw, y0 = gy + gh, y1 = gy;
  let xFor = e => map(e, 0, 300, x0, x1);
  let lossFor = l => map(constrain(l, 0, 1.8), 0, 1.8, y0, y1);
  let accFor = a => map(constrain(a, 0, 1), 0, 1, y0, y1);

  // axes
  stroke(200); strokeWeight(1); line(x0, y0, x1, y0); line(x0, y0, x0, y1); line(x1, y0, x1, y1);
  noStroke(); textSize(11);
  fill('#c62828'); textAlign(RIGHT, CENTER); text('loss', x0 - 4, y1 + 6);
  fill('#1565c0'); textAlign(LEFT, CENTER); text('val acc', x1 + 4, y1 + 6);
  fill('dimgray'); textAlign(CENTER, TOP); text('epoch', (x0 + x1) / 2, y0 + 4);
  textAlign(LEFT, CENTER); text('0', x0, y0 + 10); textAlign(RIGHT, CENTER); text('300', x1, y0 + 10);
  fill('#1565c0'); textAlign(LEFT, CENTER); text('1.0', x1 + 4, y1 + 18);

  let upto = Math.floor(curEpoch);

  // best-checkpoint vertical line
  if (upto >= bestEpoch) {
    stroke('#777'); strokeWeight(1); drawingContext.setLineDash([4, 4]);
    line(xFor(bestEpoch), y1, xFor(bestEpoch), y0); drawingContext.setLineDash([]);
    noStroke(); fill('#555'); textSize(10); textAlign(CENTER, TOP);
    text('best @ ' + bestEpoch, xFor(bestEpoch), y1 - 2);
    // early-stopping annotation
    let stopEp = Math.min(300, bestEpoch + 50);
    if (upto >= stopEp) {
      stroke('#e65100'); strokeWeight(1); drawingContext.setLineDash([2, 3]);
      line(xFor(stopEp), y1, xFor(stopEp), y0); drawingContext.setLineDash([]);
      noStroke(); fill('#e65100'); textAlign(CENTER, BOTTOM); text('STOP (patience=50)', xFor(stopEp), y0 - 2);
    }
  }

  // loss curve (red)
  stroke('#c62828'); strokeWeight(2); noFill(); beginShape();
  for (let e = 0; e <= upto; e++) vertex(xFor(e), lossFor(lossArr[e]));
  endShape();
  // acc curve (blue)
  stroke('#1565c0'); strokeWeight(2); noFill(); beginShape();
  for (let e = 0; e <= upto; e++) vertex(xFor(e), accFor(accArr[e]));
  endShape();

  // hover
  if (mouseX >= x0 && mouseX <= x1 && mouseY >= y1 && mouseY <= y0) {
    let e = Math.round(map(mouseX, x0, x1, 0, 300));
    e = constrain(e, 0, upto);
    if (e >= 0 && upto > 0) hoverChart = { e, loss: lossArr[e], acc: accArr[e], x: xFor(e) };
  }

  // current accuracy readout
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  text('epoch ' + upto + '   loss ' + nf(lossArr[upto] || 0, 1, 2) + '   val acc ' + nf(accArr[upto] || 0, 1, 2), x0, y0 + 22);
}

function drawGraphPanel(px, py, pw, ph) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, py, pw, ph);
  noStroke(); fill('black'); textAlign(CENTER, TOP); textSize(13);
  text('Predicted classes (20 nodes)', px + pw / 2, py + 4);
  let acc = accArr[Math.floor(curEpoch)] || 0;
  // edges
  stroke(220); strokeWeight(1);
  for (let e of nodes20.edges) {
    let a = nodes20.pts[e[0]], b = nodes20.pts[e[1]];
    line(px + a.x * pw, py + 24 + a.y * (ph - 34), px + b.x * pw, py + 24 + b.y * (ph - 34));
  }
  for (let i = 0; i < 20; i++) {
    let a = nodes20.pts[i];
    let x = px + a.x * pw, y = py + 24 + a.y * (ph - 34);
    // node predicted correct if its deterministic threshold < accuracy
    let correct = nodes20.thr[i] < acc;
    let pred = correct ? nodes20.trueCls[i] : 1 - nodes20.trueCls[i];
    noStroke(); fill(pred === 0 ? '#3b82f6' : '#ef4444');
    circle(x, y, 16);
    stroke(correct ? '#2e7d32' : '#bbb'); strokeWeight(correct ? 2 : 1); noFill(); circle(x, y, 16);
  }
  noStroke(); fill('dimgray'); textAlign(CENTER, BOTTOM); textSize(11);
  text('green ring = correct', px + pw / 2, py + ph - 4);
}

function drawControlLabels() {
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('Depth = ' + depth + ' layers', 10, drawHeight + 49);
  text('Dropout = ' + nf(dropout, 1, 2), 10, drawHeight + 83);
}

function drawChartTooltip() {
  let lines = ['epoch ' + hoverChart.e, 'loss ' + nf(hoverChart.loss, 1, 3), 'val acc ' + nf(hoverChart.acc, 1, 3)];
  let w = 120, h = lines.length * 16 + 10;
  let tx = constrain(hoverChart.x + 10, 4, canvasWidth - w - 4), ty = 50;
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 5 + k * 16);
}

// ---------- synthetic training simulation ----------

function simulate() {
  // over-smoothing penalty: deeper hurts peak; residual delays the onset
  let onset = skip ? 8 : 3;
  let perLayer = skip ? 0.014 : 0.045;
  let oversmooth = Math.max(0, depth - onset) * perLayer;
  let peak = constrain(0.90 - oversmooth - dropout * 0.06, 0.32, 0.90);
  let trainRate = 0.035 / (1 + depth * 0.07);
  lossArr = []; accArr = [];
  for (let e = 0; e <= 300; e++) {
    let loss = 1.65 * Math.exp(-trainRate * e) + 0.05 + 0.02 * Math.sin(e * 0.3) * Math.exp(-e * 0.01);
    let rise = peak * (1 - Math.exp(-0.045 * e));
    let overfit = e > 90 ? (e - 90) * 0.00065 * (1 - dropout) : 0;
    let acc = constrain(rise - overfit, 0, 1);
    lossArr.push(loss); accArr.push(acc);
  }
  bestEpoch = accArr.indexOf(Math.max(...accArr));
  // Show the full new curves immediately on any parameter change; Run restarts the animation.
  curEpoch = 300; running = false; if (runButton) runButton.html('Run');
}

function initNodes() {
  let rng = mulberry32(7);
  let pts = [], trueCls = [], thr = [];
  for (let i = 0; i < 20; i++) {
    let cls = i < 10 ? 0 : 1;
    let cx = cls === 0 ? 0.3 : 0.7;
    pts.push({ x: cx + (rng() - 0.5) * 0.4, y: 0.15 + rng() * 0.7 });
    trueCls.push(cls);
    thr.push(0.15 + rng() * 0.75);   // node becomes correct once acc exceeds this
  }
  // a few intra/inter edges
  let edges = [];
  for (let i = 0; i < 20; i++) for (let k = 0; k < 2; k++) {
    let j = Math.floor(rng() * 20); if (j !== i) edges.push([i, j]);
  }
  nodes20 = { pts, trueCls, thr, edges };
}

function toggleRun() { if (curEpoch >= 300) curEpoch = 0; running = !running; runButton.html(running ? 'Pause' : 'Run'); }

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function sizeSliders() { let w = canvasWidth - sliderLeftMargin - margin; [depthSlider, dropoutSlider].forEach(s => s && s.size(w)); }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); sizeSliders(); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
