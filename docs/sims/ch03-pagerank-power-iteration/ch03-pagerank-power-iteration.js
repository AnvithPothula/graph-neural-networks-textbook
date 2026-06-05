// PageRank Power Iteration Simulator
// CANVAS_HEIGHT: 520
// Learning objective (Apply, Bloom L3): watch power iteration propagate rank,
// see why well-connected nodes accumulate rank, and how the damping factor
// controls convergence. Handles spider traps and dead ends via teleportation.
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

// controls
let stepButton, runButton, resetButton, presetSelect, dampingSlider;

// model
let nodes = [];      // {rx, ry}
let outAdj = [];     // targets per node
let inAdj = [];      // sources per node
let r = [];          // rank vector
let damping = 0.85;
let iter = 0;
let converged = false;
let convergedAt = -1;
let diffHistory = [];   // L1 diffs per step
const THRESHOLD = 1e-6;

// run state
let running = false;
let lastRun = 0;
let selNode = -1;

const INDIGO_DARK = '#1a237e';

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  stepButton = createButton('Step');
  stepButton.position(10, drawHeight + 8);
  stepButton.mousePressed(() => { running = false; runButton.html('Run'); doStep(); });

  runButton = createButton('Run');
  runButton.position(64, drawHeight + 8);
  runButton.mousePressed(toggleRun);

  resetButton = createButton('Reset');
  resetButton.position(128, drawHeight + 8);
  resetButton.mousePressed(() => { running = false; runButton.html('Run'); resetRanks(); });

  presetSelect = createSelect();
  presetSelect.position(196, drawHeight + 8);
  presetSelect.option('Default (spider trap + dead end)');
  presetSelect.option('Karate Club (10-node sample)');
  presetSelect.option('Star graph');
  presetSelect.option('Cycle graph');
  presetSelect.changed(() => { running = false; runButton.html('Run'); loadPreset(); });

  dampingSlider = createSlider(0.5, 1.0, 0.85, 0.05);
  dampingSlider.position(sliderLeftMargin, drawHeight + 45);
  dampingSlider.size(canvasWidth - sliderLeftMargin - margin);
  dampingSlider.input(() => { damping = dampingSlider.value(); resetRanks(); });

  loadPreset();
  describe('Interactive PageRank power iteration on a small directed graph with a ' +
    'spider trap and a dead end. Step or run the iteration and watch node sizes and ' +
    'colors track rank, with a log-scale convergence plot. The damping slider controls ' +
    'teleportation. Click a node for its score and degree.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue'); stroke('silver'); strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  // auto-run
  if (running && !converged && millis() - lastRun > 500) { doStep(); lastRun = millis(); }
  if (running && converged) { running = false; runButton.html('Run'); }

  let graphW = canvasWidth * 0.62;
  let plotX = graphW + 12;
  let plotW = canvasWidth - plotX - margin;

  // title
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(20);
  text('PageRank Power Iteration', graphW / 2, 6);

  // iteration counter
  textSize(15);
  if (converged) { fill('#2e7d32'); text('Converged at iteration ' + convergedAt, graphW / 2, 30); }
  else { fill('#333'); text('Iteration: ' + iter, graphW / 2, 30); }

  drawGraph(0, 52, graphW, drawHeight - 52 - 10);
  drawConvergencePlot(plotX, 52, plotW, 200);
  if (selNode >= 0) drawInfobox(plotX, 270, plotW);

  // control labels
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('Damping d = ' + nf(damping, 1, 2), 10, drawHeight + 52);
}

// ---------- graph drawing ----------

function nodeScreen(i, gx, gy, gw, gh) {
  return { x: gx + 30 + nodes[i].rx * (gw - 60), y: gy + 20 + nodes[i].ry * (gh - 40) };
}

function drawGraph(gx, gy, gw, gh) {
  let maxR = Math.max(...r, 1e-9);

  // directed edges as curved arrows
  for (let j = 0; j < nodes.length; j++) {
    for (let t of outAdj[j]) {
      let a = nodeScreen(j, gx, gy, gw, gh);
      let b = nodeScreen(t, gx, gy, gw, gh);
      let bidir = outAdj[t].includes(j);
      drawArrow(a.x, a.y, b.x, b.y, bidir ? 0.18 : 0.0, nodeRad(t, maxR));
    }
  }

  // nodes
  let topNode = r.indexOf(maxR);
  for (let i = 0; i < nodes.length; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    let rad = nodeRad(i, maxR);
    let frac = constrain(r[i] / maxR, 0, 1);
    if (i === topNode) { // golden glow
      noStroke(); fill(255, 215, 0, 90); circle(p.x, p.y, rad * 2 + 16);
    }
    stroke(i === selNode ? 'black' : INDIGO_DARK);
    strokeWeight(i === selNode ? 3 : 1.5);
    fill(lerpColor(color('#bbdefb'), color(INDIGO_DARK), frac));
    circle(p.x, p.y, rad * 2);
    noStroke(); fill(frac > 0.5 ? 'white' : 'black');
    textAlign(CENTER, CENTER); textSize(13);
    text(i, p.x, p.y);
  }
}

function nodeRad(i, maxR) {
  return map(r[i], 0, maxR, 12, 36);
}

function drawArrow(x1, y1, x2, y2, curve, targetR) {
  // shorten to target circle edge
  let ang = atan2(y2 - y1, x2 - x1);
  let ex = x2 - cos(ang) * (targetR + 3);
  let ey = y2 - sin(ang) * (targetR + 3);
  stroke(150); strokeWeight(1.5); noFill();
  let mx = (x1 + ex) / 2, my = (y1 + ey) / 2;
  // perpendicular offset for curve
  let nx = -(ey - y1), ny = (ex - x1);
  let nlen = Math.sqrt(nx * nx + ny * ny) || 1;
  let cx = mx + (nx / nlen) * curve * dist(x1, y1, ex, ey);
  let cy = my + (ny / nlen) * curve * dist(x1, y1, ex, ey);
  beginShape(); vertex(x1, y1); quadraticVertex(cx, cy, ex, ey); endShape();
  // arrowhead
  let ha = atan2(ey - cy, ex - cx);
  fill(150); noStroke();
  push(); translate(ex, ey); rotate(ha);
  triangle(0, 0, -8, 3, -8, -3); pop();
}

function drawConvergencePlot(px, py, pw, ph) {
  fill('white'); stroke('silver'); strokeWeight(1);
  rect(px, py, pw, ph);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(13);
  text('Convergence: ‖rₜ₊₁ − rₜ‖₁', px + 4, py - 18);

  // log scale from 1e-8 to 1
  let lo = -8, hi = 0;
  let yFor = v => {
    let lg = Math.log10(Math.max(v, 1e-9));
    return map(constrain(lg, lo, hi), lo, hi, py + ph - 6, py + 6);
  };
  // threshold line
  stroke('#c62828'); strokeWeight(1); drawingContext.setLineDash([4, 4]);
  let ty = yFor(THRESHOLD); line(px + 30, ty, px + pw - 4, ty);
  drawingContext.setLineDash([]);
  noStroke(); fill('#c62828'); textSize(10); textAlign(LEFT, CENTER);
  text('1e-6', px + 2, ty);

  // axis labels
  fill('dimgray'); textAlign(LEFT, CENTER); textSize(10);
  text('1', px + 2, py + 8); text('1e-8', px + 2, py + ph - 8);

  // data line
  if (diffHistory.length > 1) {
    stroke('#1565c0'); strokeWeight(2); noFill();
    beginShape();
    for (let k = 0; k < diffHistory.length; k++) {
      let x = map(k, 0, Math.max(diffHistory.length - 1, 1), px + 32, px + pw - 6);
      vertex(x, yFor(diffHistory[k]));
    }
    endShape();
  }
}

function drawInfobox(px, py, pw) {
  let i = selNode;
  let sorted = r.map((v, idx) => [v, idx]).sort((a, b) => b[0] - a[0]);
  let top3 = sorted.slice(0, 3);
  let lines = [
    'Node ' + i,
    'PageRank: ' + nf(r[i], 1, 4),
    'In-degree: ' + inAdj[i].length + '   Out-degree: ' + outAdj[i].length
  ];
  let ph = 110;
  fill(255, 255, 255, 245); stroke(120); strokeWeight(1);
  rect(px, py, pw, ph, 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(13);
  for (let k = 0; k < lines.length; k++) text(lines[k], px + 10, py + 8 + k * 18);
  // mini bar chart vs top-3
  let by = py + 66, bx = px + 10, bw = pw - 70;
  let maxR = top3[0][0] || 1e-9;
  textSize(11);
  for (let k = 0; k < top3.length; k++) {
    let yy = by + k * 14;
    fill('dimgray'); text('n' + top3[k][1], bx, yy);
    fill(top3[k][1] === i ? '#1a237e' : '#90caf9');
    let w = (top3[k][0] / maxR) * bw;
    rect(bx + 26, yy, w, 10, 2);
  }
}

// ---------- PageRank ----------

function resetRanks() {
  let N = nodes.length;
  r = new Array(N).fill(1 / N);
  iter = 0; converged = false; convergedAt = -1;
  diffHistory = [];
  damping = dampingSlider ? dampingSlider.value() : 0.85;
}

function doStep() {
  if (converged) return;
  let N = nodes.length;
  let dangling = 0;
  for (let j = 0; j < N; j++) if (outAdj[j].length === 0) dangling += r[j];
  let base = (1 - damping) / N + damping * dangling / N;
  let nr = new Array(N).fill(base);
  for (let j = 0; j < N; j++) {
    if (outAdj[j].length === 0) continue;
    let share = damping * r[j] / outAdj[j].length;
    for (let t of outAdj[j]) nr[t] += share;
  }
  let diff = 0;
  for (let i = 0; i < N; i++) diff += Math.abs(nr[i] - r[i]);
  r = nr; iter++;
  diffHistory.push(diff);
  if (diff < THRESHOLD) { converged = true; convergedAt = iter; }
}

// ---------- presets ----------

function loadPreset() {
  let name = presetSelect ? presetSelect.value() : 'Default (spider trap + dead end)';
  selNode = -1;
  let dir = [];   // directed edges
  if (name.startsWith('Default')) {
    nodes = [{ rx: 0.15, ry: 0.45 }, { rx: 0.35, ry: 0.15 }, { rx: 0.35, ry: 0.75 },
             { rx: 0.15, ry: 0.85 }, { rx: 0.55, ry: 0.45 }, { rx: 0.55, ry: 0.9 },
             { rx: 0.85, ry: 0.3 }, { rx: 0.85, ry: 0.65 }];
    dir = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 1], [1, 4], [2, 4], [3, 4],
           [4, 0], [4, 5], [4, 6], [6, 7], [7, 6]];
  } else if (name.startsWith('Karate')) {
    placeRing(10);
    let und = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3],
               [4, 5], [5, 6], [5, 7], [6, 7], [0, 5], [8, 9], [2, 8]];
    und.forEach(e => { dir.push([e[0], e[1]]); dir.push([e[1], e[0]]); });
  } else if (name.startsWith('Star')) {
    nodes = [{ rx: 0.5, ry: 0.5 }];
    let leaves = 7;
    for (let k = 0; k < leaves; k++) {
      let a = TWO_PI * k / leaves - HALF_PI;
      nodes.push({ rx: 0.5 + 0.42 * Math.cos(a), ry: 0.5 + 0.42 * Math.sin(a) });
      dir.push([0, k + 1]); dir.push([k + 1, 0]);
    }
  } else {
    let n = 8; placeRing(n);
    for (let k = 0; k < n; k++) dir.push([k, (k + 1) % n]);
  }
  buildAdj(dir);
  resetRanks();
}

function placeRing(n) {
  nodes = [];
  for (let i = 0; i < n; i++) {
    let a = TWO_PI * i / n - HALF_PI;
    nodes.push({ rx: 0.5 + 0.4 * Math.cos(a), ry: 0.5 + 0.4 * Math.sin(a) });
  }
}

function buildAdj(dir) {
  let N = nodes.length;
  outAdj = Array.from({ length: N }, () => []);
  inAdj = Array.from({ length: N }, () => []);
  for (let e of dir) { outAdj[e[0]].push(e[1]); inAdj[e[1]].push(e[0]); }
}

// ---------- interaction ----------

function toggleRun() {
  if (converged) return;
  running = !running;
  runButton.html(running ? 'Pause' : 'Run');
  lastRun = 0;
}

function mousePressed() {
  let gw = canvasWidth * 0.62;
  if (mouseY < 50 || mouseY > drawHeight || mouseX > gw) { return; }
  let maxR = Math.max(...r, 1e-9);
  for (let i = 0; i < nodes.length; i++) {
    let p = nodeScreen(i, 0, 52, gw, drawHeight - 52 - 10);
    if (dist(mouseX, mouseY, p.x, p.y) <= nodeRad(i, maxR) + 2) {
      selNode = (selNode === i) ? -1 : i;
      return;
    }
  }
  selNode = -1;
}

// ---------- responsive ----------

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  dampingSlider.size(canvasWidth - sliderLeftMargin - margin);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
