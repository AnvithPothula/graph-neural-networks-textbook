// Label Propagation Step-by-Step Simulator (Karate Club)
// CANVAS_HEIGHT: 520
// Learning objective (Understand, Bloom L2): watch label information flow outward
// from two seed nodes, building intuition for the harmonic-averaging property and
// convergence of label propagation / spreading.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 215;
let defaultTextSize = 16;

let stepButton, autoButton, resetButton, alphaSlider;

const N = 34;
let adj = [], faction = [], pos = [];
let f = [];           // soft label in [0,1]
const SEEDS = { 0: 0, 33: 1 };  // node -> fixed class
let iter = 0, converged = false, convergedAt = -1;
let alpha = 0.8;
let autoOn = false, lastStep = 0;
let hoverNode = -1;

const BLUE = '#3b82f6', RED = '#ef4444';

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildKarate();

  stepButton = createButton('Step');
  stepButton.position(10, drawHeight + 8);
  stepButton.mousePressed(() => { autoOn = false; autoButton.html('Auto'); doStep(); });

  autoButton = createButton('Auto');
  autoButton.position(64, drawHeight + 8);
  autoButton.mousePressed(toggleAuto);

  resetButton = createButton('Reset');
  resetButton.position(118, drawHeight + 8);
  resetButton.mousePressed(resetLabels);

  alphaSlider = createSlider(0.1, 0.9, 0.8, 0.1);
  alphaSlider.position(sliderLeftMargin, drawHeight + 45);
  alphaSlider.size(canvasWidth - sliderLeftMargin - margin);
  alphaSlider.input(() => alpha = alphaSlider.value());

  resetLabels();
  describe('Label propagation on the Zachary Karate Club graph. Node 0 (blue) and ' +
    'node 33 (red) are fixed seeds; every other node updates toward the average of its ' +
    'neighbors each step. Watch labels diffuse and converge to a harmonic solution; ' +
    'accuracy is compared against the true factions.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  if (autoOn && !converged && millis() - lastStep > 600) { doStep(); lastStep = millis(); }
  if (autoOn && converged) { autoOn = false; autoButton.html('Auto'); }

  // title + status
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(20);
  text('Label Propagation Stepper', canvasWidth / 2, 6);
  textSize(14);
  let status = converged ? ('Converged at iteration ' + convergedAt) : ('Iteration: ' + iter);
  fill(converged ? '#2e7d32' : '#333');
  textAlign(LEFT, TOP); text(status, 12, 32);
  textAlign(RIGHT, TOP); fill('#333');
  text('Accuracy: ' + (iter === 0 ? '— (step to begin)' : accuracy() + '%'), canvasWidth - 12, 32);

  drawGraph(0, 54, canvasWidth, drawHeight - 54 - 8);
  if (hoverNode >= 0) drawTooltip();

  // control labels
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('Graph influence α = ' + nf(alpha, 1, 1), 10, drawHeight + 52);
}

function drawGraph(gx, gy, gw, gh) {
  hoverNode = -1;
  // edges
  stroke(205); strokeWeight(1);
  for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) {
    let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh);
    line(a.x, a.y, b.x, b.y);
  }
  // nodes
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    if (dist(mouseX, mouseY, p.x, p.y) <= 12) hoverNode = i;
    let col = lerpColor(color(BLUE), color(RED), constrain(f[i], 0, 1));
    let isSeed = (i in SEEDS);
    let uncertain = Math.abs(f[i] - 0.5) < 0.05;
    if (isSeed) { stroke('black'); strokeWeight(3); drawingContext.setLineDash([4, 3]); }
    else if (uncertain) { stroke(150); strokeWeight(1); drawingContext.setLineDash([]); }
    else { stroke(40); strokeWeight(2); drawingContext.setLineDash([]); }
    fill(col); circle(p.x, p.y, 24);
    drawingContext.setLineDash([]);
    noStroke(); fill(Math.abs(f[i] - 0.5) > 0.3 ? 'white' : 'black');
    textAlign(CENTER, CENTER); textSize(10); text(i, p.x, p.y);
  }
}

function drawTooltip() {
  let i = hoverNode;
  let lines = ['Node ' + i, 'f = ' + nf(f[i], 1, 2), 'degree = ' + adj[i].length + (i in SEEDS ? '  (seed)' : '')];
  let w = 130, h = lines.length * 18 + 12;
  let p = nodeScreenCached(i);
  let tx = constrain(p.x + 16, 4, canvasWidth - w - 4), ty = constrain(p.y - h / 2, 50, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 18);
}

// ---------- label propagation ----------

function resetLabels() {
  autoOn = false; if (autoButton) autoButton.html('Auto');
  f = new Array(N).fill(0.5);
  for (let s in SEEDS) f[s] = SEEDS[s];
  iter = 0; converged = false; convergedAt = -1;
}

function doStep() {
  if (converged) return;
  let nf2 = f.slice();
  let maxChange = 0;
  for (let v = 0; v < N; v++) {
    if (v in SEEDS) continue;
    if (adj[v].length === 0) continue;
    let mean = adj[v].reduce((s, u) => s + f[u], 0) / adj[v].length;
    nf2[v] = (1 - alpha) * f[v] + alpha * mean;
    maxChange = Math.max(maxChange, Math.abs(nf2[v] - f[v]));
  }
  f = nf2; iter++;
  if (maxChange < 0.001) { converged = true; convergedAt = iter; }
}

function accuracy() {
  let correct = 0;
  for (let v = 0; v < N; v++) {
    let pred = f[v] >= 0.5 ? 1 : 0;
    if (pred === faction[v]) correct++;
  }
  return Math.round(100 * correct / N);
}

// ---------- interaction ----------

function toggleAuto() { if (converged) return; autoOn = !autoOn; autoButton.html(autoOn ? 'Pause' : 'Auto'); lastStep = 0; }

// ---------- layout helpers ----------

let _gx, _gy, _gw, _gh;
function nodeScreen(i, gx, gy, gw, gh) { _gx = gx; _gy = gy; _gw = gw; _gh = gh; return { x: gx + 24 + pos[i].x * (gw - 48), y: gy + 16 + pos[i].y * (gh - 32) }; }
function nodeScreenCached(i) { return { x: _gx + 24 + pos[i].x * (_gw - 48), y: _gy + 16 + pos[i].y * (_gh - 32) }; }

// ---------- Karate graph ----------

function buildKarate() {
  let edges = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],[14,33],
    [15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],[23,33],
    [24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  let officer = new Set([8,9,14,15,18,20,22,23,24,25,26,27,28,29,30,31,32,33]);
  faction = []; for (let i = 0; i < N; i++) faction.push(officer.has(i) ? 1 : 0);
  adj = Array.from({ length: N }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  pos = springLayout(edges);
}

// deterministic spring layout (seeded), normalized to [0,1]
function springLayout(edges) {
  let p = [];
  for (let i = 0; i < N; i++) { let a = TWO_PI * i / N; p.push({ x: 0.5 + 0.4 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a) }); }
  for (let iterC = 0; iterC < 250; iterC++) {
    let fx = new Array(N).fill(0), fy = new Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y;
      let d2 = dx * dx + dy * dy + 0.0001, d = Math.sqrt(d2);
      let rep = 0.0008 / d2;
      fx[i] += dx / d * rep; fy[i] += dy / d * rep;
      fx[j] -= dx / d * rep; fy[j] -= dy / d * rep;
    }
    for (let e of edges) {
      let a = p[e[0]], b = p[e[1]];
      let dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.0001;
      let spr = (d - 0.12) * 0.02;
      fx[e[0]] += dx / d * spr; fy[e[0]] += dy / d * spr;
      fx[e[1]] -= dx / d * spr; fy[e[1]] -= dy / d * spr;
    }
    for (let i = 0; i < N; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  // normalize to [0,1]
  let xs = p.map(q => q.x), ys = p.map(q => q.y);
  let minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - minx) / (maxx - minx + 1e-9), y: (q.y - miny) / (maxy - miny + 1e-9) }));
}

// ---------- responsive ----------

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  alphaSlider.size(canvasWidth - sliderLeftMargin - margin);
  redraw();
}
function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
