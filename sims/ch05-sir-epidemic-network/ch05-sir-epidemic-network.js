// SIR Epidemic Dynamics on Network Structures
// CANVAS_HEIGHT: 555
// Learning objective (Analyze, Bloom L4): compare SIR spread on a well-mixed
// population (classical ODE) vs. an explicit network, and see how network
// heterogeneity (hubs) changes epidemic behavior.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let seedButton, resetButton, netSelect, betaSlider, gammaSlider;

const NN = 60;
let adj = [], deg = [], pos = [];
let state = [];        // 0=S, 1=I, 2=R
let tInfected = [];    // sim time when infected
let beta = 0.1, gamma = 0.05;
let running = false;
let simTime = 0;
const DT = 0.1;
let netHist = [];      // {t,S,I,R} fractions (network)
let odeHist = [];      // precomputed ODE curve
let hoverNode = -1;
let rng = mulberry32(12345);

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  seedButton = createButton('Seed (infect 2)');
  seedButton.position(10, drawHeight + 8);
  seedButton.mousePressed(seedInfection);

  resetButton = createButton('Reset');
  resetButton.position(132, drawHeight + 8);
  resetButton.mousePressed(resetSim);

  netSelect = createSelect();
  netSelect.position(196, drawHeight + 8);
  netSelect.option('Heterogeneous (hubs)');
  netSelect.option('Erdős–Rényi (homogeneous)');
  netSelect.option('Star (extreme hub)');
  netSelect.changed(() => { buildNetwork(); resetSim(); });

  betaSlider = createSlider(0.01, 0.5, 0.1, 0.01);
  betaSlider.position(sliderLeftMargin, drawHeight + 45);
  betaSlider.input(() => { beta = betaSlider.value(); precomputeODE(); });

  gammaSlider = createSlider(0.01, 0.3, 0.05, 0.01);
  gammaSlider.position(sliderLeftMargin, drawHeight + 80);
  gammaSlider.input(() => { gamma = gammaSlider.value(); precomputeODE(); });
  sizeSliders();

  buildNetwork();
  resetSim();
  describe('Side-by-side SIR epidemic: a network simulation (left) and the classical ' +
    'well-mixed ODE (right). Adjust transmission β and recovery γ, pick a network ' +
    'structure, seed two infections, and compare how network heterogeneity changes the ' +
    'epidemic curve versus the homogeneous-mixing prediction.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  // simulation tick
  if (running) { simStep(); }

  let half = canvasWidth / 2;
  // titles
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(16);
  text('Network spread', half / 2, 6);
  text('S / I / R over time', half + half / 2, 6);

  // R0
  let r0 = beta / gamma;
  textSize(14); textAlign(CENTER, TOP);
  fill(r0 > 1 ? '#c62828' : '#2e7d32');
  text('R₀ = β/γ = ' + nf(r0, 1, 2), half / 2, 26);

  stroke('silver'); strokeWeight(1); line(half, 46, half, drawHeight - 8);

  drawNetwork(0, 48, half, drawHeight - 48 - 8);
  drawTimeSeries(half + 8, 48, half - 8 - margin, drawHeight - 48 - 8);
  if (hoverNode >= 0) drawTooltip();

  drawControlLabels();
}

// ---------- network rendering ----------

function nodeScreen(i, gx, gy, gw, gh) { return { x: gx + 18 + pos[i].x * (gw - 36), y: gy + 14 + pos[i].y * (gh - 28) }; }

function drawNetwork(gx, gy, gw, gh) {
  hoverNode = -1;
  stroke(220); strokeWeight(0.7);
  for (let i = 0; i < NN; i++) for (let j of adj[i]) if (i < j) {
    let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh);
    line(a.x, a.y, b.x, b.y);
  }
  let maxDeg = Math.max(...deg, 1);
  for (let i = 0; i < NN; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    let rad = map(deg[i], 1, maxDeg, 4, 12);
    if (dist(mouseX, mouseY, p.x, p.y) <= rad + 2) hoverNode = i;
    let col = state[i] === 0 ? '#1565c0' : state[i] === 1 ? '#d32f2f' : '#2e7d32';
    noStroke(); fill(col); circle(p.x, p.y, rad * 2);
  }
  // legend
  noStroke(); textAlign(LEFT, CENTER); textSize(11);
  fill('#1565c0'); text('● S', gx + 4, gy + gh - 6);
  fill('#d32f2f'); text('● I', gx + 36, gy + gh - 6);
  fill('#2e7d32'); text('● R', gx + 64, gy + gh - 6);
}

function drawTimeSeries(px, py, pw, ph) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, py, pw, ph);
  let x0 = px + 4, x1 = px + pw - 4, y0 = py + ph - 16, y1 = py + 6;
  let xFor = t => map(constrain(t, 0, 200), 0, 200, x0, x1);
  let yFor = v => map(constrain(v, 0, 1), 0, 1, y0, y1);

  // axes
  stroke(200); strokeWeight(1); line(x0, y0, x1, y0); line(x0, y0, x0, y1);
  noStroke(); fill('dimgray'); textSize(10); textAlign(LEFT, TOP);
  text('0', x0, y0 + 2); textAlign(RIGHT, TOP); text('t=200', x1, y0 + 2);
  textAlign(LEFT, CENTER); text('1.0', x0 + 2, y1 + 2); text('frac', x0 + 2, (y0 + y1) / 2);

  // ODE dashed
  drawingContext.setLineDash([4, 4]);
  plotSeries(odeHist, 'S', '#1565c0', xFor, yFor);
  plotSeries(odeHist, 'I', '#d32f2f', xFor, yFor);
  plotSeries(odeHist, 'R', '#2e7d32', xFor, yFor);
  drawingContext.setLineDash([]);
  // network solid
  plotSeries(netHist, 'S', '#1565c0', xFor, yFor);
  plotSeries(netHist, 'I', '#d32f2f', xFor, yFor);
  plotSeries(netHist, 'R', '#2e7d32', xFor, yFor);

  // current-time marker
  if (netHist.length) {
    stroke('#999'); strokeWeight(1); drawingContext.setLineDash([2, 3]);
    let xx = xFor(simTime); line(xx, y1, xx, y0); drawingContext.setLineDash([]);
  }
  // legend
  noStroke(); textAlign(RIGHT, TOP); textSize(10); fill('dimgray');
  text('dashed = ODE   solid = network', x1, y1 - 0);
}

function plotSeries(hist, key, col, xFor, yFor) {
  if (hist.length < 2) return;
  stroke(col); strokeWeight(2); noFill();
  beginShape();
  for (let h of hist) vertex(xFor(h.t), yFor(h[key]));
  endShape();
}

function drawTooltip() {
  let i = hoverNode;
  let st = state[i] === 0 ? 'S' : state[i] === 1 ? 'I' : 'R';
  let lines = ['Node ' + i, 'State: ' + st, 'Degree: ' + deg[i],
    'Infected at: ' + (tInfected[i] >= 0 ? 't=' + nf(tInfected[i], 1, 1) : '—')];
  let w = 150, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 50, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

function drawControlLabels() {
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('β = ' + nf(beta, 1, 2), 10, drawHeight + 52);
  text('γ = ' + nf(gamma, 1, 2), 10, drawHeight + 87);
}

// ---------- SIR simulation ----------

function resetSim() {
  running = false;
  state = new Array(NN).fill(0);
  tInfected = new Array(NN).fill(-1);
  simTime = 0;
  netHist = [{ t: 0, S: 1, I: 0, R: 0 }];
  precomputeODE();
}

function seedInfection() {
  resetSim();
  let a = Math.floor(rng() * NN), b = Math.floor(rng() * NN);
  while (b === a) b = Math.floor(rng() * NN);
  state[a] = 1; state[b] = 1; tInfected[a] = 0; tInfected[b] = 0;
  recordNet();
  running = true;
}

function simStep() {
  let pInf = 1 - Math.exp(-beta * DT);
  let pRec = 1 - Math.exp(-gamma * DT);
  let newInfections = [];
  let recoveries = [];
  for (let i = 0; i < NN; i++) {
    if (state[i] !== 1) continue;
    for (let j of adj[i]) if (state[j] === 0 && rng() < pInf) newInfections.push(j);
    if (rng() < pRec) recoveries.push(i);
  }
  for (let j of newInfections) if (state[j] === 0) { state[j] = 1; tInfected[j] = simTime; }
  for (let i of recoveries) state[i] = 2;
  simTime += DT;
  recordNet();
  // stop when no infected
  if (!state.includes(1) || simTime > 200) running = false;
}

function recordNet() {
  let s = 0, inf = 0, r = 0;
  for (let st of state) { if (st === 0) s++; else if (st === 1) inf++; else r++; }
  netHist.push({ t: simTime, S: s / NN, I: inf / NN, R: r / NN });
}

function precomputeODE() {
  // well-mixed SIR, Euler dt=0.01, S(0)=N-2, I0=2
  odeHist = [];
  let S = (NN - 2) / NN, I = 2 / NN, R = 0;
  let kbar = deg.reduce((a, b) => a + b, 0) / NN;   // contacts/node
  let dt = 0.01;
  for (let t = 0; t <= 200; t += dt) {
    if (t % 0.5 < dt) odeHist.push({ t, S, I, R });
    let inf = beta * kbar * S * I;     // mass-action with mean degree
    let rec = gamma * I;
    S += (-inf) * dt; I += (inf - rec) * dt; R += rec * dt;
    S = Math.max(0, S); I = Math.max(0, I);
  }
}

// ---------- networks (deterministic) ----------

function buildNetwork() {
  let name = netSelect ? netSelect.value() : 'Heterogeneous (hubs)';
  rng = mulberry32(99);
  adj = Array.from({ length: NN }, () => []);
  let edgeSet = new Set();
  const add = (a, b) => {
    if (a === b) return; let k = Math.min(a, b) + '-' + Math.max(a, b);
    if (edgeSet.has(k)) return; edgeSet.add(k); adj[a].push(b); adj[b].push(a);
  };

  if (name.startsWith('Star')) {
    for (let i = 1; i < NN; i++) add(0, i);
  } else if (name.startsWith('Erd')) {
    // ER with mean degree ~4 → p = 4/(N-1)
    let p = 4 / (NN - 1);
    for (let i = 0; i < NN; i++) for (let j = i + 1; j < NN; j++) if (rng() < p) add(i, j);
  } else {
    // Barabási–Albert preferential attachment, m=2 → heterogeneous, mean deg ~4
    let targets = [0, 1, 2];
    add(0, 1); add(1, 2); add(0, 2);
    let repeated = [0, 1, 2];
    for (let v = 3; v < NN; v++) {
      let chosen = new Set();
      while (chosen.size < 2) chosen.add(repeated[Math.floor(rng() * repeated.length)]);
      for (let t of chosen) { add(v, t); repeated.push(t); repeated.push(v); }
    }
  }
  deg = adj.map(a => a.length);
  pos = springLayout();
}

function springLayout() {
  let p = [];
  for (let i = 0; i < NN; i++) { let a = TWO_PI * i / NN; p.push({ x: 0.5 + 0.42 * Math.cos(a), y: 0.5 + 0.42 * Math.sin(a) }); }
  for (let it = 0; it < 180; it++) {
    let fx = new Array(NN).fill(0), fy = new Array(NN).fill(0);
    for (let i = 0; i < NN; i++) for (let j = i + 1; j < NN; j++) {
      let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2);
      let rep = 0.0004 / d2;
      fx[i] += dx / d * rep; fy[i] += dy / d * rep; fx[j] -= dx / d * rep; fy[j] -= dy / d * rep;
    }
    for (let i = 0; i < NN; i++) for (let j of adj[i]) if (i < j) {
      let dx = p[j].x - p[i].x, dy = p[j].y - p[i].y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4;
      let spr = (d - 0.1) * 0.015;
      fx[i] += dx / d * spr; fy[i] += dy / d * spr; fx[j] -= dx / d * spr; fy[j] -= dy / d * spr;
    }
    for (let i = 0; i < NN; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y);
  let mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ---------- responsive ----------

function sizeSliders() {
  let w = canvasWidth - sliderLeftMargin - margin;
  [betaSlider, gammaSlider].forEach(s => s && s.size(w));
}
function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  sizeSliders();
  redraw();
}
function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
