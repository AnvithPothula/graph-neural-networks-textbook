// Drug Discovery GNN Pipeline (3 stages)
// CANVAS_HEIGHT: 520
// Learning objective (Understand): how GNNs contribute across a three-stage drug
// discovery workflow — node classification (target ID), generation (hit generation),
// and link prediction (safety screening).
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 470;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let stepButton, resetButton;
let activeStage = -1, pulse = 0, hoverTip = null;
let ppiNodes = [], ppiEdges = [], ddiNodes = [], ddiEdges = [];

const STAGES = [
  { title: 'Stage 1 — Target Identification', tint: '#e3f2fd', label: 'GNN Node Classification on PPI Network', cap: 'Identifies high-centrality disease-target proteins' },
  { title: 'Stage 2 — Hit Generation', tint: '#e8f5e9', label: 'GCPN / DiGress — Conditional Molecule Generation', cap: 'Generates candidate molecules maximizing predicted binding affinity' },
  { title: 'Stage 3 — Safety Screening', tint: '#fff8e1', label: 'GNN Link Prediction on DDI Network', cap: 'Flags dangerous drug-drug interactions before clinical testing' },
];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildNetworks();

  stepButton = createButton('Step Through Pipeline'); stepButton.position(10, drawHeight + 8); stepButton.mousePressed(() => { activeStage = (activeStage + 1) % 3; });
  resetButton = createButton('Reset'); resetButton.position(180, drawHeight + 8); resetButton.mousePressed(() => activeStage = -1);

  describe('A three-stage GNN drug-discovery pipeline. Stage 1 uses node classification on ' +
    'a protein-protein interaction network to identify disease targets; Stage 2 uses a ' +
    'generative model to design candidate molecules; Stage 3 uses link prediction on a ' +
    'drug-drug interaction network to flag dangerous interactions. Step through the stages ' +
    'and hover the highlighted elements for details.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();
  pulse += 0.08;

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18); text('Drug Discovery GNN Pipeline', canvasWidth / 2, 6);

  let ph = 122, gap = 30, top = 34, pw = canvasWidth - 2 * margin;
  hoverTip = null;
  for (let s = 0; s < 3; s++) {
    let y = top + s * (ph + gap);
    drawPanel(s, margin, y, pw, ph);
    if (s < 2) drawArrow(canvasWidth / 2, y + ph, canvasWidth / 2, y + ph + gap, s);
  }
  if (hoverTip) drawTip();
}

function drawPanel(s, x, y, w, h) {
  let active = activeStage === s;
  stroke(active ? '#1565c0' : '#cfd8dc'); strokeWeight(active ? 3 : 1);
  fill(STAGES[s].tint); rect(x, y, w, h, 10);
  // stage label
  noStroke(); fill('#333'); textAlign(LEFT, TOP); textSize(13); textStyle(BOLD); text(STAGES[s].title, x + 12, y + 8); textStyle(NORMAL);
  // description text is width-bounded to the left ~45% so it wraps instead of running under the viz
  fill('#555'); textSize(11.5); text(STAGES[s].label, x + 12, y + 28, w * 0.45, 34);
  fill('#777'); textSize(10.5); text(STAGES[s].cap, x + 12, y + 64, w * 0.45, 48);
  // visualization on the right side of the panel
  let vx = x + w * 0.52, vy = y + 8, vw = w * 0.46, vh = h - 16;
  if (s === 0) drawPPI(vx, vy, vw, vh, active);
  else if (s === 1) drawMolecule(vx, vy, vw, vh, active);
  else drawDDI(vx, vy, vw, vh, active);
}

function panelNode(n, vx, vy, vw, vh) { return { x: vx + n.x * vw, y: vy + n.y * vh }; }

function drawPPI(vx, vy, vw, vh, active) {
  stroke(170); strokeWeight(1.2);
  for (let e of ppiEdges) { let a = panelNode(ppiNodes[e[0]], vx, vy, vw, vh), b = panelNode(ppiNodes[e[1]], vx, vy, vw, vh); line(a.x, a.y, b.x, b.y); }
  for (let i = 0; i < ppiNodes.length; i++) {
    let p = panelNode(ppiNodes[i], vx, vy, vw, vh), disease = ppiNodes[i].disease;
    if (dist(mouseX, mouseY, p.x, p.y) <= 9 && disease) hoverTip = { lines: [ppiNodes[i].name, ppiNodes[i].prob + '% disease-pathway probability', 'betweenness centrality: ' + ppiNodes[i].btw] };
    stroke(40); strokeWeight(1); fill(disease ? '#e53935' : '#90caf9'); circle(p.x, p.y, disease ? 14 : 11);
  }
}

function drawMolecule(vx, vy, vw, vh, active) {
  // generative box -> benzene ring
  noStroke(); fill('#a5d6a7'); stroke('#66bb6a'); rect(vx, vy + vh / 2 - 16, 90, 32, 5);
  noStroke(); fill('#1b5e20'); textAlign(CENTER, CENTER); textSize(10); text('GCPN /\nDiGress', vx + 45, vy + vh / 2);
  stroke('#2e7d32'); strokeWeight(1.5); drawArrowSeg(vx + 92, vy + vh / 2, vx + 120, vy + vh / 2, 6);
  // benzene ring
  let cx = vx + vw - 60, cy = vy + vh / 2, r = Math.min(vw * 0.18, 32);
  let ring = [];
  for (let i = 0; i < 6; i++) { let a = TWO_PI * i / 6 - HALF_PI; ring.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) }); }
  stroke('#333'); strokeWeight(2);
  for (let i = 0; i < 6; i++) line(ring[i].x, ring[i].y, ring[(i + 1) % 6].x, ring[(i + 1) % 6].y);
  for (let i = 0; i < 6; i++) { noStroke(); fill(i % 2 ? '#ff7043' : '#5c6bc0'); circle(ring[i].x, ring[i].y, 11); }
  if (dist(mouseX, mouseY, cx, cy) <= r + 6) hoverTip = { lines: ['Candidate molecule', 'QED: 0.84   SA: 2.1', 'Predicted IC50: 23 nM'] };
}

function drawDDI(vx, vy, vw, vh, active) {
  stroke('#ef9a9a'); strokeWeight(1.4);
  for (let e of ddiEdges) { let a = panelNode(ddiNodes[e[0]], vx, vy, vw, vh), b = panelNode(ddiNodes[e[1]], vx, vy, vw, vh); stroke(e[2] ? '#e53935' : 180); strokeWeight(e[2] ? 2 : 1); line(a.x, a.y, b.x, b.y); if (e[2] && distToSeg(mouseX, mouseY, a.x, a.y, b.x, b.y) < 5) hoverTip = { lines: ['Predicted interaction:', 'candidate + warfarin →', 'bleeding risk (conf 0.88)'] }; }
  for (let i = 0; i < ddiNodes.length; i++) {
    let p = panelNode(ddiNodes[i], vx, vy, vw, vh), cand = ddiNodes[i].cand;
    stroke(40); strokeWeight(1);
    fill(cand ? color(255, 140 + 80 * Math.sin(pulse), 0) : '#b0bec5'); circle(p.x, p.y, cand ? 14 : 10);
  }
}

function drawArrow(x1, y1, x2, y2, s) {
  let active = activeStage === s + 1 || activeStage === s;
  stroke(active ? '#1565c0' : '#90a4ae'); strokeWeight(2);
  drawArrowSeg(x1, y1, x2, y2, 6);
  if (dist(mouseX, mouseY, (x1 + x2) / 2, (y1 + y2) / 2) < 10) hoverTip = { lines: ['Stage ' + (s + 1) + ' → ' + (s + 2), s === 0 ? 'target identity + binding pocket' : 'candidate molecule structure', s === 0 ? 'pass as conditioning signal' : 'passes to safety screening'] };
}
function drawArrowSeg(x1, y1, x2, y2, rad) { let ang = atan2(y2 - y1, x2 - x1), ex = x2 - cos(ang) * rad, ey = y2 - sin(ang) * rad; line(x1, y1, ex, ey); push(); translate(ex, ey); rotate(ang); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -7, 2.5, -7, -2.5); pop(); }

function drawTip() {
  let w = 230, h = hoverTip.lines.length * 16 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < hoverTip.lines.length; k++) { if (k === 0) textStyle(BOLD); else textStyle(NORMAL); text(hoverTip.lines[k], tx + 8, ty + 6 + k * 16); }
  textStyle(NORMAL);
}

function distToSeg(px, py, x1, y1, x2, y2) { let dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy; let t = l2 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l2)) : 0; return dist(px, py, x1 + t * dx, y1 + t * dy); }

function buildNetworks() {
  let rng = mulberry32(7);
  ppiNodes = []; let names = ['EGFR', 'TP53', 'KRAS', 'MYC', 'PTEN', 'AKT1', 'BRAF', 'PIK3CA', 'MTOR', 'VEGFA'];
  for (let i = 0; i < 10; i++) { ppiNodes.push({ x: 0.1 + rng() * 0.8, y: 0.1 + rng() * 0.8, disease: i < 3, name: names[i], prob: 88 + Math.floor(rng() * 10), btw: nf(0.4 + rng() * 0.5, 1, 2) }); }
  ppiEdges = []; for (let i = 0; i < 10; i++) for (let j = i + 1; j < 10; j++) if (rng() < 0.28) ppiEdges.push([i, j]);
  ddiNodes = []; for (let i = 0; i < 8; i++) ddiNodes.push({ x: 0.12 + rng() * 0.76, y: 0.12 + rng() * 0.76, cand: i === 0 });
  ddiEdges = []; for (let i = 1; i < 8; i++) for (let j = i + 1; j < 8; j++) if (rng() < 0.25) ddiEdges.push([i, j, 0]);
  // candidate (node 0) interactions in red
  for (let j = 1; j < 8; j++) if (rng() < 0.4) ddiEdges.push([0, j, 1]);
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
