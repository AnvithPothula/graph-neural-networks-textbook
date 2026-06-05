// Self-Supervised Learning Concept Map (DGI vs Graph Contrastive Learning)
// CANVAS_HEIGHT: 480
// Learning objective (Analyze): compare DGI and graph contrastive learning by exploring
// how each satisfies the three core SSL properties, via an interactive concept map.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 420;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let resetButton;
let nodes = [], edges = [], sel = -1, hoverN = -1, pulse = 0;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildMap();
  resetButton = createButton('Reset'); resetButton.position(10, drawHeight + 12); resetButton.mousePressed(() => sel = -1);
  describe('A concept map comparing two self-supervised graph learning methods, DGI and ' +
    'graph contrastive learning, against the three core SSL properties (invariance, ' +
    'discrimination, structural context) and the mechanisms each uses. Click a method to ' +
    'highlight the properties it satisfies and its mechanisms; click any box for a definition.', LABEL);
}

function draw() {
  updateCanvasSize();
  // radial gradient-ish background
  fill('white'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke(); pulse += 0.06;

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18); text('Self-Supervised Learning: DGI vs Contrastive', canvasWidth / 2, 6);
  // tier labels
  fill('#999'); textSize(11); textAlign(LEFT, CENTER);
  text('Core SSL properties', 12, 60); text('Methods', 12, drawHeight * 0.5); text('Mechanisms', 12, drawHeight - 40);

  let hl = highlightSet();
  hoverN = -1;
  // edges
  for (let e of edges) {
    let a = ptOf(e[0]), b = ptOf(e[1]);
    let on = hl && hl.has(e[0]) && hl.has(e[1]);
    if (on) { stroke(e[2] === 'partial' ? '#e65100' : '#2e7d32'); strokeWeight(2.5); }
    else { stroke(sel >= 0 ? color(200, 200, 200, 60) : color(180)); strokeWeight(1); }
    if (on) { // animated pulse dot
      drawingContext.setLineDash([6, 6]); drawingContext.lineDashOffset = -pulse * 12;
    }
    line(a.x, a.y, b.x, b.y); drawingContext.setLineDash([]);
    if (on && e[2] === 'partial') { noStroke(); fill('#e65100'); textAlign(CENTER, CENTER); textSize(9); text('partial', (a.x + b.x) / 2, (a.y + b.y) / 2); }
  }
  // nodes
  for (let i = 0; i < nodes.length; i++) {
    let n = nodes[i], p = ptOf(i);
    let w = n.w, h = 30;
    if (mouseX > p.x - w / 2 && mouseX < p.x + w / 2 && mouseY > p.y - h / 2 && mouseY < p.y + h / 2) hoverN = i;
    let dim = hl && !hl.has(i);
    push(); if (dim) drawingContext.globalAlpha = 0.22;
    stroke(i === sel ? 'black' : '#888'); strokeWeight(i === sel ? 3 : 1); fill(n.col);
    if (n.tier === 1) ellipse(p.x, p.y, w, h + 6); else rect(p.x - w / 2, p.y - h / 2, w, h, 8);
    noStroke(); fill('#222'); textAlign(CENTER, CENTER); textSize(n.tier === 1 ? 13 : 10.5); text(n.label, p.x, p.y);
    pop();
  }
  if (hoverN >= 0) drawTip(hoverN);
}

function ptOf(i) { let n = nodes[i]; return { x: margin + 60 + n.x * (canvasWidth - 2 * margin - 120), y: n.tier === 0 ? 70 : n.tier === 1 ? drawHeight * 0.5 : drawHeight - 50 }; }

function highlightSet() {
  if (sel < 0) return null;
  let s = new Set([sel]); let n = nodes[sel];
  if (n.tier === 1) { for (let e of edges) if (e[0] === sel || e[1] === sel) { s.add(e[0]); s.add(e[1]); } }
  else if (n.tier === 0) { for (let e of edges) if (e[1] === sel && nodes[e[0]].tier === 1) { s.add(e[0]); } }
  else { for (let e of edges) if (e[0] === sel) s.add(e[1]); }
  return s;
}

function drawTip(i) {
  let n = nodes[i];
  let lines = [n.label, n.desc];
  let w = 250, h = 16 + Math.ceil(textW(n.desc, w - 16) ) ;
  let nl = wrapLines(n.desc, w - 16, 11); h = (nl.length + 1) * 15 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 250); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12); textStyle(BOLD); text(n.label, tx + 8, ty + 6); textStyle(NORMAL); textSize(11);
  for (let k = 0; k < nl.length; k++) text(nl[k], tx + 8, ty + 24 + k * 15);
}
function textW(s, w) { return 1; }
function wrapLines(s, w, sz) { textSize(sz); let words = s.split(' '), line = '', out = []; for (let wd of words) { if (textWidth(line + wd + ' ') > w) { out.push(line); line = wd + ' '; } else line += wd + ' '; } out.push(line); return out; }

function buildMap() {
  // tier 0 = properties, 1 = methods, 2 = mechanisms
  nodes = [
    { label: 'Invariance to\nAugmentation', tier: 0, x: 0.16, w: 130, col: '#bbdefb', desc: 'Representations should be stable under semantics-preserving graph augmentations (edge dropout, feature masking).' },
    { label: 'Discrimination\nAcross Nodes', tier: 0, x: 0.5, w: 130, col: '#c8e6c9', desc: 'Different nodes (or graphs) should map to distinguishable representations, not collapse to one point.' },
    { label: 'Structural\nContext', tier: 0, x: 0.84, w: 130, col: '#ffe0b2', desc: 'Representations should encode each node\'s graph neighborhood and global position.' },
    { label: 'DGI', tier: 1, x: 0.3, w: 90, col: '#d1c4e9', desc: 'Deep Graph Infomax: maximizes mutual information between local patch embeddings and a global summary, using a corrupted graph as negatives.' },
    { label: 'Graph Contrastive\nLearning', tier: 1, x: 0.7, w: 130, col: '#b2dfdb', desc: 'Creates two augmented views and pulls matching node embeddings together while pushing non-matching ones apart (NT-Xent loss).' },
    { label: 'Graph\nCorruption', tier: 2, x: 0.1, w: 100, col: '#ede7f6', desc: 'Row-shuffle node features to build a negative (fake) graph for the discriminator.' },
    { label: 'Mutual Info\n(JSD)', tier: 2, x: 0.3, w: 100, col: '#ede7f6', desc: 'Jensen-Shannon mutual-information estimator separates real patch-summary pairs from corrupted ones.' },
    { label: 'Edge Dropout /\nFeat. Masking', tier: 2, x: 0.55, w: 110, col: '#e0f2f1', desc: 'Stochastic augmentations that produce two correlated but different views of the same graph.' },
    { label: 'NT-Xent\nLoss', tier: 2, x: 0.74, w: 90, col: '#e0f2f1', desc: 'Normalized temperature-scaled cross-entropy: softmax over cosine similarities with a temperature τ.' },
    { label: 'Two-View\nEncoder', tier: 2, x: 0.92, w: 90, col: '#e0f2f1', desc: 'A shared GNN encodes both augmented views into the same embedding space.' },
  ];
  // edges: method->property ; mechanism->method
  edges = [
    [3, 1, 'full'], [3, 2, 'full'], [3, 0, 'partial'],   // DGI
    [4, 0, 'full'], [4, 1, 'full'], [4, 2, 'full'],       // GCL
    [5, 3, ''], [6, 3, ''],                               // mechanisms->DGI
    [7, 4, ''], [8, 4, ''], [9, 4, ''],                   // mechanisms->GCL
  ];
}

function mousePressed() {
  if (mouseY > drawHeight) return;
  for (let i = 0; i < nodes.length; i++) { let p = ptOf(i), w = nodes[i].w, h = 34; if (mouseX > p.x - w / 2 && mouseX < p.x + w / 2 && mouseY > p.y - h / 2 && mouseY < p.y + h / 2) { sel = (sel === i ? -1 : i); return; } }
  sel = -1;
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
