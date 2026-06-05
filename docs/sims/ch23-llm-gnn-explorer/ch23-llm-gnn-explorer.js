// LLM + GNN Explorer — Encoder / Reasoner / Joint modes
// CANVAS_HEIGHT: 560
// Learning objective (Evaluate/Apply): trace text-attributed features through the
// LLM+GNN pipeline three ways — LLM-as-encoder, LLM-as-reasoner, and joint training —
// and weigh what each contributes.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 960;
let drawHeight = 480;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let encBtn, reaBtn, joiBtn, nodeSelect, actionBtn, nbToggle;
let mode = 'enc', encStage = 4, showNeighbors = true, hoverTip = null;
let backpropT = -1, sendT = -1;
const DIM = 32;
let nodes = [];

const CATS = ['cs.CV', 'cs.CL', 'cs.LG', 'cs.SI', 'stat.ML'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildNodes();

  encBtn = createButton('Encoder Mode'); encBtn.position(10, drawHeight + 8); encBtn.mousePressed(() => mode = 'enc');
  reaBtn = createButton('Reasoner Mode'); reaBtn.position(118, drawHeight + 8); reaBtn.mousePressed(() => { mode = 'rea'; sendT = -1; });
  joiBtn = createButton('Joint Mode'); joiBtn.position(232, drawHeight + 8); joiBtn.mousePressed(() => { mode = 'joi'; backpropT = -1; });
  nodeSelect = createSelect(); nodeSelect.position(372, drawHeight + 8); nodes.forEach((n, i) => nodeSelect.option(n.title, i)); nodeSelect.changed(() => { encStage = 4; });
  nbToggle = createCheckbox(' Show neighbors', true); nbToggle.position(10, drawHeight + 44); nbToggle.changed(() => showNeighbors = nbToggle.checked());
  actionBtn = createButton('action'); actionBtn.position(160, drawHeight + 44); actionBtn.mousePressed(doAction);

  describe('LLM+GNN explorer with three modes. Encoder mode shows text encoded to an ' +
    'embedding heat strip, neighbor strips, GNN aggregation, and classification. Reasoner ' +
    'mode assembles a text prompt for an LLM and shows a mock reasoned answer. Joint mode ' +
    'shows gradients backpropagating through both the GNN and the LLM.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('#f5f5f5'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  highlightTabs();
  styleAction();

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18);
  text('LLM + GNN Explorer — ' + (mode === 'enc' ? 'Encoder' : mode === 'rea' ? 'Reasoner' : 'Joint') + ' Mode', canvasWidth / 2, 6);

  hoverTip = null;
  let nd = nodes[nodeSelect ? nodeSelect.elt.selectedIndex : 0];
  if (mode === 'enc') drawEncoder(nd);
  else if (mode === 'rea') drawReasoner(nd);
  else drawJoint(nd);
  if (hoverTip) drawTip();

  fill('#555'); noStroke(); textAlign(LEFT, CENTER); textSize(12); text('node:', 334, drawHeight + 18 + 0);
}

// ---------- encoder mode ----------

function card(x, y, w, h, title, col, on) {
  push(); if (!on) drawingContext.globalAlpha = 0.25;
  stroke('#bbb'); strokeWeight(1); fill('white'); rect(x, y, w, h, 8);
  noStroke(); fill(col); rect(x, y, w, 20, 8, 8, 0, 0); fill('white'); textAlign(CENTER, CENTER); textSize(11); text(title, x + w / 2, y + 10);
  pop();
}
function heatStrip(vec, x, y, w, h) {
  let n = vec.length, cw = w / n;
  for (let i = 0; i < n; i++) { let v = vec[i]; let c = v >= 0 ? lerpColor(color('#fff'), color('#d32f2f'), v) : lerpColor(color('#fff'), color('#1565c0'), -v); fill(c); stroke('#eee'); strokeWeight(0.4); rect(x + i * cw, y, cw, h); }
  noStroke();
}

function drawEncoder(nd) {
  let n = 5, gap = 12, x0 = margin, top = 40, cw = (canvasWidth - 2 * margin - (n - 1) * gap) / n, ch = drawHeight - top - 20;
  let titles = ['Node Text', 'LLM Encode', 'Neighbor Embeddings', 'GNN Aggregate', 'Classification'];
  let cols = ['#1565c0', '#fb8c00', '#42a5f5', '#2e7d32', '#6a1b9a'];
  for (let s = 0; s < n; s++) {
    let x = x0 + s * (cw + gap), on = s <= encStage;
    if (s > 0 && on) { stroke('#888'); strokeWeight(2); drawArrowSeg(x - gap, top + ch / 2, x, top + ch / 2, 4); }
    card(x, top, cw, ch, titles[s], cols[s], on);
    if (!on) continue;
    push();
    if (s === 0) { fill('#333'); textAlign(LEFT, TOP); textSize(9.5); text(nd.text, x + 6, top + 26, cw - 12, ch - 32); }
    else if (s === 1) { heatStrip(nd.emb, x + 6, top + 40, cw - 12, 20); noStroke(); fill('#777'); textSize(9); textAlign(LEFT, TOP); text('32 of 384 dims', x + 6, top + 64); }
    else if (s === 2) { for (let k = 0; k < 3; k++) { heatStrip(nd.nbEmb[k], x + 6, top + 36 + k * 26, cw - 12, 16); } }
    else if (s === 3) { heatStrip(nd.agg, x + 6, top + 44, cw - 12, 22); for (let k = 0; k < 3; k++) { stroke('#42a5f5'); strokeWeight(1.5); drawArrowSeg(x + cw / 2 - 16 + k * 16, top + 30, x + cw / 2, top + 42, 3); } }
    else { drawBars(nd, x + 6, top + 28, cw - 12, ch - 40); }
    pop();
    if (mouseX > x && mouseX < x + cw && mouseY > top && mouseY < top + ch) hoverTip = encTip(s);
  }
}
function drawBars(nd, x, y, w, h) {
  for (let i = 0; i < nd.probs.length; i++) { let yy = y + i * 24; let p = nd.probs[i]; fill(i === nd.label ? '#f9a825' : '#90caf9'); stroke('#999'); strokeWeight(0.5); rect(x, yy, w * p, 15, 2); noStroke(); fill('#222'); textAlign(LEFT, CENTER); textSize(9); text(CATS[i] + ' ' + Math.round(p * 100) + '%', x + 2, yy + 7); }
}
function encTip(s) { let t = [['Node text', 'the abstract attribute of the node'], ['LLM encode', 'text → 384-dim embedding (heat strip shows 32 dims)'], ['Neighbor embeddings', 'each neighbor encoded the same way'], ['GNN aggregate', 'mean of node + neighbor embeddings'], ['Classification', 'linear head → category probabilities']][s]; return { lines: t }; }

// ---------- reasoner mode ----------

function drawReasoner(nd) {
  let x = margin, y = 44, w = canvasWidth - 2 * margin;
  // prompt card
  stroke('#bbb'); fill('white'); rect(x, y, w, 150, 8); noStroke(); fill('#1565c0'); rect(x, y, w, 22, 8, 8, 0, 0); fill('white'); textAlign(LEFT, CENTER); textSize(12); text('Assembled prompt', x + 8, y + 11);
  fill('#333'); textAlign(LEFT, TOP); textSize(12);
  let prompt = 'Node abstract: "' + nd.text + '"\n\nNeighbors: ' + nd.nbTitles.join('; ') + '\n\nPredict the arXiv category for this paper.';
  text(prompt, x + 10, y + 28, w - 20, 116);
  // response
  let ry = y + 168;
  if (sendT < 0) { noStroke(); fill('#777'); textAlign(CENTER, CENTER); textSize(13); text('Click "Send to LLM" to generate a reasoned prediction', x + w / 2, ry + 60); }
  else {
    let t = constrain((millis() - sendT) / 1200, 0, 1);
    // speech bubble
    stroke('#66bb6a'); strokeWeight(1.5); fill('#e8f5e9'); rect(x, ry, w, drawHeight - ry - 16, 10);
    noStroke(); fill('#1b5e20'); textAlign(LEFT, TOP); textSize(12); textStyle(BOLD); text('LLM response:', x + 10, ry + 8); textStyle(NORMAL);
    let resp = 'The abstract emphasizes ' + nd.keyphrase + ', and its neighbors are largely ' + CATS[nd.label] + ' papers. Combining textual evidence with neighborhood context, the most likely category is:';
    let shown = resp.substring(0, Math.floor(resp.length * t));
    fill('#333'); text(shown, x + 10, ry + 28, w - 20, 70);
    if (t >= 1) { fill('#f9a825'); stroke('#b8860b'); rect(x + 10, ry + drawHeight - ry - 16 - 40, 120, 26, 5); noStroke(); fill('#5d4037'); textAlign(CENTER, CENTER); textSize(14); textStyle(BOLD); text(CATS[nd.label], x + 70, ry + drawHeight - ry - 16 - 27); textStyle(NORMAL); }
  }
}

// ---------- joint mode ----------

function drawJoint(nd) {
  let cy = drawHeight / 2 - 20;
  // responsive: 4 evenly spaced boxes across the canvas width (was hardcoded for 960px)
  let labels = ['Text', 'LLM', 'GNN', 'Loss vs label'], cols = ['#1565c0', '#fb8c00', '#2e7d32', '#6a1b9a'];
  let bw = 90, edge = margin + bw / 2 + 30;
  let boxes = labels.map((label, i) => ({ x: edge + (canvasWidth - 2 * edge) * i / (labels.length - 1), label, col: cols[i] }));
  // forward arrows
  for (let i = 0; i < boxes.length - 1; i++) { stroke('#888'); strokeWeight(2); drawArrowSeg(boxes[i].x + 45, cy, boxes[i + 1].x - 45, cy, 6); }
  let bp = backpropT >= 0 ? constrain((millis() - backpropT) / 1800, 0, 1) : -1;
  // backward (gradient) arrows
  if (bp >= 0) {
    for (let i = boxes.length - 1; i > 0; i--) {
      let frac = (boxes.length - 1 - i) / (boxes.length - 1);
      if (bp > frac) { let g = constrain((bp - frac) * 3, 0, 1); stroke(lerpColor(color('#ffcdd2'), color('#d32f2f'), g)); strokeWeight(2 + g * 3); drawArrowSeg(boxes[i].x - 45, cy + 30, boxes[i - 1].x + 45, cy + 30, 6); }
    }
    noStroke(); fill('#c62828'); textAlign(CENTER, TOP); textSize(12); text('gradients flow back through GNN → LLM (red = high gradient)', canvasWidth / 2, cy + 70);
  }
  for (let b of boxes) {
    let updating = bp >= 0 && (b.label === 'LLM' || b.label === 'GNN');
    stroke(updating ? '#c62828' : '#999'); strokeWeight(updating ? 2.5 : 1); fill(b.col); rect(b.x - 45, cy - 26, 90, 52, 8);
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(13); text(b.label, b.x, cy);
    if (mouseX > b.x - 45 && mouseX < b.x + 45 && mouseY > cy - 26 && mouseY < cy + 26) hoverTip = { lines: [b.label, b.label === 'LLM' ? 'fine-tuned end-to-end with the GNN' : b.label === 'GNN' ? 'aggregates LLM embeddings over the graph' : b.label === 'Text' ? 'raw node text attribute' : 'task loss drives both components'] };
  }
  noStroke(); fill('#555'); textAlign(CENTER, TOP); textSize(13); text('Joint training: text → LLM → GNN → loss; gradients update both LLM and GNN.', canvasWidth / 2, 50);
}

// ---------- shared ----------

function doAction() {
  if (mode === 'enc') encStage = encStage >= 4 ? 0 : encStage + 1;
  else if (mode === 'rea') sendT = millis();
  else backpropT = millis();
}
function styleAction() {
  let label = mode === 'enc' ? (encStage >= 4 ? 'Restart stages' : 'Next stage →') : mode === 'rea' ? 'Send to LLM' : 'Backprop';
  actionBtn.html(label);
}
function highlightTabs() {
  [[encBtn, 'enc'], [reaBtn, 'rea'], [joiBtn, 'joi']].forEach(([b, m]) => { b.style('background-color', mode === m ? '#6a1b9a' : '#eeeeee'); b.style('color', mode === m ? 'white' : 'black'); });
}
function drawArrowSeg(x1, y1, x2, y2, rad) { let ang = atan2(y2 - y1, x2 - x1), ex = x2 - cos(ang) * rad, ey = y2 - sin(ang) * rad; line(x1, y1, ex, ey); push(); translate(ex, ey); rotate(ang); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -6, 2, -6, -2); pop(); }
function drawTip() {
  let w = 260, h = hoverTip.lines.length * 16 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(11.5);
  for (let k = 0; k < hoverTip.lines.length; k++) { if (k === 0) textStyle(BOLD); else textStyle(NORMAL); text(hoverTip.lines[k], tx + 8, ty + 6 + k * 16, w - 16); }
  textStyle(NORMAL);
}

function buildNodes() {
  let rng = mulberry32(7);
  let data = [
    { title: 'Vision paper', label: 0, key: 'convolutional image classification', text: 'A deep residual convolutional network for large-scale image recognition, improving top-1 accuracy on ImageNet via identity shortcut connections.', nbT: ['ResNet variants', 'object detection', 'image segmentation'] },
    { title: 'NLP paper', label: 1, key: 'attention-based translation', text: 'A transformer encoder-decoder for neural machine translation using multi-head self-attention to model long-range token dependencies.', nbT: ['BERT pretraining', 'summarization', 'question answering'] },
    { title: 'Graph ML paper', label: 2, key: 'neighborhood aggregation', text: 'A graph attention network for semi-supervised node classification, weighting neighbor messages by learned attention coefficients.', nbT: ['GCN', 'GraphSAGE', 'node2vec'] },
    { title: 'Network science', label: 3, key: 'community structure', text: 'Detecting communities in large social networks via modularity optimization and analyzing their temporal evolution.', nbT: ['link prediction', 'influence spread', 'centrality'] },
    { title: 'Stats/ML paper', label: 4, key: 'probabilistic inference', text: 'A variational inference framework for latent variable models with amortized encoders and reparameterized gradients.', nbT: ['VAE', 'Bayesian deep learning', 'normalizing flows'] },
  ];
  nodes = data.map((d, idx) => {
    let emb = Array.from({ length: DIM }, () => rng() * 2 - 1);
    let nbEmb = [0, 1, 2].map(() => Array.from({ length: DIM }, () => rng() * 2 - 1));
    let agg = emb.map((v, i) => (v + nbEmb[0][i] + nbEmb[1][i] + nbEmb[2][i]) / 4);
    let probs = new Array(5).fill(0.05); probs[d.label] = 0.6 + rng() * 0.2;
    let s = probs.reduce((a, b) => a + b, 0); probs = probs.map(p => p / s);
    return { title: d.title, text: d.text, label: d.label, keyphrase: d.key, nbTitles: d.nbT, emb, nbEmb, agg, probs };
  });
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
