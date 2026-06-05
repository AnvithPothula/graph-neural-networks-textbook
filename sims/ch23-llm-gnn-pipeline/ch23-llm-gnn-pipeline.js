// LLM + GNN Pipeline (5-stage step-through)
// CANVAS_HEIGHT: 500
// Learning objective (Understand): trace a text-described node through the LLM+GNN
// pipeline — raw text → tokenize → LLM encode → GNN aggregate → predicted label.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 420;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let stepButton, autoButton, resetButton, exSelect;
let stage = 4, autoOn = false, lastStep = 0, hoverTip = null;   // start fully revealed (Reset returns to 0)

const EXAMPLES = [
  { name: 'Computer Vision', text: 'We propose a convolutional architecture with residual connections for large-scale image classification, achieving state-of-the-art top-1 accuracy on ImageNet…', probs: [['cs.CV', 0.81], ['cs.LG', 0.12], ['cs.AI', 0.07]] },
  { name: 'NLP', text: 'A transformer-based sequence-to-sequence model for neural machine translation, using multi-head self-attention to capture long-range dependencies across tokens…', probs: [['cs.CL', 0.78], ['cs.LG', 0.15], ['cs.AI', 0.07]] },
  { name: 'Graph ML', text: 'We introduce a graph neural network that aggregates neighbor representations via attention for semi-supervised node classification on citation networks…', probs: [['cs.LG', 0.72], ['cs.SI', 0.20], ['stat.ML', 0.08]] },
];
const STAGE_NAMES = ['Node Text (Abstract)', 'Tokenize', 'LLM Encoder', 'GNN Aggregation', 'Predicted Label'];
const STAGE_INFO = [
  'The raw text attribute of the node (a paper abstract).',
  'The LLM tokenizer splits text into sub-word tokens.',
  'The LLM encoder maps tokens to a single embedding e_v ∈ ℝ³⁸⁴.',
  'The GNN mixes the node embedding with its neighbors\' embeddings.',
  'A classifier head outputs class probabilities; argmax = label.',
];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  stepButton = createButton('Step →'); stepButton.position(10, drawHeight + 8); stepButton.mousePressed(() => { autoOn = false; autoButton.html('Auto Play'); stage = Math.min(4, stage + 1); });
  autoButton = createButton('Auto Play'); autoButton.position(72, drawHeight + 8); autoButton.mousePressed(() => { autoOn = !autoOn; autoButton.html(autoOn ? 'Pause' : 'Auto Play'); lastStep = 0; });
  resetButton = createButton('Reset'); resetButton.position(160, drawHeight + 8); resetButton.mousePressed(() => { stage = 0; autoOn = false; autoButton.html('Auto Play'); });
  exSelect = createSelect(); exSelect.position(224, drawHeight + 8); EXAMPLES.forEach((e, i) => exSelect.option(e.name, i)); exSelect.changed(() => { stage = 0; });

  describe('Five-stage LLM+GNN pipeline. A paper abstract is tokenized, encoded by an LLM ' +
    'into a single embedding, mixed with neighbor embeddings by a GNN, and classified. Step ' +
    'through the stages and pick among Computer Vision, NLP, and Graph ML abstracts to see the ' +
    'predicted category change.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();
  drawGrid();

  if (autoOn && millis() - lastStep > 1500) { stage = Math.min(4, stage + 1); lastStep = millis(); if (stage === 4) { autoOn = false; autoButton.html('Auto Play'); } }

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18); text('LLM + GNN Pipeline', canvasWidth / 2, 6);

  let ex = EXAMPLES[exSelect ? exSelect.elt.selectedIndex : 0];
  let n = 5, gap = 12, x0 = margin, top = 44, cw = (canvasWidth - 2 * margin - (n - 1) * gap) / n, ch = drawHeight - top - 24;
  hoverTip = null;
  for (let s = 0; s < n; s++) {
    let x = x0 + s * (cw + gap);
    let revealed = s <= stage;
    if (s > 0 && s <= stage) drawConnector(x - gap, top + ch / 2, x, top + ch / 2);
    drawStage(s, x, top, cw, ch, revealed, ex);
  }
  if (hoverTip) drawTip();
  fill('#555'); noStroke(); textAlign(LEFT, CENTER); textSize(13); text('Abstract:', 224, drawHeight + 40);
}

function drawGrid() { stroke(235); strokeWeight(1); for (let x = 0; x < canvasWidth; x += 30) line(x, 0, x, drawHeight); for (let y = 0; y < drawHeight; y += 30) line(0, y, canvasWidth, y); }

function drawStage(s, x, y, w, h, on, ex) {
  let cols = ['#1565c0', '#fb8c00', '#fb8c00', '#2e7d32', '#6a1b9a'];
  push(); if (!on) drawingContext.globalAlpha = 0.22;
  stroke(s === stage ? 'black' : '#bbb'); strokeWeight(s === stage ? 2.5 : 1); fill('white'); rect(x, y, w, h, 8);
  // header
  noStroke(); fill(cols[s]); rect(x, y, w, 22, 8, 8, 0, 0); fill('white'); textAlign(CENTER, CENTER); textSize(11); text(STAGE_NAMES[s], x + w / 2, y + 11);
  if (mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h) hoverTip = { lines: [STAGE_NAMES[s], STAGE_INFO[s]] };
  let cy = y + 22, ch2 = h - 22;
  noStroke();
  if (s === 0) { fill('#333'); textAlign(LEFT, TOP); textSize(9.5); text(ex.text, x + 6, cy + 6, w - 12, ch2 - 12); }
  else if (s === 1) { drawTokens(ex.text, x + 6, cy + 8, w - 12, ch2 - 12); }
  else if (s === 2) { drawEncoder(x, cy, w, ch2); }
  else if (s === 3) { drawAgg(x, cy, w, ch2); }
  else { drawBars(ex, x, cy, w, ch2); }
  pop();
}

function drawTokens(abstract, x, y, w, h) {
  let toks = abstract.replace('…', '').split(/\s+/).slice(0, 16);
  let tx = x, ty = y; textSize(9);
  for (let i = 0; i < toks.length; i++) {
    let tw = textWidth(toks[i]) + 8;
    if (tx + tw > x + w) { tx = x; ty += 16; }
    if (ty > y + h - 12) break;
    fill(['#90caf9', '#ffcc80', '#a5d6a7', '#ce93d8', '#ef9a9a'][i % 5]); stroke('#bbb'); strokeWeight(0.5); rect(tx, ty, tw, 13, 3);
    noStroke(); fill('#222'); textAlign(LEFT, CENTER); text(toks[i], tx + 4, ty + 7);
    if (mouseX > tx && mouseX < tx + tw && mouseY > ty && mouseY < ty + 13) hoverTip = { lines: ['Token: ' + toks[i], 'token id: ' + (1000 + (i * 137 % 8000))] };
    tx += tw + 3;
  }
}
function drawEncoder(x, y, w, h) {
  let bx = x + w / 2 - 18;
  for (let k = 0; k < 4; k++) { fill('#fb8c00'); stroke('#e65100'); rect(bx, y + 8 + k * 14, 36, 11, 2); }
  noStroke(); fill('#2e7d32'); rect(x + w / 2 - 22, y + h - 22, 44, 12, 2);
  fill('#1b5e20'); textAlign(CENTER, CENTER); textSize(8); text('eᵥ∈ℝ³⁸⁴', x + w / 2, y + h - 16);
  if (mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h) hoverTip = { lines: ['LLM encoder output eᵥ', 'dim 384 · e.g. SciBERT/BERT'] };
}
function drawAgg(x, y, w, h) {
  let cx = x + w / 2, cy = y + h / 2;
  let nb = [[cx - 22, cy - 20], [cx + 22, cy - 18], [cx, cy + 24]];
  for (let p of nb) { stroke('#90caf9'); strokeWeight(1.5); drawArrowSeg(p[0], p[1], cx, cy, 11); noStroke(); fill('#42a5f5'); circle(p[0], p[1], 11); }
  noStroke(); fill('#2e7d32'); circle(cx, cy, 18);
  noStroke(); fill('#555'); textAlign(CENTER, TOP); textSize(8); text('+neighbors', cx, y + h - 12);
}
function drawBars(ex, x, y, w, h) {
  let bx = x + 8, bw = w - 16;
  for (let i = 0; i < ex.probs.length; i++) {
    let yy = y + 10 + i * 22; let [cls, p] = ex.probs[i];
    fill(i === 0 ? '#f9a825' : '#90caf9'); stroke('#999'); strokeWeight(0.5); rect(bx, yy, bw * p, 14, 2);
    noStroke(); fill('#222'); textAlign(LEFT, CENTER); textSize(9); text(cls + ' ' + Math.round(p * 100) + '%', bx + 2, yy + 7);
    if (mouseX > bx && mouseX < bx + bw && mouseY > yy && mouseY < yy + 14) hoverTip = { lines: [cls, 'confidence: ' + nf(p, 1, 2)] };
  }
}

function drawConnector(x1, y1, x2, y2) { stroke('#888'); strokeWeight(2); drawArrowSeg(x1, y1, x2, y2, 4); }
function drawArrowSeg(x1, y1, x2, y2, rad) { let ang = atan2(y2 - y1, x2 - x1), ex = x2 - cos(ang) * rad, ey = y2 - sin(ang) * rad; line(x1, y1, ex, ey); push(); translate(ex, ey); rotate(ang); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -6, 2, -6, -2); pop(); }

function drawTip() {
  let w = 250, h = hoverTip.lines.length * 16 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(11.5);
  for (let k = 0; k < hoverTip.lines.length; k++) { if (k === 0) textStyle(BOLD); else textStyle(NORMAL); text(hoverTip.lines[k], tx + 8, ty + 6 + k * 16, w - 16); }
  textStyle(NORMAL);
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
