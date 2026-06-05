// Multi-Hop KG Reasoning Agent
// CANVAS_HEIGHT: 560
// Learning objective (Apply): trace an agent's query → retrieve → reason → refine loop
// across a knowledge graph, watching each hop narrow toward the answer.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 960;
let drawHeight = 480;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let nextButton, autoButton, resetButton, qSelect;
let ent = [], rel = [], pos = [], questions = [], qi = 0, hop = 0, autoOn = false, lastStep = 0, hoverN = -1;
let activeNodes = new Set(), pathNodes = new Set(), activeEdges = new Set();
const TYPE_COL = { person: '#90caf9', inst: '#a5d6a7', model: '#ffcc80' };

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildKG();

  nextButton = createButton('Next Hop →'); nextButton.position(10, drawHeight + 8); nextButton.mousePressed(() => { autoOn = false; autoButton.html('Auto Run'); doHop(); });
  autoButton = createButton('Auto Run'); autoButton.position(108, drawHeight + 8); autoButton.mousePressed(() => { autoOn = !autoOn; autoButton.html(autoOn ? 'Pause' : 'Auto Run'); lastStep = 0; });
  resetButton = createButton('Reset'); resetButton.position(190, drawHeight + 8); resetButton.mousePressed(resetQ);
  qSelect = createSelect(); qSelect.position(256, drawHeight + 8); questions.forEach((q, i) => qSelect.option((i + 1) + '-hop: ' + q.q.substring(0, 28) + '…', i)); qSelect.changed(() => { qi = qSelect.elt.selectedIndex; resetQ(); });

  resetQ();
  describe('A multi-hop reasoning agent over a small academic knowledge graph. Pick a ' +
    'question, then step the agent through its hops: it retrieves a subgraph, reasons over it, ' +
    'refines its query, and advances along the answer path while its confidence grows. The ' +
    'right panel logs the query / retrieved triples / reasoning / refined query at each hop.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let q = questions[qi];
  if (autoOn && hop < q.hops.length && millis() - lastStep > 2000) { doHop(); lastStep = millis(); }
  if (autoOn && hop >= q.hops.length) { autoOn = false; autoButton.html('Auto Run'); }

  let lw = canvasWidth - 340;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(17); text('Multi-Hop KG Reasoning Agent', lw / 2, 6);

  hoverN = -1;
  drawKG(0, 30, lw, drawHeight - 38);
  drawLog(lw, 30, 340 - margin, drawHeight - 38, q);
  if (hoverN >= 0) drawTip();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13); text('Question:', 256, drawHeight + 40);
}

function ptOf(i, gx, gy, gw, gh) { return { x: gx + 26 + pos[i].x * (gw - 52), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawKG(gx, gy, gw, gh) {
  // edges
  for (let r of rel) {
    let a = ptOf(r[0], gx, gy, gw, gh), b = ptOf(r[1], gx, gy, gw, gh);
    let key = r[0] + '_' + r[1], on = activeEdges.has(key);
    stroke(on ? '#f9a825' : color(120, 120, 120, 80)); strokeWeight(on ? 3 : 1.2);
    drawArrowSeg(a.x, a.y, b.x, b.y, 18);
    if (on) {
      // relation label with a white backing pill so it stays readable over edges/nodes
      let mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 - 6;
      noStroke(); textAlign(CENTER, CENTER); textSize(9); let w = textWidth(r[2]) + 6;
      fill(255, 255, 255, 230); rectMode(CENTER); rect(mx, my, w, 12, 3); rectMode(CORNER);
      fill('#e65100'); text(r[2], mx, my);
    }
  }
  // nodes
  for (let i = 0; i < ent.length; i++) {
    let p = ptOf(i, gx, gy, gw, gh);
    if (dist(mouseX, mouseY, p.x, p.y) <= 22) hoverN = i;
    let path = pathNodes.has(i), act = activeNodes.has(i);
    if (act) { noStroke(); fill(255, 193, 7, 120); circle(p.x, p.y, 58); }
    stroke(path ? '#2e7d32' : (act ? '#f9a825' : '#555')); strokeWeight(path || act ? 3.5 : 1.5);
    fill(TYPE_COL[ent[i].type]); circle(p.x, p.y, 44);
    // short two-line label, CENTER/CENTER align with a box centered on the node (p5 text() x,y is the box's top-left, so offset by half)
    noStroke(); fill('#0d1b2a'); textAlign(CENTER, CENTER); textSize(8.5);
    text(ent[i].short, p.x - 21, p.y - 14, 42, 28);
  }
}

function drawLog(gx, gy, gw, gh, q) {
  fill('#1a1a2e'); stroke('#33344e'); strokeWeight(1); rect(gx + 6, gy, gw, gh, 6);
  let tx = gx + 16, ty = gy + 10, lh = 16;
  noStroke(); fill('#bbb'); textAlign(LEFT, TOP); textSize(12); textStyle(BOLD); text('Agent reasoning log', tx, ty); textStyle(NORMAL); ty += 22;
  fill('#90caf9'); textSize(11); text('Hop ' + hop + ' of ' + q.hops.length + ' max', tx, ty); ty += 18;
  if (hop === 0) { fill('#ccc'); textStyle(ITALIC); text('Q: ' + q.q, tx, ty, gw - 20, 60); textStyle(NORMAL); ty += 64; fill('#888'); text('Click "Next Hop →" to begin.', tx, ty); }
  else {
    for (let k = 0; k < hop; k++) {
      let h = q.hops[k];
      fill('#64b5f6'); textStyle(ITALIC); textSize(10.5); text('Q' + (k + 1) + ': ' + h.query, tx, ty, gw - 20, 40); textStyle(NORMAL); ty += textHeight(h.query, gw - 20, 10.5) + 4;
      fill('#aaa'); textSize(9.5); text('retrieved: ' + h.triples, tx, ty, gw - 20, 40); ty += textHeight(h.triples, gw - 20, 9.5) + 3;
      fill('#ddd'); text('reason: ' + h.reason, tx, ty, gw - 20, 60); ty += textHeight(h.reason, gw - 20, 9.5) + 6;
    }
    if (hop >= q.hops.length) { fill('#81c784'); textSize(12); textStyle(BOLD); text('✓ Answer: ' + q.answer, tx, ty); textStyle(NORMAL); ty += 20; }
  }
  // confidence bar
  let conf = hop > 0 ? q.hops[Math.min(hop, q.hops.length) - 1].conf : 0;
  let by = gy + gh - 28;
  noStroke(); fill('#888'); textAlign(LEFT, TOP); textSize(10); text('Confidence', tx, by - 14);
  fill('#33344e'); rect(tx, by, gw - 24, 12, 3); fill(conf >= 0.85 ? '#43a047' : '#fdd835'); rect(tx, by, (gw - 24) * conf, 12, 3);
  fill('#fff'); textAlign(RIGHT, CENTER); text(Math.round(conf * 100) + '%', gx + gw - 4, by + 6);
}

function textHeight(s, w, sz) { textSize(sz); let words = s.split(' '), line = '', lines = 1; for (let wd of words) { if (textWidth(line + wd + ' ') > w) { lines++; line = wd + ' '; } else line += wd + ' '; } return lines * (sz + 3); }

function drawTip() {
  let i = hoverN; let outs = rel.filter(r => r[0] === i).map(r => r[2] + '→' + ent[r[1]].name);
  let lines = [ent[i].name, ent[i].type, outs.length ? outs.slice(0, 4).join(', ') : '(no outgoing relations)'];
  let w = 220, h = lines.length * 16 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(11.5);
  for (let k = 0; k < lines.length; k++) { if (k === 0) textStyle(BOLD); else textStyle(NORMAL); text(lines[k], tx + 8, ty + 6 + k * 16, w - 16); }
  textStyle(NORMAL);
}

function drawArrowSeg(x1, y1, x2, y2, rad) { let ang = atan2(y2 - y1, x2 - x1), ex = x2 - cos(ang) * rad, ey = y2 - sin(ang) * rad; line(x1, y1, ex, ey); push(); translate(ex, ey); rotate(ang); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -7, 2.5, -7, -2.5); pop(); }

// ---------- reasoning ----------

function doHop() {
  let q = questions[qi]; if (hop >= q.hops.length) return;
  let h = q.hops[hop];
  h.activate.forEach(n => { activeNodes.add(n); pathNodes.add(n); });
  h.edges.forEach(e => activeEdges.add(e[0] + '_' + e[1]));
  hop++;
}
function resetQ() { hop = 0; activeNodes = new Set(); pathNodes = new Set(); activeEdges = new Set(); autoOn = false; if (autoButton) autoButton.html('Auto Run'); let s = questions[qi].seeds; s.forEach(n => { activeNodes.add(n); pathNodes.add(n); }); }

function buildKG() {
  ent = [
    { name: 'Bahdanau', short: 'Bah-\ndanau', type: 'person' }, { name: 'Vaswani', short: 'Vas-\nwani', type: 'person' }, { name: 'Veličković', short: 'Velič-\nković', type: 'person' },
    { name: 'U.Montreal', short: 'U.Mon-\ntreal', type: 'inst' }, { name: 'GoogleBrain', short: 'Google\nBrain', type: 'inst' }, { name: 'Stanford', short: 'Stan-\nford', type: 'inst' },
    { name: 'Attention', short: 'Atten-\ntion', type: 'model' }, { name: 'Transformer', short: 'Trans-\nformer', type: 'model' }, { name: 'GAT', short: 'GAT', type: 'model' },
    { name: 'GCN', short: 'GCN', type: 'model' }, { name: 'GraphSAGE', short: 'Graph\nSAGE', type: 'model' }, { name: 'GIN', short: 'GIN', type: 'model' },
  ];
  rel = [
    [0, 6, 'invented'], [1, 7, 'invented'], [2, 8, 'invented'], [7, 6, 'uses'], [8, 6, 'uses'], [8, 9, 'extends'],
    [0, 3, 'affiliated'], [1, 4, 'affiliated'], [2, 3, 'affiliated'], [10, 9, 'extends'], [11, 9, 'extends'],
    [9, 6, 'inspired-by'], [5, 9, 'studies'], [4, 7, 'developed'], [10, 6, 'aggregates'], [11, 6, 'aggregates'], [3, 2, 'employs'], [5, 8, 'teaches'],
  ];
  // questions
  questions = [
    { q: 'Who invented the Attention mechanism?', seeds: [6], answer: 'Bahdanau', hops: [
      { query: 'who invented Attention?', triples: '(Bahdanau, invented, Attention)', reason: 'Attention has one inventor edge → Bahdanau.', activate: [0], edges: [[0, 6]], conf: 0.9 } ] },
    { q: 'What model uses the same mechanism GAT uses?', seeds: [8], answer: 'Transformer', hops: [
      { query: 'what does GAT use?', triples: '(GAT, uses, Attention)', reason: 'GAT uses Attention; pivot on Attention.', activate: [6], edges: [[8, 6]], conf: 0.5 },
      { query: 'what else uses Attention?', triples: '(Transformer, uses, Attention)', reason: 'Transformer also uses Attention → answer.', activate: [7], edges: [[7, 6]], conf: 0.9 } ] },
    { q: 'Who else works where the GAT inventor is affiliated?', seeds: [8], answer: 'Bahdanau', hops: [
      { query: 'who invented GAT?', triples: '(Veličković, invented, GAT)', reason: 'GAT inventor is Veličković.', activate: [2], edges: [[2, 8]], conf: 0.4 },
      { query: 'where is Veličković affiliated?', triples: '(Veličković, affiliated, U.Montreal)', reason: 'Affiliated with U.Montreal.', activate: [3], edges: [[2, 3]], conf: 0.65 },
      { query: 'who else is at U.Montreal?', triples: '(Bahdanau, affiliated, U.Montreal)', reason: 'Bahdanau is also at U.Montreal → answer.', activate: [0], edges: [[0, 3]], conf: 0.9 } ] },
  ];
  pos = springLayout();
}

function springLayout() {
  let n = ent.length, p = []; for (let i = 0; i < n; i++) { let a = TWO_PI * i / n; p.push({ x: 0.5 + 0.4 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a) }); }
  for (let it = 0; it < 280; it++) {
    let fx = new Array(n).fill(0), fy = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), r = 0.0011 / d2; fx[i] += dx / d * r; fy[i] += dy / d * r; fx[j] -= dx / d * r; fy[j] -= dy / d * r; }
    for (let e of rel) { let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, s = (d - 0.16) * 0.02; fx[e[0]] += dx / d * s; fy[e[0]] += dy / d * s; fx[e[1]] -= dx / d * s; fy[e[1]] -= dy / d * s; }
    for (let i = 0; i < n; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
