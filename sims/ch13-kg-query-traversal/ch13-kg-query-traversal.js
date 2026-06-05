// Query2Box Multi-Hop Query Traversal
// CANVAS_HEIGHT: 520
// Learning objective (Analyze): watch Query2Box project and intersect boxes to
// resolve 1p / 2p / 2i queries over a toy KG. Each projection shifts and widens
// the box; intersection narrows to the overlap; entities inside the final box are
// exactly the answers.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 460;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let querySelect, anchorSelect, relSelect, stepButton, resetButton;

const ENT = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
// embedding-space coordinates (also used as KG layout)
const EMB = [[0.2, 0.25], [0.5, 0.25], [0.8, 0.25], [0.2, 0.55], [0.5, 0.55], [0.8, 0.55], [0.35, 0.85], [0.65, 0.85]];
const REL = ['r0 (→)', 'r1 (↓)', 'r2 (↘)'];
const TRANS = [[0.3, 0], [0, 0.3], [0.15, 0.3]];
const WIDEN = [[0.07, 0.05], [0.05, 0.07], [0.08, 0.08]];
const REL_COL = ['#1565c0', '#c62828', '#6a1b9a'];

let edges = [];   // {h, r, t}
let queryType = '1p', anchor = 0, rel = 0;
let stepN = 0, boxes = [], answers = [];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildKG();

  querySelect = createSelect(); querySelect.position(10, drawHeight + 8);
  ['1p (one hop)', '2p (two hops)', '2i (intersection)'].forEach(o => querySelect.option(o));
  querySelect.changed(() => { queryType = querySelect.value().slice(0, 2); resetQuery(); });

  anchorSelect = createSelect(); anchorSelect.position(150, drawHeight + 8);
  ENT.forEach((e, i) => anchorSelect.option('anchor ' + e, i));
  anchorSelect.changed(() => { anchor = parseInt(anchorSelect.value()); resetQuery(); });

  relSelect = createSelect(); relSelect.position(250, drawHeight + 8);
  REL.forEach((r, i) => relSelect.option(r, i));
  relSelect.changed(() => { rel = parseInt(relSelect.value()); resetQuery(); });

  stepButton = createButton('Step'); stepButton.position(360, drawHeight + 8); stepButton.mousePressed(doStep);
  resetButton = createButton('Reset'); resetButton.position(414, drawHeight + 8); resetButton.mousePressed(resetQuery);

  resetQuery();
  describe('Query2Box traversal over a toy knowledge graph. Pick a query type (1p, 2p, ' +
    'or 2i), an anchor entity, and a relation, then Step to project boxes through the ' +
    'embedding space. Each projection shifts and widens the box; the 2i query intersects ' +
    'two chains. Entities inside the final box are the answers, highlighted in both panels.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let half = canvasWidth / 2;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(16);
  text('Knowledge graph', half / 2, 6);
  text('Embedding space (boxes)', half + half / 2, 6);
  stroke('silver'); line(half, 28, half, drawHeight - 8);

  drawKG(0, 32, half, drawHeight - 40);
  drawBoxes(half, 32, half, drawHeight - 40);

  // status
  noStroke(); textAlign(CENTER, BOTTOM); textSize(13); fill('#333');
  let msg = queryLabel() + '   ·   step ' + stepN + '/' + maxSteps() + (stepN >= maxSteps() ? '  → answers: ' + (answers.map(i => ENT[i]).join(', ') || 'none') : '');
  text(msg, canvasWidth / 2, drawHeight - 2);
}

function embScreen(e, gx, gy, gw, gh) { return { x: gx + 20 + e[0] * (gw - 40), y: gy + 16 + e[1] * (gh - 36) }; }

function drawKG(gx, gy, gw, gh) {
  // relation edges
  for (let e of edges) {
    let a = embScreen(EMB[e.h], gx, gy, gw, gh), b = embScreen(EMB[e.t], gx, gy, gw, gh);
    stroke(REL_COL[e.r]); strokeWeight(1.5);
    let ang = atan2(b.y - a.y, b.x - a.x);
    let ex = b.x - cos(ang) * 16, ey = b.y - sin(ang) * 16;
    line(a.x, a.y, ex, ey);
    push(); translate(ex, ey); rotate(ang); noStroke(); fill(REL_COL[e.r]); triangle(0, 0, -7, 2.5, -7, -2.5); pop();
  }
  // entities
  for (let i = 0; i < ENT.length; i++) {
    let p = embScreen(EMB[i], gx, gy, gw, gh);
    let isAnchor = (queryType !== '2i' && i === anchor) || (queryType === '2i' && (i === 3 || i === 1));
    let isAns = answers.includes(i) && stepN >= maxSteps();
    if (isAns) { stroke('#f9a825'); strokeWeight(4); } else if (isAnchor) { stroke('#2e7d32'); strokeWeight(3); } else { stroke(70); strokeWeight(1.5); }
    fill(isAns ? '#fff8e1' : 'white'); circle(p.x, p.y, 26);
    noStroke(); fill('black'); textAlign(CENTER, CENTER); textSize(13); text(ENT[i], p.x, p.y);
  }
  noStroke(); textAlign(LEFT, BOTTOM); textSize(10);
  for (let i = 0; i < REL.length; i++) { fill(REL_COL[i]); text(REL[i], gx + 6 + i * 56, gy + gh + 2); }
}

function drawBoxes(gx, gy, gw, gh) {
  // entity points
  for (let i = 0; i < ENT.length; i++) {
    let p = embScreen(EMB[i], gx, gy, gw, gh);
    let isAns = answers.includes(i) && stepN >= maxSteps();
    stroke(isAns ? '#f9a825' : 120); strokeWeight(isAns ? 3 : 1); fill(isAns ? '#f9a825' : '#cfd8dc');
    circle(p.x, p.y, isAns ? 14 : 9);
    noStroke(); fill('black'); textAlign(CENTER, BOTTOM); textSize(10); text(ENT[i], p.x, p.y - 7);
  }
  // boxes
  for (let bx of boxes) {
    let lo = embScreen([bx.cx - bx.ox, bx.cy - bx.oy], gx, gy, gw, gh);
    let hi = embScreen([bx.cx + bx.ox, bx.cy + bx.oy], gx, gy, gw, gh);
    let col = bx.kind === 'final' ? color(46, 125, 50) : bx.kind === 'inter' ? color(249, 168, 37) : color(21, 101, 192);
    stroke(col); strokeWeight(2);
    fill(red(col), green(col), blue(col), bx.kind === 'inter' ? 70 : 30);
    rectMode(CORNERS); rect(lo.x, lo.y, hi.x, hi.y); rectMode(CORNER);
  }
}

// ---------- query logic ----------

function maxSteps() { return queryType === '1p' ? 1 : 2; }

function resetQuery() {
  stepN = 0; answers = []; boxes = [];
  if (queryType === '2i') {
    // fixed illustrative chains: (D -r0-> x) ∩ (B -r1-> x) → E
    boxes = [pointBox(3, 'cur'), pointBox(1, 'cur')];
  } else {
    boxes = [pointBox(anchor, 'cur')];
  }
}

function pointBox(entIdx, kind) { return { cx: EMB[entIdx][0], cy: EMB[entIdx][1], ox: 0.015, oy: 0.015, kind }; }

function doStep() {
  if (stepN >= maxSteps()) return;
  if (queryType === '2i') {
    if (stepN === 0) {
      // project chain1 by r0, chain2 by r1
      boxes = [project(boxes[0], 0, 'cur'), project(boxes[1], 1, 'cur')];
    } else {
      // intersect
      let inter = intersect(boxes[0], boxes[1]);
      inter.kind = 'final';
      boxes = [{ ...boxes[0], kind: 'cur' }, { ...boxes[1], kind: 'cur' }, inter];
      computeAnswers(inter);
    }
  } else {
    let nb = project(boxes[boxes.length - 1], rel, stepN + 1 === maxSteps() ? 'final' : 'cur');
    boxes.push(nb);
    if (stepN + 1 === maxSteps()) computeAnswers(nb);
  }
  stepN++;
}

function project(box, r, kind) {
  return { cx: box.cx + TRANS[r][0], cy: box.cy + TRANS[r][1], ox: box.ox + WIDEN[r][0], oy: box.oy + WIDEN[r][1], kind };
}

function intersect(a, b) {
  let lo0 = Math.max(a.cx - a.ox, b.cx - b.ox), hi0 = Math.min(a.cx + a.ox, b.cx + b.ox);
  let lo1 = Math.max(a.cy - a.oy, b.cy - b.oy), hi1 = Math.min(a.cy + a.oy, b.cy + b.oy);
  if (hi0 < lo0 || hi1 < lo1) return { cx: (a.cx + b.cx) / 2, cy: (a.cy + b.cy) / 2, ox: 0.005, oy: 0.005, kind: 'inter' };
  return { cx: (lo0 + hi0) / 2, cy: (lo1 + hi1) / 2, ox: (hi0 - lo0) / 2, oy: (hi1 - lo1) / 2, kind: 'inter' };
}

function computeAnswers(box) {
  answers = [];
  for (let i = 0; i < ENT.length; i++) {
    if (Math.abs(EMB[i][0] - box.cx) <= box.ox + 0.04 && Math.abs(EMB[i][1] - box.cy) <= box.oy + 0.04) answers.push(i);
  }
}

function queryLabel() {
  if (queryType === '2i') return '2i:  (D, r0, x) ∧ (B, r1, x)';
  if (queryType === '2p') return '2p:  ' + ENT[anchor] + ' →' + REL[rel] + '→ →' + REL[rel] + '→ x';
  return '1p:  ' + ENT[anchor] + ' →' + REL[rel] + '→ x';
}

// ---------- KG construction ----------

function buildKG() {
  edges = [];
  for (let h = 0; h < ENT.length; h++) for (let r = 0; r < REL.length; r++) {
    let tx = EMB[h][0] + TRANS[r][0], ty = EMB[h][1] + TRANS[r][1];
    for (let t = 0; t < ENT.length; t++) if (t !== h && Math.hypot(EMB[t][0] - tx, EMB[t][1] - ty) < 0.06) edges.push({ h, r, t });
  }
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
