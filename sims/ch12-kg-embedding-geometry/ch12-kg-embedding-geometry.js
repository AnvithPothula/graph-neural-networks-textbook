// TransE Embedding Geometry in 2D
// CANVAS_HEIGHT: 520
// Learning objective (Apply): watch TransE training drive valid triples toward
// h + r ≈ t, and see why a symmetric relation (France–borders–Germany and the
// reverse) forces its vector toward zero — TransE cannot model symmetry.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let stepButton, autoButton, resetButton, invalidCheckbox, entSelect, relSelect;

const ENT = ['France', 'Germany', 'Japan', 'Paris', 'Berlin', 'Tokyo'];
const REL = ['hasCapital', 'borders'];
// positive triples [h, r, t]
const TRIP = [[0, 0, 3], [1, 0, 4], [2, 0, 5], [0, 1, 1], [1, 1, 0]];
const REL_COL = ['#1565c0', '#c62828'];
const MARGIN = 1.0, LR = 0.04;

let E = [], R = [], epoch = 0, lossHist = [];
let selEnt = 0, selRel = 0, showInvalid = false, autoOn = false;
let rng = mulberry32(7);

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  stepButton = createButton('Step'); stepButton.position(10, drawHeight + 8); stepButton.mousePressed(() => { autoOn = false; autoButton.html('Auto'); trainEpoch(); });
  autoButton = createButton('Auto'); autoButton.position(64, drawHeight + 8); autoButton.mousePressed(() => { autoOn = !autoOn; autoButton.html(autoOn ? 'Pause' : 'Auto'); });
  resetButton = createButton('Reset'); resetButton.position(124, drawHeight + 8); resetButton.mousePressed(reset);
  invalidCheckbox = createCheckbox(' Show invalid triples', false); invalidCheckbox.position(186, drawHeight + 10); invalidCheckbox.changed(() => showInvalid = invalidCheckbox.checked());

  entSelect = createSelect(); entSelect.position(10, drawHeight + 45); ENT.forEach((e, i) => entSelect.option(e, i)); entSelect.changed(() => selEnt = parseInt(entSelect.value()));
  relSelect = createSelect(); relSelect.position(150, drawHeight + 45); REL.forEach((r, i) => relSelect.option(r, i)); relSelect.changed(() => selRel = parseInt(relSelect.value()));

  reset();
  for (let i = 0; i < 140; i++) trainEpoch();   // show a converged geometry on load
  describe('TransE training in 2D. Six entities (circles) and two relations (translation ' +
    'vectors) are learned so that head + relation ≈ tail for each true triple. Step through ' +
    'epochs and watch the dotted h+r arrow close onto the solid h→t arrow. The symmetric ' +
    'borders relation is forced toward zero, illustrating a key TransE limitation.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  if (autoOn && epoch < 400) { trainEpoch(); } if (autoOn && epoch >= 400) { autoOn = false; autoButton.html('Auto'); }

  let gw = canvasWidth * 0.64;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(19);
  text('TransE Embedding Geometry', gw / 2, 6);
  textSize(13); fill('#555'); text('epoch ' + epoch + '   ·   total loss ' + nf(lossHist[lossHist.length - 1] || 0, 1, 3), gw / 2, 28);

  drawPlot(0, 46, gw, drawHeight - 54);
  drawPanel(gw, 46, canvasWidth - gw - margin);

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13);
  text('head', 12, drawHeight + 38); text('relation', 152, drawHeight + 38);
}

// ---------- plot ----------

function plotMap(v, gx, gy, gw, gh) {
  // autoscale: fit max norm into half-min-dim
  let maxN = 1.2; for (let e of E) maxN = Math.max(maxN, Math.hypot(e[0], e[1]));
  let s = (Math.min(gw, gh) / 2 - 30) / maxN;
  return { x: gx + gw / 2 + v[0] * s, y: gy + gh / 2 - v[1] * s };
}

function drawPlot(gx, gy, gw, gh) {
  // axes
  stroke(225); strokeWeight(1);
  let o = plotMap([0, 0], gx, gy, gw, gh);
  line(gx + 4, o.y, gx + gw - 4, o.y); line(o.x, gy + 4, o.x, gy + gh - 4);

  // selected triple: dotted h→h+r, solid h→t
  let trip = TRIP.find(t => t[0] === selEnt && t[1] === selRel);
  if (trip) {
    let h = E[trip[0]], r = R[trip[1]], t = E[trip[2]];
    let ph = plotMap(h, gx, gy, gw, gh), phr = plotMap([h[0] + r[0], h[1] + r[1]], gx, gy, gw, gh), pt = plotMap(t, gx, gy, gw, gh);
    // h -> t solid (green)
    stroke('#2e7d32'); strokeWeight(2.5); drawArrow(ph.x, ph.y, pt.x, pt.y);
    // h -> h+r dotted (relation color)
    drawingContext.setLineDash([5, 4]); stroke(REL_COL[trip[1]]); strokeWeight(2.5); drawArrow(ph.x, ph.y, phr.x, phr.y); drawingContext.setLineDash([]);
  }

  // entities (circles first, so labels draw on top of every circle)
  let cen = plotMap([0, 0], gx, gy, gw, gh);
  for (let i = 0; i < ENT.length; i++) {
    let p = plotMap(E[i], gx, gy, gw, gh);
    let isCap = i >= 3;
    stroke(i === selEnt ? 'black' : 80); strokeWeight(i === selEnt ? 3 : 1.5);
    fill(isCap ? '#fff3e0' : '#e3f2fd');
    circle(p.x, p.y, 14);
  }
  // labels last, offset radially outward, then greedily de-overlapped by nudging down
  textSize(11); textAlign(CENTER, CENTER);
  let placed = [];   // {x, y, w} of labels already drawn
  // place denser-cluster labels last so isolated ones anchor first
  let order = ENT.map((_, i) => i).sort((a, b) => {
    let pa = plotMap(E[a], gx, gy, gw, gh), pb = plotMap(E[b], gx, gy, gw, gh);
    return (pa.y) - (pb.y);   // top-to-bottom
  });
  for (let i of order) {
    let p = plotMap(E[i], gx, gy, gw, gh);
    let dx = p.x - cen.x, dy = p.y - cen.y, dn = Math.hypot(dx, dy) || 1;
    let lx = p.x + (dx / dn) * 16;
    let ly = p.y + (dy / dn) * 16;
    let w = textWidth(ENT[i]) + 6;
    // nudge down until this label box clears all previously placed boxes
    let guard = 0;
    while (guard++ < 12 && placed.some(q => Math.abs(q.x - lx) < (q.w + w) / 2 && Math.abs(q.y - ly) < 15)) {
      ly += 15;
    }
    lx = constrain(lx, gx + 4 + w / 2, gx + gw - 4 - w / 2);
    ly = constrain(ly, gy + 10, gy + gh - 6);
    placed.push({ x: lx, y: ly, w });
    noStroke(); fill(255, 255, 255, 220);
    rectMode(CENTER); rect(lx, ly, w, 14, 3); rectMode(CORNER);
    fill(i === selEnt ? 'black' : '#333'); text(ENT[i], lx, ly);
  }

  // invalid triples
  if (showInvalid) {
    for (let trp of TRIP) {
      let h = E[trp[0]], r = R[trp[1]], t = E[trp[2]];
      let d = Math.hypot(h[0] + r[0] - t[0], h[1] + r[1] - t[1]);
      if (d > MARGIN * 0.5) {
        let pt = plotMap(t, gx, gy, gw, gh);
        noStroke(); fill(198, 40, 40, 60); circle(pt.x, pt.y, 28);
      }
    }
  }
}

function drawArrow(x1, y1, x2, y2) {
  line(x1, y1, x2, y2);
  let a = atan2(y2 - y1, x2 - x1);
  push(); translate(x2, y2); rotate(a); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -9, 3, -9, -3); pop();
}

function drawPanel(px, py, pw) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, py, pw, drawHeight - py - 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14);
  let tx = px + 10, ty = py + 8;
  text('Selected triple', tx, ty); ty += 20;
  let trip = TRIP.find(t => t[0] === selEnt && t[1] === selRel);
  textSize(12);
  if (!trip) { fill('#888'); text('(' + ENT[selEnt] + ', ' + REL[selRel] + ', ?)', tx, ty); text('not a known triple', tx, ty + 16); ty += 40; }
  else {
    let h = E[trip[0]], r = R[trip[1]], t = E[trip[2]];
    let d = Math.hypot(h[0] + r[0] - t[0], h[1] + r[1] - t[1]);
    fill('black'); text(ENT[trip[0]] + ' + ' + REL[trip[1]] + ' ≈ ' + ENT[trip[2]], tx, ty); ty += 18;
    text('‖h + r − t‖ = ' + nf(d, 1, 3), tx, ty); ty += 22;
  }
  // relation magnitudes
  text('Relation magnitudes:', tx, ty); ty += 16;
  for (let i = 0; i < REL.length; i++) {
    fill(REL_COL[i]); text('  ' + REL[i] + ': ' + nf(Math.hypot(R[i][0], R[i][1]), 1, 2), tx, ty); ty += 16;
  }
  fill('#888'); textSize(11); text('(borders → 0: symmetric', tx, ty); ty += 13; text(' relations collapse)', tx, ty); ty += 20;

  // loss curve
  fill('black'); textSize(12); textAlign(LEFT, TOP); text('Loss', tx, ty); ty += 4;
  let lx = tx, ly = ty + 14, lw = pw - 24, lh = drawHeight - 8 - ly - 6;
  stroke('silver'); strokeWeight(1); noFill(); rect(lx, ly, lw, lh);
  if (lossHist.length > 1) {
    let mx = Math.max(...lossHist, 0.01);
    stroke('#c62828'); strokeWeight(2); noFill(); beginShape();
    for (let k = 0; k < lossHist.length; k++) vertex(lx + (k / (lossHist.length - 1)) * lw, ly + lh - (lossHist[k] / mx) * lh);
    endShape();
  }
}

// ---------- TransE training ----------

function reset() {
  rng = mulberry32(7);
  E = ENT.map(() => norm2([rng() * 2 - 1, rng() * 2 - 1]));
  R = REL.map(() => [rng() * 0.6 - 0.3, rng() * 0.6 - 0.3]);
  epoch = 0; lossHist = [totalLoss()]; autoOn = false; if (autoButton) autoButton.html('Auto');
}

function trainEpoch() {
  // Attractive term: drive h + r toward t for every positive triple.
  // For the symmetric pair (F,borders,G) and (G,borders,F) the two updates sum to
  // -2*LR*r on R[borders], so the relation vector provably decays toward zero.
  for (let [h, r, t] of TRIP) {
    let d = sub(add(E[h], R[r]), E[t]);   // h + r - t
    let g = vscale(d, LR);
    E[h] = sub(E[h], g); R[r] = sub(R[r], g); E[t] = add(E[t], g);
  }
  // Repulsion: keep distinct entities apart so they don't collapse to one point.
  for (let i = 0; i < E.length; i++) for (let j = i + 1; j < E.length; j++) {
    let d = sub(E[i], E[j]); let dn = Math.hypot(d[0], d[1]) + 1e-3;
    if (dn < 0.6) { let push = vscale(d, LR * 0.6 * (0.6 - dn) / dn); E[i] = add(E[i], push); E[j] = sub(E[j], push); }
  }
  for (let i = 0; i < E.length; i++) E[i] = norm2(E[i]);   // unit-norm constraint
  epoch++;
  lossHist.push(totalLoss());
  if (lossHist.length > 400) lossHist.shift();
}

function totalLoss() {
  let L = 0;
  for (let [h, r, t] of TRIP) {
    let dp = Math.hypot(E[h][0] + R[r][0] - E[t][0], E[h][1] + R[r][1] - E[t][1]);
    L += Math.max(0, dp);   // distance as a proxy aggregate
  }
  return L;
}

function add(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
function vscale(a, s) { return [a[0] * s, a[1] * s]; }
function norm2(a) { let n = Math.hypot(a[0], a[1]) || 1; return [a[0] / n, a[1] / n]; }
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
