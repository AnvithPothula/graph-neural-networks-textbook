// ULTRA-style Structural Embedding Transfer across two KGs
// CANVAS_HEIGHT: 520
// Learning objective (Analyze): structurally equivalent entities in two different
// KGs (different IDs) get the same embedding because the representation encodes
// structural role, not identity. Shuffling IDs changes nothing; perturbing
// structure changes the embedding.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let shuffleButton, transferButton, perturbButton, resetButton;

const NT = 12;
let pos = [];                 // shared layout (topology identical)
let edgesTrain = [], edgesTest = [];
let labelTrain = [], labelTest = [];
let proj = [];                // fixed 4x3 random projection (fingerprint -> RGB)
let showTransfer = false, hoverNode = null;
const REL_COL = ['#009688', '#ff7043', '#7e57c2'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  // base topology (relation type on each edge)
  let base = [[0, 1, 0], [0, 2, 0], [0, 4, 2], [0, 9, 0], [1, 3, 1], [1, 5, 2], [1, 8, 1],
              [2, 3, 1], [2, 6, 0], [3, 7, 1], [4, 5, 0], [6, 7, 2], [5, 9, 2], [6, 10, 0], [7, 11, 1]];
  edgesTrain = base.map(e => e.slice());
  edgesTest = base.map(e => e.slice());
  layout();
  let rng = mulberry32(2024);
  for (let k = 0; k < 3; k++) { let v = []; for (let d = 0; d < 4; d++) v.push(rng() * 2 - 1); proj.push(v); }
  labelTrain = idLabels(0); labelTest = idLabels(100);

  shuffleButton = createButton('Shuffle Entity IDs'); shuffleButton.position(10, drawHeight + 8); shuffleButton.mousePressed(() => { labelTest = shuffled(labelTest); });
  transferButton = createButton('Show Transfer'); transferButton.position(150, drawHeight + 8); transferButton.mousePressed(() => { showTransfer = !showTransfer; });
  perturbButton = createButton('Perturb Structure'); perturbButton.position(262, drawHeight + 8); perturbButton.mousePressed(perturb);
  resetButton = createButton('Reset'); resetButton.position(392, drawHeight + 8); resetButton.mousePressed(reset);

  describe('Two knowledge graphs side by side share an identical structure but use ' +
    'different entity IDs. Each node is colored by a fixed projection of its structural ' +
    'fingerprint (degree + relation-type counts), so structurally equivalent entities get ' +
    'the same color in both KGs. Shuffle IDs (color unchanged) and Perturb Structure ' +
    '(color changes) to see that the representation encodes role, not identity.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let half = canvasWidth / 2;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(16);
  text('Training KG', half / 2, 6); text('Test KG (new IDs, same structure)', half + half / 2, 6);
  stroke('silver'); line(half, 28, half, drawHeight - 8);

  hoverNode = null;
  let gy = 36, gh = drawHeight - gy - 10;
  let fpTrain = fingerprints(edgesTrain), fpTest = fingerprints(edgesTest);

  // transfer arcs (drawn under nodes)
  if (showTransfer) {
    for (let i = 0; i < NT; i++) {
      let best = -1, bd = 1e9;
      for (let j = 0; j < NT; j++) { let d = fpDist(fpTest[i], fpTrain[j]); if (d < bd) { bd = d; best = j; } }
      let a = nodeXY(i, half, half, gy, gh), b = nodeXY(best, 0, half, gy, gh);
      let sim = 1 / (1 + bd);
      stroke(120, 120, 120, 40 + sim * 150); strokeWeight(1 + sim * 2); noFill();
      let mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 - 30;
      beginShape(); vertex(a.x, a.y); quadraticVertex(mx, my, b.x, b.y); endShape();
    }
  }

  drawKG(0, half, gy, gh, edgesTrain, labelTrain, fpTrain, 0);
  drawKG(half, half, gy, gh, edgesTest, labelTest, fpTest, 1);

  if (hoverNode) drawTooltip(hoverNode);
}

function nodeXY(i, px, pw, gy, gh) { return { x: px + 30 + pos[i].x * (pw - 60), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawKG(px, pw, gy, gh, edges, labels, fp, side) {
  // edges
  strokeWeight(2);
  for (let e of edges) { stroke(REL_COL[e[2]]); let a = nodeXY(e[0], px, pw, gy, gh), b = nodeXY(e[1], px, pw, gy, gh); line(a.x, a.y, b.x, b.y); }
  // nodes
  for (let i = 0; i < NT; i++) {
    let p = nodeXY(i, px, pw, gy, gh);
    let deg = fp[i][0];
    let rad = map(deg, 1, 5, 9, 18);
    if (dist(mouseX, mouseY, p.x, p.y) <= rad + 2) hoverNode = { side, i, fp: fp[i], deg };
    let c = fpColor(fp[i]);
    stroke(40); strokeWeight(1.5); fill(c);
    circle(p.x, p.y, rad * 2);
    noStroke(); fill(lumOf(c) > 60 ? 'black' : 'white'); textAlign(CENTER, CENTER); textSize(10); text(labels[i], p.x, p.y);
  }
}

function drawTooltip(h) {
  let role = h.deg >= 4 ? 'Hub' : h.deg >= 2 ? 'Mid' : 'Leaf';
  let lines = ['Entity ' + (h.side === 0 ? labelTrain : labelTest)[h.i], role + ': degree ' + h.deg,
    'rel counts [' + h.fp.slice(1).join(',') + ']', 'embed (' + nf(embed2(h.fp)[0], 1, 2) + ', ' + nf(embed2(h.fp)[1], 1, 2) + ')'];
  let w = 200, ht = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 30, drawHeight - ht - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, ht, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- structural fingerprint ----------

function fingerprints(edges) {
  let fp = Array.from({ length: NT }, () => [0, 0, 0, 0]);   // [deg, t0, t1, t2]
  for (let e of edges) { fp[e[0]][0]++; fp[e[1]][0]++; fp[e[0]][1 + e[2]]++; fp[e[1]][1 + e[2]]++; }
  return fp;
}
function fpDist(a, b) { let s = 0; for (let k = 0; k < 4; k++) s += (a[k] - b[k]) ** 2; return Math.sqrt(s); }
function embed2(fp) { return [proj[0][0] * fp[0] + proj[0][1] * fp[1] + proj[0][2] * fp[2], proj[1][0] * fp[0] + proj[1][1] * fp[1] + proj[1][2] * fp[2]]; }
function fpColor(fp) {
  // project to RGB via the 4x3 matrix
  let r = 0, g = 0, b = 0;
  for (let k = 0; k < 4; k++) { r += proj[0][k] * fp[k]; g += proj[1][k] * fp[k]; b += proj[2][k] * fp[k]; }
  return color(map(r, -8, 8, 60, 235), map(g, -8, 8, 80, 235), map(b, -8, 8, 90, 235));
}
function lumOf(c) { return (red(c) + green(c) + blue(c)) / 3 - 100; }

// ---------- interactions ----------

function layout() { pos = []; for (let i = 0; i < NT; i++) { let a = TWO_PI * i / NT - HALF_PI; let r = (i % 3 === 0) ? 0.22 : 0.42; pos.push({ x: 0.5 + r * Math.cos(a), y: 0.5 + r * Math.sin(a) }); } }
function idLabels(off) { return Array.from({ length: NT }, (_, i) => off + i); }
function shuffled(arr) { let a = arr.slice(); let rng = mulberry32(Math.floor(arr[0] + a.length * 7 + frameCount)); for (let i = a.length - 1; i > 0; i--) { let j = Math.floor(rng() * (i + 1));[a[i], a[j]] = [a[j], a[i]]; } return a; }
function perturb() {
  // rewire 2 random edges in the test KG
  let rng = mulberry32(Math.floor(frameCount + 13));
  for (let n = 0; n < 2; n++) {
    let idx = Math.floor(rng() * edgesTest.length);
    let nt = Math.floor(rng() * NT);
    edgesTest[idx] = [edgesTest[idx][0], nt === edgesTest[idx][0] ? (nt + 1) % NT : nt, edgesTest[idx][2]];
  }
}
function reset() { edgesTest = edgesTrain.map(e => e.slice()); labelTest = idLabels(100); showTransfer = false; }

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
