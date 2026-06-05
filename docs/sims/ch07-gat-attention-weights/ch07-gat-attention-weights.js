// GAT Attention Weights Visualizer
// CANVAS_HEIGHT: 500
// Learning objective (Apply, Bloom L3): click a node and see attention coefficients
// over its neighbors as edge width/color, across 4 independent attention heads,
// connecting the softmax definition to visual intuition.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 450;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let headButtons = [], resetButton;

const N = 12;
let adj = [], pos = [], faction = [];
let focal = -1, head = 0;
let proj = [];   // 4 projection vectors of length 4
let hoverEdge = null;

const BLUE = '#3b82f6', RED = '#ef4444';

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildGraph();
  // 4 fixed projection vectors (length 4: [posi.x,posi.y,posj.x,posj.y])
  let rng = mulberry32(2024);
  for (let k = 0; k < 4; k++) { let v = []; for (let d = 0; d < 4; d++) v.push(rng() * 2 - 1); proj.push(v); }

  for (let k = 0; k < 4; k++) {
    let b = createButton('Head ' + (k + 1));
    b.position(10 + k * 64, drawHeight + 8);
    b.mousePressed(() => head = k);
    headButtons.push(b);
  }
  resetButton = createButton('Reset');
  resetButton.position(10 + 4 * 64 + 8, drawHeight + 8);
  resetButton.mousePressed(() => focal = -1);

  // default focal = highest-degree node, so attention is visible on load
  let degs = adj.map(a => a.length);
  focal = degs.indexOf(Math.max(...degs));

  describe('GAT attention visualizer on a 12-node graph. Click a node to see its ' +
    'attention coefficients over neighbors, drawn as edge width and a cool-to-warm color ' +
    'scale, with the values labeled. Toggle among four independent attention heads.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let panelW = 196, gw = canvasWidth - panelW;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(20);
  text('GAT Attention Weights', gw / 2, 6);

  drawGraph(0, 36, gw, drawHeight - 44);
  drawPanel(gw, 36, panelW);
  highlightHead();
  if (hoverEdge) drawEdgeTooltip();
}

function nodeScreen(i, gx, gy, gw, gh) { return { x: gx + 28 + pos[i].x * (gw - 56), y: gy + 20 + pos[i].y * (gh - 40) }; }

function drawGraph(gx, gy, gw, gh) {
  hoverEdge = null;
  let att = focal >= 0 ? attentionFor(focal) : null;   // map neighbor -> alpha

  // edges
  for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) {
    let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh);
    let isFocalEdge = att && (i === focal || j === focal);
    if (isFocalEdge) {
      let nb = (i === focal) ? j : i;
      let alpha = att[nb];
      stroke(lerpColor(color('#93c5fd'), color('#f97316'), alpha));
      strokeWeight(1 + alpha * 11);
      line(a.x, a.y, b.x, b.y);
      // hover detection
      if (distToSeg(mouseX, mouseY, a.x, a.y, b.x, b.y) < 6) hoverEdge = { i: focal, j: nb, alpha };
      // label midway
      noStroke(); fill('#7c2d12'); textAlign(CENTER, CENTER); textSize(11);
      text(nf(alpha, 1, 2), (a.x + b.x) / 2, (a.y + b.y) / 2 - 8);
    } else {
      stroke(att ? color(200, 200, 200, 60) : color(180));
      strokeWeight(1); line(a.x, a.y, b.x, b.y);
    }
  }

  // nodes
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    if (i === focal) { stroke('white'); strokeWeight(4); } else { stroke(40); strokeWeight(1.5); }
    fill(faction[i] === 0 ? BLUE : RED);
    circle(p.x, p.y, 28);
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(12); text(i, p.x, p.y);
  }

  if (focal < 0) { noStroke(); fill('dimgray'); textAlign(CENTER, TOP); textSize(13); text('Click a node to inspect its attention pattern', gw / 2, gy + gh - 6); }
}

function drawPanel(px, py, pw) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, 30, pw, drawHeight - 30);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14);
  let tx = px + 10, ty = 40;
  text('Head ' + (head + 1), tx, ty); ty += 24;
  if (focal < 0) { textSize(13); fill('dimgray'); text('No focal node selected.', tx, ty); return; }
  text('Focal node: ' + focal, tx, ty); ty += 22;
  textSize(12); fill('dimgray'); text('Neighbor   αᵢⱼ', tx, ty); ty += 16;
  let att = attentionFor(focal);
  let rows = Object.keys(att).map(Number).sort((a, b) => att[b] - att[a]);
  let valX = px + pw - 8;          // right edge for value labels (inside panel)
  let barX = tx + 40;              // bar start
  let barMaxW = (valX - 34) - barX; // leave room for the value label
  for (let nb of rows) {
    if (ty > drawHeight - 16) break;
    let alpha = att[nb];
    noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12); text('' + nb, tx, ty);
    // bar
    stroke('silver'); strokeWeight(1); fill(lerpColor(color('#93c5fd'), color('#f97316'), alpha));
    rect(barX, ty, Math.max(1, alpha * barMaxW), 11, 2);
    noStroke(); fill('black'); textAlign(RIGHT, TOP); text(nf(alpha, 1, 2), valX, ty - 1);
    ty += 16;
  }
  textAlign(LEFT, TOP);
}

function drawEdgeTooltip() {
  let lines = ['Edge (' + hoverEdge.i + '→' + hoverEdge.j + ')', 'αᵢⱼ = ' + nf(hoverEdge.alpha, 1, 3), 'Head: ' + (head + 1)];
  let w = 150, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 36, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

function highlightHead() {
  for (let k = 0; k < 4; k++) {
    headButtons[k].style('background-color', k === head ? '#6366f1' : '#eeeeee');
    headButtons[k].style('color', k === head ? 'white' : 'black');
  }
}

// ---------- synthetic attention ----------

function attentionFor(i) {
  let p = proj[head];
  let scores = {};
  for (let j of adj[i]) {
    let v = [pos[i].x - 0.5, pos[i].y - 0.5, pos[j].x - 0.5, pos[j].y - 0.5];
    let e = Math.tanh(p[0] * v[0] + p[1] * v[1] + p[2] * v[2] + p[3] * v[3]);
    scores[j] = e;
  }
  // softmax over neighbors
  let vals = Object.values(scores), mx = Math.max(...vals);
  let sum = 0; for (let j in scores) { scores[j] = Math.exp(scores[j] - mx); sum += scores[j]; }
  for (let j in scores) scores[j] /= sum;
  return scores;
}

function distToSeg(px, py, x1, y1, x2, y2) {
  let dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy;
  let t = l2 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l2)) : 0;
  return dist(px, py, x1 + t * dx, y1 + t * dy);
}

// ---------- graph ----------

function buildGraph() {
  // Karate subgraph on nodes 0-11
  let all = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[1,2],[1,3],[1,7],
    [2,3],[2,7],[2,8],[2,9],[3,7],[4,6],[4,10],[5,6],[5,10],[6,10],[8,9]];
  let edges = all.filter(e => e[0] < N && e[1] < N);
  adj = Array.from({ length: N }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  let officer = new Set([8, 9]);
  faction = []; for (let i = 0; i < N; i++) faction.push(officer.has(i) ? 1 : 0);
  pos = springLayout(edges);
  buildGraph.edges = edges;
}

function springLayout(edges) {
  let p = [];
  for (let i = 0; i < N; i++) { let a = TWO_PI * i / N; p.push({ x: 0.5 + 0.38 * Math.cos(a), y: 0.5 + 0.38 * Math.sin(a) }); }
  for (let it = 0; it < 300; it++) {
    let fx = new Array(N).fill(0), fy = new Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), rep = 0.0016 / d2;
      fx[i] += dx / d * rep; fy[i] += dy / d * rep; fx[j] -= dx / d * rep; fy[j] -= dy / d * rep;
    }
    for (let e of edges) {
      let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, spr = (d - 0.2) * 0.02;
      fx[e[0]] += dx / d * spr; fy[e[0]] += dy / d * spr; fx[e[1]] -= dx / d * spr; fy[e[1]] -= dy / d * spr;
    }
    for (let i = 0; i < N; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// ---------- interaction ----------

function mousePressed() {
  let gw = canvasWidth - 196;
  if (mouseY < 32 || mouseY > drawHeight || mouseX > gw) return;
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, 0, 36, gw, drawHeight - 44);
    if (dist(mouseX, mouseY, p.x, p.y) <= 15) { focal = (focal === i) ? -1 : i; return; }
  }
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
