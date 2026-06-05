// LightGCN Bipartite Propagation Explorer
// CANVAS_HEIGHT: 520
// Learning objective (Apply): multi-hop paths through the user-item bipartite graph
// generate recommendation candidates; more LightGCN layers expand the receptive field
// of collaborative signal.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 460;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let kSlider, weightToggle, resetButton;

const NU = 10, NI = 15;
let adj = [];                 // adjacency over 0..NU+NI-1 (users 0..NU-1, items NU..)
let deg = [];
let queryUser = -1, K = 2, showWeights = false, hops = [], hoverNode = -1;

const C_USER = '#4fc3f7', C_ITEM = '#ff9800', BG = 'aliceblue';

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildGraph();

  kSlider = createSlider(1, 3, 2, 1); kSlider.position(sliderLeftMargin, drawHeight + 18); kSlider.size(220); kSlider.input(() => { K = kSlider.value(); propagate(); });
  weightToggle = createButton('Show edge weights'); weightToggle.position(400, drawHeight + 14); weightToggle.mousePressed(() => { showWeights = !showWeights; });
  resetButton = createButton('Reset'); resetButton.position(540, drawHeight + 14); resetButton.mousePressed(() => { queryUser = -1; hops = []; });

  describe('LightGCN propagation on a user-item bipartite graph. Click a user to trace ' +
    'collaborative signal: 1-hop items they interacted with (orange), 2-hop users with shared ' +
    'items (blue), and 3-hop recommendation candidates (gold). The layer slider K controls how ' +
    'many propagation hops are shown; toggle normalized edge weights 1/√(dᵤ·dᵢ).', LABEL);
}

function draw() {
  updateCanvasSize();
  // dark drawing area
  fill(BG); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);

  let gw = canvasWidth - 290;
  noStroke(); fill('black'); textAlign(CENTER, TOP); textSize(18); text('LightGCN Collaborative Propagation', gw / 2, 8);

  hoverNode = -1;
  drawGraph(0, 34, gw, drawHeight - 42);
  drawPanel(gw, 34, 290 - margin);

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13); text('Layers K = ' + K, 10, drawHeight + 25);
  if (hoverNode >= 0) drawTip();
}

function nodeXY(i, gx, gy, gw, gh) {
  if (i < NU) return { x: gx + 70, y: gy + 20 + (gh - 40) * i / (NU - 1) };
  let k = i - NU; return { x: gx + gw - 70, y: gy + 20 + (gh - 40) * k / (NI - 1) };
}

function hopColor(h) { return h === 1 ? C_ITEM : h === 2 ? C_USER : h === 3 ? '#ffd54f' : null; }

function drawGraph(gx, gy, gw, gh) {
  // edges
  for (let i = 0; i < NU + NI; i++) for (let j of adj[i]) if (i < j) {
    let a = nodeXY(i, gx, gy, gw, gh), b = nodeXY(j, gx, gy, gw, gh);
    let active = queryUser >= 0 && hops[i] >= 0 && hops[j] >= 0 && Math.abs(hops[i] - hops[j]) === 1;
    let w = showWeights ? (1 / Math.sqrt(deg[i] * deg[j])) * 12 : 1;
    stroke(active ? color(33, 33, 33, 210) : color(90, 90, 90, 70)); strokeWeight(active ? Math.max(1.5, w) : Math.max(0.6, w * 0.6));
    line(a.x, a.y, b.x, b.y);
  }
  // nodes
  for (let i = 0; i < NU + NI; i++) {
    let p = nodeXY(i, gx, gy, gw, gh);
    let isUser = i < NU, r = isUser ? 11 : 9;
    if (dist(mouseX, mouseY, p.x, p.y) <= r + 3) hoverNode = i;
    let faded = queryUser >= 0 && (hops[i] < 0 && i !== queryUser);
    let hc = (queryUser >= 0 && hops[i] >= 1 && hops[i] <= K + 1) ? hopColor(hops[i]) : null;
    let base = isUser ? C_USER : C_ITEM;
    let fillC = i === queryUser ? '#ef5350' : (hc || base);
    push(); if (faded) drawingContext.globalAlpha = 0.2;
    stroke(i === queryUser ? 'black' : color(0, 0, 0, 120)); strokeWeight(i === queryUser ? 3 : 1);
    fill(fillC);
    if (isUser) circle(p.x, p.y, r * 2); else { rectMode(CENTER); rect(p.x, p.y, r * 2, r * 2, 3); rectMode(CORNER); }
    pop();
  }
  noStroke(); fill('#444'); textAlign(LEFT, TOP); textSize(11); text('users', gx + 50, gy + gh - 4); textAlign(RIGHT, TOP); text('items', gx + gw - 50, gy + gh - 4);
}

function drawPanel(px, py, pw) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, py, pw, drawHeight - py - 8);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14);
  let tx = px + 12, ty = py + 10, lh = 20;
  if (queryUser < 0) { text('Click a user node', tx, ty); ty += 18; text('to trace propagation.', tx, ty); ty += 26; }
  else {
    text('Query user: ' + queryUser, tx, ty); ty += lh;
    let h1 = countHop(1), h2 = countHop(2), h3 = countHop(3);
    fill('#e65100'); text('1-hop items: ' + h1, tx, ty); ty += lh;
    fill('#0277bd'); text('2-hop users: ' + h2, tx, ty); ty += lh;
    fill('#f9a825'); text('3-hop candidates: ' + h3, tx, ty); ty += lh + 4;
  }
  // popularity bar chart
  fill('black'); textSize(13); text('Item popularity (degree)', tx, ty); ty += 18;
  let items = []; for (let k = 0; k < NI; k++) items.push([k, deg[NU + k]]);
  items.sort((a, b) => b[1] - a[1]);
  let maxD = Math.max(...items.map(x => x[1]), 1);
  for (let [k, d] of items) {
    if (ty > drawHeight - 18) break;
    let inViz = queryUser >= 0 && hops[NU + k] >= 1;
    fill('#555'); textSize(10); text('i' + k, tx, ty);
    stroke('silver'); strokeWeight(1); fill(inViz ? '#f9a825' : '#90caf9'); rect(tx + 22, ty, (d / maxD) * (pw - 60), 8, 2);
    noStroke(); ty += 11;
  }
}

function drawTip() {
  let i = hoverNode, isUser = i < NU;
  let lines = [(isUser ? 'User ' + i : 'Item ' + (i - NU)), 'degree: ' + deg[i]];
  if (!isUser && queryUser >= 0 && hops[i] >= 0) lines.push('hops from user: ' + hops[i]);
  let w = 150, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 30, drawHeight - h - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

function countHop(h) { let c = 0; for (let i = 0; i < NU + NI; i++) if (hops[i] === h) c++; return c; }

// ---------- propagation ----------

function propagate() {
  hops = new Array(NU + NI).fill(-1);
  if (queryUser < 0) return;
  hops[queryUser] = 0; let q = [queryUser];
  while (q.length) { let u = q.shift(); if (hops[u] >= K + 1) continue; for (let v of adj[u]) if (hops[v] < 0) { hops[v] = hops[u] + 1; q.push(v); } }
}

function buildGraph() {
  let rng = mulberry32(99);
  adj = Array.from({ length: NU + NI }, () => []);
  let set = new Set(); let count = 0;
  // ensure each user has >=1 edge
  for (let u = 0; u < NU; u++) { let it = NU + Math.floor(rng() * NI); adj[u].push(it); adj[it].push(u); set.add(u + '_' + it); count++; }
  while (count < 25) { let u = Math.floor(rng() * NU), it = NU + Math.floor(rng() * NI); let k = u + '_' + it; if (set.has(k)) continue; set.add(k); adj[u].push(it); adj[it].push(u); count++; }
  deg = adj.map(a => a.length);
}

function mousePressed() {
  let gw = canvasWidth - 290;
  if (mouseY < 30 || mouseY > drawHeight) return;
  for (let i = 0; i < NU; i++) { let p = nodeXY(i, 0, 34, gw, drawHeight - 42); if (dist(mouseX, mouseY, p.x, p.y) <= 13) { queryUser = i; propagate(); return; } }
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
