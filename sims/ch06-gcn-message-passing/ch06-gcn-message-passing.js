// GCN Message Passing Animation (Karate Club)
// CANVAS_HEIGHT: 520
// Learning objective (Understand, Bloom L2): watch a GCN layer aggregate neighbor
// representations into a node's new embedding, and see the receptive field expand
// across layers (mean aggregation, no learned weights).
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 440;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

let stepButton, resetButton, kSlider;

const N = 34;
let adj = [], deg = [], pos = [];
let h = [], hHist = [];     // current embeddings + history per layer
let layer = 0, K = 3;
let focal = -1;
let animStart = -10000;
const ANIM_MS = 1200;
let hoverNode = -1;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildKarate();

  stepButton = createButton('Step');
  stepButton.position(10, drawHeight + 8);
  stepButton.mousePressed(doStep);

  resetButton = createButton('Reset');
  resetButton.position(64, drawHeight + 8);
  resetButton.mousePressed(resetLayers);

  kSlider = createSlider(1, 3, 3, 1);
  kSlider.position(sliderLeftMargin, drawHeight + 45);
  kSlider.size(canvasWidth - sliderLeftMargin - margin);
  kSlider.input(() => K = kSlider.value());

  resetLayers();
  describe('GCN message passing on the Karate Club graph. Click a focal node, then ' +
    'Step to animate one layer: neighbor messages flow in, get averaged, and update the ' +
    'node embedding (color). Concentric rings show the receptive field growing with depth.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let panelW = 196;
  let gw = canvasWidth - panelW;

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(20);
  text('GCN Message Passing', gw / 2, 6);
  textSize(14); fill('#333');
  text('Layer: ' + layer + ' / ' + K, gw / 2, 30);

  drawGraph(0, 50, gw, drawHeight - 50 - 8);
  drawPanel(gw, 50, panelW);
  if (hoverNode >= 0) drawTooltip();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(defaultTextSize);
  text('Layers K = ' + K, 10, drawHeight + 52);
}

function nodeScreen(i, gx, gy, gw, gh) { return { x: gx + 22 + pos[i].x * (gw - 44), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawGraph(gx, gy, gw, gh) {
  hoverNode = -1;
  // edges
  stroke(205); strokeWeight(1);
  for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) {
    let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh);
    line(a.x, a.y, b.x, b.y);
  }

  // receptive field rings for focal
  let hop = focal >= 0 ? bfsHops(focal) : null;

  // animated messages
  let anim = millis() - animStart < ANIM_MS;
  if (anim && focal >= 0) {
    let t = (millis() - animStart) / ANIM_MS;
    let fp = nodeScreen(focal, gx, gy, gw, gh);
    for (let nb of adj[focal]) {
      let np = nodeScreen(nb, gx, gy, gw, gh);
      let mx = lerp(np.x, fp.x, constrain(t * 1.4, 0, 1));
      let my = lerp(np.y, fp.y, constrain(t * 1.4, 0, 1));
      stroke('#ffb300'); strokeWeight(2); line(np.x, np.y, mx, my);
      noStroke(); fill('#ff8f00'); circle(mx, my, 7);
    }
  }

  // nodes
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    let rad = map(deg[i], 1, Math.max(...deg), 9, 18);
    if (dist(mouseX, mouseY, p.x, p.y) <= rad + 2) hoverNode = i;
    // receptive-field ring
    if (hop && hop[i] >= 1 && hop[i] <= layer) {
      let ringCol = hop[i] === 1 ? '#0d47a1' : hop[i] === 2 ? '#1976d2' : '#64b5f6';
      noStroke(); fill(red(color(ringCol)), green(color(ringCol)), blue(color(ringCol)), 70);
      circle(p.x, p.y, rad * 2 + 12);
    }
    let col = lerpColor(color('#cfcfcf'), color('#1a237e'), constrain(h[i], 0, 1));
    if (i === focal) { stroke('gold'); strokeWeight(4); }
    else { stroke(70); strokeWeight(1); }
    fill(col); circle(p.x, p.y, rad * 2);
    noStroke(); fill(h[i] > 0.5 ? 'white' : 'black'); textAlign(CENTER, CENTER); textSize(9);
    text(nf(h[i], 1, 2), p.x, p.y);
  }
}

function drawPanel(px, py, pw) {
  fill('white'); stroke('silver'); strokeWeight(1); rect(px, 44, pw, drawHeight - 44);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14);
  let tx = px + 10, ty = 54, lh = 20;
  text('Focal node', tx, ty); ty += lh + 2;
  textSize(13);
  if (focal < 0) { text('Click a node to', tx, ty); ty += 18; text('select a focal node.', tx, ty); ty += 26; }
  else {
    text('Node: ' + focal, tx, ty); ty += lh;
    text('Degree: ' + deg[focal], tx, ty); ty += lh;
    text('hᵥ = ' + nf(h[focal], 1, 3), tx, ty); ty += lh + 6;
  }
  // receptive-field legend
  textSize(12); fill('#0d47a1'); text('● 1-hop', tx, ty); ty += 16;
  fill('#1976d2'); text('● 2-hop', tx, ty); ty += 16;
  fill('#64b5f6'); text('● 3-hop', tx, ty); ty += 18;
  fill('dimgray'); textSize(11);
  text('color = hᵥ (gray→indigo).', tx, ty); ty += 14;
  text('Each layer averages', tx, ty); ty += 14;
  text('neighbors → smoothing.', tx, ty);
}

function drawTooltip() {
  let i = hoverNode;
  let nbrs = adj[i].slice(0, 8).join(',') + (adj[i].length > 8 ? '…' : '');
  let lines = ['Node ' + i, 'hᵥ = ' + nf(h[i], 1, 3), 'Neighbors: ' + nbrs];
  let w = 200, hh = lines.length * 18 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 50, drawHeight - hh - 4);
  fill(255, 255, 255, 245); stroke(100); strokeWeight(1); rect(tx, ty, w, hh, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 18);
}

// ---------- GCN mean aggregation ----------

function resetLayers() {
  h = deg.map(d => d / Math.max(...deg));   // h^(0) = normalized degree
  hHist = [h.slice()];
  layer = 0;
}

function doStep() {
  if (layer >= K) return;
  let nh = new Array(N);
  for (let v = 0; v < N; v++) {
    let s = h[v], cnt = 1;
    for (let u of adj[v]) { s += h[u]; cnt++; }
    nh[v] = s / cnt;   // mean over N(v) ∪ {v}
  }
  h = nh; hHist.push(h.slice()); layer++;
  if (focal >= 0) animStart = millis();
}

function bfsHops(s) {
  let d = new Array(N).fill(-1); d[s] = 0; let q = [s];
  while (q.length) { let u = q.shift(); for (let v of adj[u]) if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); } }
  return d;
}

// ---------- interaction ----------

function mousePressed() {
  let gw = canvasWidth - 196;
  if (mouseY < 46 || mouseY > drawHeight || mouseX > gw) return;
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, 0, 50, gw, drawHeight - 50 - 8);
    let rad = map(deg[i], 1, Math.max(...deg), 9, 18);
    if (dist(mouseX, mouseY, p.x, p.y) <= rad + 2) { focal = (focal === i) ? -1 : i; return; }
  }
  focal = -1;
}

// ---------- Karate graph ----------

function buildKarate() {
  let edges = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],[14,33],
    [15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],[23,33],
    [24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  adj = Array.from({ length: N }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  deg = adj.map(a => a.length);
  pos = springLayout(edges);
}

function springLayout(edges) {
  let p = [];
  for (let i = 0; i < N; i++) { let a = TWO_PI * i / N; p.push({ x: 0.5 + 0.4 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a) }); }
  for (let it = 0; it < 250; it++) {
    let fx = new Array(N).fill(0), fy = new Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), rep = 0.0008 / d2;
      fx[i] += dx / d * rep; fy[i] += dy / d * rep; fx[j] -= dx / d * rep; fy[j] -= dy / d * rep;
    }
    for (let e of edges) {
      let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, spr = (d - 0.12) * 0.02;
      fx[e[0]] += dx / d * spr; fy[e[0]] += dy / d * spr; fx[e[1]] -= dx / d * spr; fy[e[1]] -= dy / d * spr;
    }
    for (let i = 0; i < N; i++) { p[i].x += constrain(fx[i], -0.02, 0.02); p[i].y += constrain(fy[i], -0.02, 0.02); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}

// ---------- responsive ----------

function windowResized() {
  updateCanvasSize(); resizeCanvas(containerWidth, containerHeight);
  kSlider.size(canvasWidth - sliderLeftMargin - margin); redraw();
}
function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width); canvasWidth = containerWidth;
}
