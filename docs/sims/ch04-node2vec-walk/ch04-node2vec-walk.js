// node2vec Biased Random Walk Explorer
// CANVAS_HEIGHT: 560
// Learning objective (Apply, Bloom L2-3): manipulate the return parameter p and
// in-out parameter q and watch walks shift between BFS-like (local) and DFS-like
// (global) exploration on the Karate Club graph.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 430;
let controlHeight = 130;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 210;
let defaultTextSize = 16;

// controls
let stepButton, autoButton, resetButton, startSelect, pSlider, qSlider, lenSlider;

// graph
let pos = [];     // {x,y} fixed layout in panel-relative [0,1]
let adj = [];     // neighbor lists
let faction = []; // 0 or 1
let N = 34;

// walk
let walk = [];        // sequence of node ids
let walkSet = {};     // visited set
let startNode = 0;
let pParam = 1.0, qParam = 1.0, walkLen = 30;
let autoOn = false, lastStep = 0;
let candHighlight = null;  // {cands:[{node,d,prob}]}

const STEEL = '#4682b4', CRIMSON = '#dc143c';

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  frameRate(12);

  buildKarate();

  stepButton = createButton('Step');
  stepButton.position(10, drawHeight + 8);
  stepButton.mousePressed(() => { autoOn = false; autoButton.html('Auto Walk'); walkStep(); });

  autoButton = createButton('Auto Walk');
  autoButton.position(64, drawHeight + 8);
  autoButton.mousePressed(toggleAuto);

  resetButton = createButton('Reset');
  resetButton.position(150, drawHeight + 8);
  resetButton.mousePressed(() => { autoOn = false; autoButton.html('Auto Walk'); resetWalk(Math.floor(Math.random() * N)); });

  startSelect = createSelect();
  startSelect.position(214, drawHeight + 8);
  for (let i = 0; i < N; i++) startSelect.option('Start: ' + i);
  startSelect.changed(() => { autoOn = false; autoButton.html('Auto Walk'); resetWalk(parseInt(startSelect.value().split(' ')[1])); });

  lenSlider = createSlider(5, 80, 30, 5); lenSlider.position(sliderLeftMargin, drawHeight + 40); lenSlider.input(() => walkLen = lenSlider.value());
  pSlider = createSlider(0.1, 4.0, 1.0, 0.1); pSlider.position(sliderLeftMargin, drawHeight + 72); pSlider.input(() => pParam = pSlider.value());
  qSlider = createSlider(0.1, 4.0, 1.0, 0.1); qSlider.position(sliderLeftMargin, drawHeight + 104); qSlider.input(() => qParam = qSlider.value());
  sizeSliders();

  resetWalk(0);
  describe('node2vec biased random walk on the Zachary Karate Club graph. Adjust the ' +
    'return parameter p and in-out parameter q and step the walk to see it shift between ' +
    'local BFS-like and global DFS-like exploration. Click a node to start there.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  if (autoOn && walk.length < walkLen && millis() - lastStep > 330) { walkStep(); lastStep = millis(); }
  if (autoOn && walk.length >= walkLen) { autoOn = false; autoButton.html('Auto Walk'); }

  let gw = canvasWidth * 0.68;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(20);
  text('node2vec Biased Random Walk', gw / 2, 6);

  drawGraph(0, 34, gw, drawHeight - 44);
  drawStats(gw + 12, 40, canvasWidth - gw - 12 - margin);
  drawControlLabels();
}

// ---------- rendering ----------

function nodeScreen(i, gx, gy, gw, gh) {
  return { x: gx + 26 + pos[i].x * (gw - 52), y: gy + 20 + pos[i].y * (gh - 40) };
}

function drawGraph(gx, gy, gw, gh) {
  // edges
  stroke(210); strokeWeight(1);
  for (let i = 0; i < N; i++) for (let j of adj[i]) if (i < j) {
    let a = nodeScreen(i, gx, gy, gw, gh), b = nodeScreen(j, gx, gy, gw, gh);
    line(a.x, a.y, b.x, b.y);
  }

  // walk trace (gradient indigo -> gold)
  strokeWeight(3);
  for (let k = 0; k < walk.length - 1; k++) {
    let a = nodeScreen(walk[k], gx, gy, gw, gh), b = nodeScreen(walk[k + 1], gx, gy, gw, gh);
    let t = walk.length > 1 ? k / (walk.length - 1) : 0;
    stroke(lerpColor(color('#1a237e'), color('#ffd700'), t));
    line(a.x, a.y, b.x, b.y);
  }

  // candidate highlights for next step
  let cur = walk.length ? walk[walk.length - 1] : -1;
  let prev = walk.length > 1 ? walk[walk.length - 2] : -1;
  if (candHighlight) {
    for (let c of candHighlight) {
      let a = nodeScreen(cur, gx, gy, gw, gh), b = nodeScreen(c.node, gx, gy, gw, gh);
      let col = c.d === 0 ? '#1565c0' : c.d === 1 ? '#2e7d32' : '#e65100';
      stroke(col); strokeWeight(1 + c.prob * 10);
      line(a.x, a.y, b.x, b.y);
    }
  }

  // nodes
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    let base = faction[i] === 0 ? STEEL : CRIMSON;
    let inWalk = walkSet[i];
    if (i === prev) { noStroke(); fill(255, 165, 0, 120); circle(p.x, p.y, 28); }
    if (i === cur) { noStroke(); fill(255, 215, 0, 110); circle(p.x, p.y, 30); }
    stroke(60); strokeWeight(1);
    fill(i === cur ? '#ffd700' : (inWalk ? '#ffe082' : base));
    let rad = 9;
    circle(p.x, p.y, rad * 2);
    if (i === startNode) { noStroke(); fill('black'); textAlign(CENTER, CENTER); textSize(9); text('S', p.x, p.y); }
  }
}

function drawStats(px, py, pw) {
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(14);
  let ty = py, lh = 22;
  text('Walk statistics', px, ty); ty += lh + 2;
  textSize(13);
  text('Step: ' + Math.max(0, walk.length - 1) + ' / ' + walkLen, px, ty); ty += lh;

  // fraction same faction
  let same = 0;
  for (let v of walk) if (faction[v] === faction[startNode]) same++;
  let frac = walk.length ? Math.round(100 * same / walk.length) : 0;
  text('Same faction: ' + frac + '%', px, ty); ty += lh;

  // farthest hop
  let far = farthestHop();
  text('Farthest reached: ' + far + ' hops', px, ty); ty += lh;

  // mode
  let mode = (pParam < 1 && qParam > 1) ? 'BFS-biased (local)'
    : (pParam > 1 && qParam < 1) ? 'DFS-biased (global)' : 'Balanced';
  text('Walk mode: ' + mode, px, ty); ty += lh + 6;

  // legend
  textSize(12); fill('#1565c0'); text('■ return (d=0)', px, ty); ty += 16;
  fill('#2e7d32'); text('■ shared neighbor (d=1)', px, ty); ty += 16;
  fill('#e65100'); text('■ outward (d=2)', px, ty); ty += 16;
  fill('dimgray'); text('arrow width = transition prob', px, ty);
}

function drawControlLabels() {
  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13);
  text('Length L = ' + walkLen, 10, drawHeight + 47);
  text('Return p = ' + nf(pParam, 1, 1), 10, drawHeight + 79);
  text('In-out q = ' + nf(qParam, 1, 1), 10, drawHeight + 111);
}

// ---------- node2vec walk ----------

function transitionWeights(prev, cur) {
  // returns array of {node, d, weight}
  let res = [];
  for (let nb of adj[cur]) {
    let d;
    if (prev < 0) d = 1;
    else if (nb === prev) d = 0;            // return
    else if (adj[prev].includes(nb)) d = 1; // shared neighbor (distance 1)
    else d = 2;                             // outward
    let w = d === 0 ? 1 / pParam : d === 1 ? 1 : 1 / qParam;
    res.push({ node: nb, d, weight: w });
  }
  return res;
}

function walkStep() {
  if (walk.length >= walkLen) return;
  let cur = walk[walk.length - 1];
  let prev = walk.length > 1 ? walk[walk.length - 2] : -1;
  if (adj[cur].length === 0) return;
  let w = transitionWeights(prev, cur);
  let total = w.reduce((s, x) => s + x.weight, 0);
  w.forEach(x => x.prob = x.weight / total);
  candHighlight = w;
  // weighted random choice
  let r = Math.random() * total, acc = 0, chosen = w[w.length - 1].node;
  for (let x of w) { acc += x.weight; if (r <= acc) { chosen = x.node; break; } }
  walk.push(chosen); walkSet[chosen] = true;
}

function resetWalk(start) {
  startNode = start;
  walk = [start]; walkSet = {}; walkSet[start] = true;
  candHighlight = null;
  if (startSelect) startSelect.selected('Start: ' + start);
}

function farthestHop() {
  // BFS distances from startNode, max among visited
  let dist = new Array(N).fill(-1); dist[startNode] = 0;
  let q = [startNode];
  while (q.length) { let u = q.shift(); for (let v of adj[u]) if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push(v); } }
  let m = 0; for (let v of walk) m = Math.max(m, dist[v]);
  return m;
}

// ---------- interaction ----------

function toggleAuto() {
  if (walk.length >= walkLen) return;
  autoOn = !autoOn; autoButton.html(autoOn ? 'Pause' : 'Auto Walk'); lastStep = 0;
}

function mousePressed() {
  let gw = canvasWidth * 0.68;
  if (mouseY < 30 || mouseY > drawHeight || mouseX > gw) return;
  for (let i = 0; i < N; i++) {
    let p = nodeScreen(i, 0, 34, gw, drawHeight - 44);
    if (dist(mouseX, mouseY, p.x, p.y) <= 11) { autoOn = false; autoButton.html('Auto Walk'); resetWalk(i); return; }
  }
}

// ---------- Karate Club graph ----------

function buildKarate() {
  let edges = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],[14,33],
    [15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],[23,33],
    [24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  // faction (Mr Hi group 0 vs Officer group 1) - standard split
  let officer = new Set([8,9,14,15,18,20,22,23,24,25,26,27,28,29,30,31,32,33]);
  faction = []; for (let i = 0; i < N; i++) faction.push(officer.has(i) ? 1 : 0);
  adj = Array.from({ length: N }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  pos = circleLayoutByFaction();
}

function circleLayoutByFaction() {
  // two clusters: faction 0 on left, faction 1 on right, ring within each
  let p = new Array(N);
  let g0 = [], g1 = [];
  for (let i = 0; i < N; i++) (faction[i] === 0 ? g0 : g1).push(i);
  placeCluster(g0, 0.28, 0.5, 0.22, p);
  placeCluster(g1, 0.72, 0.5, 0.22, p);
  return p;
}
function placeCluster(ids, cx, cy, r, p) {
  for (let k = 0; k < ids.length; k++) {
    let a = TWO_PI * k / ids.length - HALF_PI;
    // two concentric rings to reduce overlap
    let rr = r * (k % 2 === 0 ? 1.0 : 0.55);
    p[ids[k]] = { x: cx + rr * Math.cos(a), y: cy + rr * Math.sin(a) };
  }
}

// ---------- responsive ----------

function sizeSliders() {
  let w = canvasWidth - sliderLeftMargin - margin;
  [pSlider, qSlider, lenSlider].forEach(s => s && s.size(w));
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  sizeSliders();
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
