// Agent Tool-Use Graph — dependency traversal & parallel execution
// CANVAS_HEIGHT: 520
// Learning objective (Understand): a tool-use graph structures an agent's plan;
// dependency edges decide which tools run in parallel vs. sequentially.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 460;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let runButton, resetButton;
let tools = [], deps = [], levels = [], curLevel = -1, lastStep = 0, running = false, hoverN = -1;
const CAT = { web: '#42a5f5', code: '#ffa726', data: '#66bb6a', out: '#ab47bc', anal: '#26c6da' };

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildPlan();

  runButton = createButton('Run Plan'); runButton.position(10, drawHeight + 8); runButton.mousePressed(startRun);
  resetButton = createButton('Reset'); resetButton.position(86, drawHeight + 8); resetButton.mousePressed(reset);

  describe('An agent\'s tool-use graph for a research-report task. Directed edges are ' +
    'dependencies (a tool can only run once its prerequisites finish). Run Plan executes the ' +
    'graph in dependency order: tools with no unmet prerequisites light up gold and run in ' +
    'parallel, then turn green when done, unlocking the next wave.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  if (running && millis() - lastStep > 1500) { curLevel++; lastStep = millis(); if (curLevel >= levels.length) { running = false; runButton.html('Run Plan'); curLevel = levels.length - 1; } }

  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(18); text('Agent Tool-Use Graph', canvasWidth / 2, 6);
  noStroke(); fill('#555'); textSize(12); text(running || curLevel >= 0 ? (curLevel < levels.length - 1 || running ? 'Wave ' + (curLevel + 1) + ' / ' + levels.length + ' — gold = running in parallel, green = done' : 'Plan complete') : 'Click "Run Plan" to execute in dependency order', canvasWidth / 2, 28);

  hoverN = -1;
  // edges
  for (let e of deps) {
    let a = ptOf(e[0]), b = ptOf(e[1]);
    stroke(120, 120, 120, 150); strokeWeight(1.5); drawArrowSeg(a.x, a.y, b.x, b.y, 24);
  }
  // nodes
  for (let i = 0; i < tools.length; i++) {
    let p = ptOf(i), st = nodeState(i);
    if (dist(mouseX, mouseY, p.x, p.y) <= 22) hoverN = i;
    if (st === 'run') { noStroke(); fill(255, 193, 7, 110); circle(p.x, p.y, 56); }
    stroke(CAT[tools[i].cat]); strokeWeight(3);
    fill(st === 'done' ? '#2e7d32' : st === 'run' ? '#fdd835' : '#eceff1');
    circle(p.x, p.y, 40);
    // label wrapped + centered inside the circle (hyphens become spaces so they wrap)
    noStroke(); fill(st === 'done' ? 'white' : '#222'); textAlign(CENTER, CENTER); textSize(8.5);
    text(tools[i].name.replace(/-/g, ' '), p.x - 18, p.y - 16, 36, 32);
  }
  drawLegend();
  if (hoverN >= 0) drawTip();
}

function ptOf(i) { return { x: margin + 30 + tools[i].x * (canvasWidth - 2 * margin - 60), y: 44 + tools[i].y * (drawHeight - 70) }; }
function nodeState(i) {
  if (curLevel < 0) return 'idle';
  let lvl = tools[i].level;
  if (lvl < curLevel) return 'done';
  if (lvl === curLevel) return running ? 'run' : (curLevel === levels.length - 1 ? 'done' : 'run');
  return 'idle';
}

function drawLegend() {
  noStroke(); textAlign(LEFT, CENTER); textSize(10); let x = margin + 4, y = drawHeight - 12;
  let items = [['web', 'web'], ['code', 'code'], ['data', 'data'], ['anal', 'analysis'], ['out', 'output']];
  let xx = x; for (let [k, lab] of items) { fill(CAT[k]); circle(xx, y, 9); fill('#333'); text(lab, xx + 8, y); xx += textWidth(lab) + 26; }
}

function drawTip() {
  let t = tools[hoverN];
  let lines = [t.full, 'category: ' + t.cat, 'wave: ' + (t.level + 1) + (deps.some(e => e[1] === hoverN) ? '' : ' (no prerequisites)')];
  let w = 200, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

function drawArrowSeg(x1, y1, x2, y2, rad) { let ang = atan2(y2 - y1, x2 - x1), ex = x2 - cos(ang) * rad, ey = y2 - sin(ang) * rad; stroke(drawingContext.strokeStyle); line(x1, y1, ex, ey); push(); translate(ex, ey); rotate(ang); fill(110, 110, 110, 180); noStroke(); triangle(0, 0, -8, 3, -8, -3); pop(); }

// ---------- plan ----------

function startRun() { curLevel = 0; running = true; lastStep = millis(); runButton.html('Running…'); }
function reset() { curLevel = -1; running = false; runButton.html('Run Plan'); }

function buildPlan() {
  // [name, full, cat, x, y]
  tools = [
    { name: 'web-search', full: 'web-search', cat: 'web', x: 0.05, y: 0.15 },
    { name: 'python-exec', full: 'python-exec', cat: 'code', x: 0.05, y: 0.5 },
    { name: 'sql-query', full: 'sql-query', cat: 'data', x: 0.05, y: 0.85 },
    { name: 'retrieve-papers', full: 'retrieve-papers', cat: 'web', x: 0.34, y: 0.15 },
    { name: 'data-analysis', full: 'data-analysis', cat: 'anal', x: 0.34, y: 0.85 },
    { name: 'summarize', full: 'summarize', cat: 'anal', x: 0.6, y: 0.2 },
    { name: 'write-report', full: 'write-report', cat: 'out', x: 0.8, y: 0.5 },
    { name: 'email-send', full: 'email-send', cat: 'out', x: 0.97, y: 0.5 },
  ];
  // deps [from prerequisite -> to dependent]
  deps = [[0, 3], [3, 5], [5, 6], [1, 6], [2, 4], [4, 6], [6, 7]];
  computeLevels();
}

function computeLevels() {
  // longest-path level (topological depth)
  let indeg = tools.map(() => 0); for (let e of deps) indeg[e[1]]++;
  let level = tools.map(() => 0);
  let q = []; for (let i = 0; i < tools.length; i++) if (indeg[i] === 0) q.push(i);
  let processed = [...indeg];
  // Kahn with level = max predecessor level + 1
  let order = []; let deg2 = [...indeg];
  while (q.length) { let u = q.shift(); order.push(u); for (let e of deps) if (e[0] === u) { level[e[1]] = Math.max(level[e[1]], level[u] + 1); if (--deg2[e[1]] === 0) q.push(e[1]); } }
  tools.forEach((t, i) => t.level = level[i]);
  let maxL = Math.max(...level); levels = [];
  for (let l = 0; l <= maxL; l++) levels.push(tools.map((t, i) => i).filter(i => tools[i].level === l));
}

function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
