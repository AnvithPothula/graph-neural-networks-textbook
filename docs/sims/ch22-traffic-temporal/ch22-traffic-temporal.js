// Temporal Graph — Traffic Sensor Network
// CANVAS_HEIGHT: 520
// Learning objective (Understand): spatial dependencies (neighboring sensors show
// correlated speed drops) and temporal dependencies (predictable rush hours) motivate
// joint spatio-temporal modeling (STGCN, DCRNN).
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

let playButton, resetButton, speedSlider;

const NN = 15, T = 120;
let pos = [], adj = [], speedData = [];   // speedData[node][t]
let curT = 0, playing = false, selNode = 0, animSpeed = 3, frameAcc = 0, hoverNode = -1;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildNetwork();

  playButton = createButton('Play'); playButton.position(10, drawHeight + 8); playButton.mousePressed(() => { playing = !playing; playButton.html(playing ? 'Pause' : 'Play'); });
  resetButton = createButton('Reset'); resetButton.position(70, drawHeight + 8); resetButton.mousePressed(() => { curT = 0; playing = false; playButton.html('Play'); });
  speedSlider = createSlider(1, 10, 3, 1); speedSlider.position(sliderLeftMargin, drawHeight + 45); speedSlider.size(sizeSpeedSlider()); speedSlider.input(() => animSpeed = speedSlider.value());

  describe('A road sensor network evolving over time. Left: 15 sensors colored by current ' +
    'traffic speed (green fast → red congested) on a small road graph. Right: the speed time ' +
    'series for the selected sensor across a simulated day with morning and evening rush hours. ' +
    'Play the time scrubber and click sensors to compare; neighboring sensors drop together ' +
    '(spatial) and rushes recur (temporal).', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  if (playing) { frameAcc += animSpeed; if (frameAcc >= 10) { curT = (curT + 1) % T; frameAcc = 0; } }

  let half = canvasWidth / 2;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(15);
  text('Road sensor network', half / 2, 6); text('Speed time series — node ' + selNode, half + half / 2, 6);
  stroke('silver'); line(half, 26, half, drawHeight - 8);

  hoverNode = -1;
  drawNetwork(0, 30, half, drawHeight - 38);
  drawSeries(half, 30, half, drawHeight - 38);
  if (hoverNode >= 0) drawTip();

  fill('black'); noStroke(); textAlign(LEFT, CENTER); textSize(13);
  text('Speed ×' + animSpeed, 10, drawHeight + 52);
  // time readout: right end of the control row, right-aligned (slider is capped so this stays clear)
  let hr = Math.floor(curT * 5 / 60), mn = (curT * 5) % 60;
  fill('#1565c0'); textAlign(RIGHT, CENTER); textSize(13);
  text('time ' + nf(7 + hr, 2) + ':' + nf(mn, 2) + '  (t=' + curT + ')', canvasWidth - margin, drawHeight + 52);
}

function speedColor(s) {
  if (s >= 50) return color('#43a047'); if (s >= 35) return color('#1e88e5'); if (s >= 20) return color('#fdd835'); return color('#e53935');
}
function nodeXY(i, gx, gy, gw, gh) { return { x: gx + 24 + pos[i].x * (gw - 48), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawNetwork(gx, gy, gw, gh) {
  stroke(170); strokeWeight(2);
  for (let i = 0; i < NN; i++) for (let j of adj[i]) if (i < j) { let a = nodeXY(i, gx, gy, gw, gh), b = nodeXY(j, gx, gy, gw, gh); if (distToSeg(mouseX, mouseY, a.x, a.y, b.x, b.y) < 5) hoverNode = -2, hoverEdge = [i, j]; line(a.x, a.y, b.x, b.y); }
  for (let i = 0; i < NN; i++) {
    let p = nodeXY(i, gx, gy, gw, gh), s = speedData[i][curT];
    if (dist(mouseX, mouseY, p.x, p.y) <= 12) hoverNode = i;
    stroke(i === selNode ? 'black' : 60); strokeWeight(i === selNode ? 3 : 1);
    fill(speedColor(s)); circle(p.x, p.y, 20);
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(9); text(i, p.x, p.y);
  }
  // legend
  noStroke(); textAlign(LEFT, CENTER); textSize(10);
  let lx = gx + 6, ly = gy + gh - 6;
  fill('#43a047'); text('● fast', lx, ly); fill('#1e88e5'); text('● normal', lx + 42, ly); fill('#fdd835'); text('● slow', lx + 100, ly); fill('#e53935'); text('● congested', lx + 142, ly);
}

let hoverEdge = null;
function drawSeries(gx, gy, gw, gh) {
  let x0 = gx + 34, x1 = gx + gw - 14, y0 = gy + gh - 24, y1 = gy + 12;
  stroke(200); strokeWeight(1); line(x0, y0, x1, y0); line(x0, y0, x0, y1);
  noStroke(); fill('dimgray'); textSize(10); textAlign(RIGHT, CENTER); text('70', x0 - 4, y1); text('0', x0 - 4, y0); textAlign(CENTER, TOP); text('time (5-min steps, 10 hours)', (x0 + x1) / 2, y0 + 6);
  // rush-hour shading
  noStroke(); fill(229, 57, 53, 25);
  rect(map(30, 0, T, x0, x1), y1, map(20, 0, T, 0, x1 - x0), y0 - y1); rect(map(90, 0, T, x0, x1), y1, map(20, 0, T, 0, x1 - x0), y0 - y1);
  // series line
  stroke('#1565c0'); strokeWeight(2); noFill(); beginShape();
  for (let t = 0; t < T; t++) vertex(map(t, 0, T - 1, x0, x1), map(speedData[selNode][t], 0, 70, y0, y1));
  endShape();
  // current-time marker
  let cx = map(curT, 0, T - 1, x0, x1);
  stroke('#e53935'); strokeWeight(1.5); drawingContext.setLineDash([4, 3]); line(cx, y1, cx, y0); drawingContext.setLineDash([]);
  let cy = map(speedData[selNode][curT], 0, 70, y0, y1); noStroke(); fill('#e53935'); circle(cx, cy, 7);
  fill('black'); textAlign(LEFT, TOP); textSize(12); text('current: ' + Math.round(speedData[selNode][curT]) + ' mph', x0 + 4, y1);
}

function drawTip() {
  let i = hoverNode, s = speedData[i][curT];
  let prev = speedData[i][(curT - 1 + T) % T]; let trend = s > prev + 1 ? '↑' : s < prev - 1 ? '↓' : '→';
  let lines = ['Node ' + i, 'speed: ' + Math.round(s) + ' mph', 'trend: ' + trend];
  let w = 130, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

function distToSeg(px, py, x1, y1, x2, y2) { let dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy; let t = l2 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l2)) : 0; return dist(px, py, x1 + t * dx, y1 + t * dy); }

function buildNetwork() {
  let rng = mulberry32(11);
  // 3x5 grid with offsets
  pos = []; for (let r = 0; r < 3; r++) for (let c = 0; c < 5; c++) pos.push({ x: 0.08 + c * 0.21 + (rng() - 0.5) * 0.05, y: 0.15 + r * 0.34 + (rng() - 0.5) * 0.05 });
  adj = Array.from({ length: NN }, () => []);
  const add = (a, b) => { if (!adj[a].includes(b)) { adj[a].push(b); adj[b].push(a); } };
  for (let r = 0; r < 3; r++) for (let c = 0; c < 5; c++) { let i = r * 5 + c; if (c < 4) add(i, i + 1); if (r < 2) add(i, i + 5); }
  // synthetic speed data: base 55, morning rush 30-50, evening 90-110, with spatial correlation by row
  speedData = [];
  for (let i = 0; i < NN; i++) {
    let row = Math.floor(i / 5), phase = row * 0.4 + (i % 5) * 0.1;
    let arr = [];
    for (let t = 0; t < T; t++) {
      let s = 58;
      s -= rushDrop(t, 30, 50, 32 + row * 4);     // morning
      s -= rushDrop(t, 90, 110, 28 + row * 4);    // evening
      s += Math.sin(t * 0.3 + phase) * 3 + (rng() - 0.5) * 5;
      arr.push(constrain(s, 6, 68));
    }
    speedData.push(arr);
  }
}
function rushDrop(t, a, b, depth) { let mid = (a + b) / 2, w = (b - a) / 2; if (t < a - w || t > b + w) return 0; return depth * Math.exp(-((t - mid) ** 2) / (2 * (w * 0.7) ** 2)); }

function mousePressed() {
  let half = canvasWidth / 2;
  if (mouseX > half || mouseY < 28 || mouseY > drawHeight) return;
  for (let i = 0; i < NN; i++) { let p = nodeXY(i, 0, 30, half, drawHeight - 38); if (dist(mouseX, mouseY, p.x, p.y) <= 12) { selNode = i; return; } }
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function sizeSpeedSlider() { return Math.max(120, Math.min(360, canvasWidth - sliderLeftMargin - margin)); }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); speedSlider.size(sizeSpeedSlider()); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
