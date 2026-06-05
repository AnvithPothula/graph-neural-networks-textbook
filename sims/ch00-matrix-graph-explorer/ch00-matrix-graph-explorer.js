// Matrix x Graph Explorer - Interactive A.x Matrix Multiplication Visualizer
// CANVAS_HEIGHT: 570
// Learning objective (Understand, Bloom L2): observe how raw, mean, and
// symmetric normalizations of the adjacency matrix produce different
// aggregated features, building intuition for why GCN uses D^(-1/2) A D^(-1/2).
// MicroSim template version 2026.03

// ----- responsive canvas globals -----
let containerWidth;
let canvasWidth = 900;
let drawHeight = 490;     // top drawing region
let controlHeight = 80;   // two rows of controls
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let sliderLeftMargin = 290;
let defaultTextSize = 16;
const BAR_RESERVE = 78;   // px reserved at right of graph region for each node's output bar

// ----- controls -----
let rawButton, meanButton, symButton;
let presetSelect;
let featSlider;

// ----- model state -----
let normMode = 'raw';     // 'raw' | 'mean' | 'sym'
let nodes = [];           // {rx, ry, feat}  rx,ry in [0,1]
let adj = [];             // n x n adjacency (0/1)
let selected = -1;        // selected node index, -1 = none

// colors
const INDIGO = '#3949ab';
const INDIGO_DARK = '#1a237e';

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  // Row 1: normalization mode buttons + preset select
  rawButton = createButton('Raw  A·x');
  rawButton.position(10, drawHeight + 8);
  rawButton.mousePressed(() => { normMode = 'raw'; });

  meanButton = createButton('Mean  D⁻¹A·x');
  meanButton.position(95, drawHeight + 8);
  meanButton.mousePressed(() => { normMode = 'mean'; });

  symButton = createButton('Symmetric  D⁻½AD⁻½x');
  symButton.position(215, drawHeight + 8);
  symButton.mousePressed(() => { normMode = 'sym'; });

  presetSelect = createSelect();
  presetSelect.position(440, drawHeight + 8);
  presetSelect.option('Hub-Spoke (default)');
  presetSelect.option('Path Graph (n=5)');
  presetSelect.option('Complete Graph K4');
  presetSelect.option('Karate Club Sample (10)');
  presetSelect.changed(loadPreset);

  // Row 2: selected-node feature slider
  featSlider = createSlider(0.1, 2.0, 1.0, 0.1);
  featSlider.position(sliderLeftMargin, drawHeight + 45);
  featSlider.size(canvasWidth - sliderLeftMargin - margin);
  featSlider.input(() => {
    if (selected >= 0) nodes[selected].feat = featSlider.value();
  });

  loadPreset();
  describe('Interactive visualization comparing raw, mean, and symmetric ' +
    'normalization of adjacency-matrix aggregation on a small graph. Click a ' +
    'node to inspect its computation; use the slider to change its input feature.',
    LABEL);
}

function draw() {
  updateCanvasSize();

  // backgrounds
  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  // title
  fill('black');
  noStroke();
  textAlign(CENTER, TOP);
  textSize(22);
  text('Matrix × Graph Explorer', canvasWidth / 2, 8);

  // layout regions
  let graphX = margin;
  let graphY = 44;
  let graphW = canvasWidth * 0.50 - margin;
  let graphH = drawHeight - graphY - 40;

  let rightX = canvasWidth * 0.52;
  let rightW = canvasWidth - rightX - margin;

  highlightActiveModeButton();

  let outputs = computeAllOutputs();
  let maxOut = 0.001;
  for (let v of outputs) maxOut = max(maxOut, v);

  drawGraph(graphX, graphY, graphW, graphH, outputs, maxOut);
  drawMatrix(rightX, graphY, rightW);
  drawInfoPanel(rightX, graphY + min(rightW, 220) + 50, rightW, outputs);

  // control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  let featTxt = selected >= 0
    ? 'Node ' + selected + ' input xᵢ: ' + nf(nodes[selected].feat, 1, 1)
    : 'Selected node xᵢ: (click a node)';
  text(featTxt, 10, drawHeight + 52);
}

// ---------- drawing helpers ----------

function nodeScreen(i, gx, gy, gw, gh) {
  return {
    x: gx + nodes[i].rx * (gw - BAR_RESERVE),
    y: gy + 6 + nodes[i].ry * (gh - 40)
  };
}

function drawGraph(gx, gy, gw, gh, outputs, maxOut) {
  let n = nodes.length;

  // edges
  stroke(170);
  strokeWeight(2);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (adj[i][j]) {
        let a = nodeScreen(i, gx, gy, gw, gh);
        let b = nodeScreen(j, gx, gy, gw, gh);
        line(a.x, a.y, b.x, b.y);
      }
    }
  }

  // neighbors of selected highlighted
  let nbrs = {};
  if (selected >= 0) {
    for (let j = 0; j < n; j++) if (adj[selected][j]) nbrs[j] = true;
  }

  // nodes + output bars
  let r = 20;
  for (let i = 0; i < n; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);

    // output bar to the right of the node
    let frac = constrain(outputs[i] / maxOut, 0, 1);
    let barW = 46 * frac + 2;
    let barH = 12;
    let bx = p.x + r + 4;
    let by = p.y - barH / 2;
    noStroke();
    fill(lerpColor(color('white'), color(INDIGO_DARK), frac));
    stroke('silver');
    strokeWeight(1);
    rect(bx, by, barW, barH, 2);

    // node circle
    if (i === selected) {
      fill('gold'); stroke(INDIGO_DARK); strokeWeight(3);
    } else if (nbrs[i]) {
      fill('khaki'); stroke(INDIGO); strokeWeight(2);
    } else {
      fill(INDIGO); stroke(INDIGO_DARK); strokeWeight(2);
    }
    circle(p.x, p.y, r * 2);

    // feature label inside node
    noStroke();
    fill(i === selected || nbrs[i] ? 'black' : 'white');
    textAlign(CENTER, CENTER);
    textSize(13);
    text(nf(nodes[i].feat, 1, 1), p.x, p.y);

    // node id below
    fill('dimgray');
    textSize(11);
    text('n' + i, p.x, p.y + r + 9);
  }

  // legend
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(12);
  text('inside node = input xᵢ   •   bar = output magnitude', gx, gy + gh + 2);
}

function drawMatrix(rx, ry, rw) {
  let n = nodes.length;
  let area = min(rw, 220);
  let cell = (area - 24) / n;
  let ox = rx + 22;
  let oy = ry + 56;

  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(14);
  text('Adjacency matrix A', rx, ry);
  fill('dimgray');
  textSize(12);
  text('(click a cell to toggle an edge)', rx, ry + 15);

  textAlign(CENTER, CENTER);
  textSize(11);
  for (let i = 0; i < n; i++) {
    // row/col headers
    fill('dimgray'); noStroke();
    text(i, ox + i * cell + cell / 2, oy - 9);
    text(i, ox - 11, oy + i * cell + cell / 2);
    for (let j = 0; j < n; j++) {
      if (i === j) {
        fill('gainsboro');
      } else if (adj[i][j]) {
        fill(INDIGO);
      } else {
        fill('whitesmoke');
      }
      stroke('silver');
      strokeWeight(1);
      rect(ox + j * cell, oy + i * cell, cell, cell);
    }
  }
}

function drawInfoPanel(px, py, pw, outputs) {
  let ph = drawHeight - py - 14;
  fill(255, 255, 255, 235);
  stroke(200);
  strokeWeight(1);
  rect(px, py, pw, ph, 10);

  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(14);

  let modeName = normMode === 'raw' ? 'Raw  (A·x)'
    : normMode === 'mean' ? 'Mean  (D⁻¹A·x)'
    : 'Symmetric  (D⁻½AD⁻½x)';

  let tx = px + 12;
  let ty = py + 10;
  let lh = 20;

  text('Mode: ' + modeName, tx, ty); ty += lh + 4;

  if (selected < 0) {
    text('Click a node to see its', tx, ty); ty += lh;
    text('aggregation computed three ways.', tx, ty);
    return;
  }

  let i = selected;
  let nbrs = [];
  for (let j = 0; j < nodes.length; j++) if (adj[i][j]) nbrs.push(j);
  let di = nbrs.length;

  text('Node ' + i + '  (degree dᵢ = ' + di + ')', tx, ty); ty += lh;

  if (di === 0) {
    text('No neighbors → output = 0', tx, ty);
    return;
  }

  // raw
  let sumStr = nbrs.map(j => 'x' + j).join(' + ');
  let sumVal = nbrs.reduce((s, j) => s + nodes[j].feat, 0);
  textSize(13);
  text('Raw = ' + sumStr, tx, ty); ty += lh - 2;
  text('    = ' + nf(sumVal, 1, 2), tx, ty); ty += lh + 2;

  // mean
  text('Mean = Raw / dᵢ = ' + nf(sumVal, 1, 2) + ' / ' + di, tx, ty); ty += lh - 2;
  text('     = ' + nf(sumVal / di, 1, 2), tx, ty); ty += lh + 2;

  // symmetric
  let symVal = nbrs.reduce((s, j) => {
    let dj = degree(j);
    return s + nodes[j].feat / Math.sqrt(di * dj);
  }, 0);
  text('Sym = Σ xⱼ / √(dᵢ·dⱼ)', tx, ty); ty += lh - 2;
  text('    = ' + nf(symVal, 1, 2), tx, ty);
}

// ---------- math ----------

function degree(i) {
  let d = 0;
  for (let j = 0; j < nodes.length; j++) d += adj[i][j];
  return d;
}

function computeAllOutputs() {
  let n = nodes.length;
  let out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let di = degree(i);
    let s = 0;
    for (let j = 0; j < n; j++) {
      if (adj[i][j]) {
        if (normMode === 'raw') s += nodes[j].feat;
        else if (normMode === 'mean') s += nodes[j].feat / max(di, 1);
        else s += nodes[j].feat / Math.sqrt(max(di, 1) * max(degree(j), 1));
      }
    }
    out[i] = s;
  }
  return out;
}

// ---------- presets ----------

function loadPreset() {
  let name = presetSelect ? presetSelect.value() : 'Hub-Spoke (default)';
  selected = -1;
  nodes = [];
  if (name.startsWith('Hub-Spoke')) {
    nodes.push({ rx: 0.5, ry: 0.5, feat: 1.5 });   // hub
    let leaves = 4;
    for (let k = 0; k < leaves; k++) {
      let ang = TWO_PI * k / leaves - HALF_PI;
      nodes.push({ rx: 0.5 + 0.42 * cos(ang), ry: 0.5 + 0.42 * sin(ang), feat: 1.0 });
    }
    buildEmptyAdj();
    for (let k = 1; k <= leaves; k++) addEdge(0, k);
  } else if (name.startsWith('Path')) {
    let n = 5;
    for (let k = 0; k < n; k++) {
      nodes.push({ rx: 0.1 + 0.8 * k / (n - 1), ry: 0.5, feat: 1.0 });
    }
    buildEmptyAdj();
    for (let k = 0; k < n - 1; k++) addEdge(k, k + 1);
  } else if (name.startsWith('Complete')) {
    let n = 4;
    let pos = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]];
    for (let k = 0; k < n; k++) nodes.push({ rx: pos[k][0], ry: pos[k][1], feat: 1.0 });
    buildEmptyAdj();
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) addEdge(i, j);
  } else {
    // Karate Club sample (10 nodes) - small representative subgraph
    let n = 10;
    for (let k = 0; k < n; k++) {
      let ang = TWO_PI * k / n - HALF_PI;
      nodes.push({ rx: 0.5 + 0.4 * cos(ang), ry: 0.5 + 0.4 * sin(ang), feat: 1.0 });
    }
    buildEmptyAdj();
    let edges = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
                 [2, 3], [4, 5], [5, 6], [5, 7], [6, 7], [0, 5], [8, 9], [2, 8]];
    for (let e of edges) addEdge(e[0], e[1]);
  }
  randomizeFeatures();
}

function buildEmptyAdj() {
  let n = nodes.length;
  adj = [];
  for (let i = 0; i < n; i++) adj.push(new Array(n).fill(0));
}

function addEdge(i, j) { adj[i][j] = 1; adj[j][i] = 1; }

function randomizeFeatures() {
  for (let nd of nodes) nd.feat = round((0.5 + Math.random() * 1.3) * 10) / 10;
}

// ---------- interaction ----------

function mousePressed() {
  // node hit test (graph region)
  let gx = margin, gy = 44;
  let gw = canvasWidth * 0.50 - margin;
  let gh = drawHeight - gy - 40;
  let r = 20;
  for (let i = 0; i < nodes.length; i++) {
    let p = nodeScreen(i, gx, gy, gw, gh);
    if (dist(mouseX, mouseY, p.x, p.y) <= r) {
      selected = i;
      if (featSlider) featSlider.value(nodes[i].feat);
      return;
    }
  }

  // matrix cell hit test
  let rx = canvasWidth * 0.52;
  let rw = canvasWidth - rx - margin;
  let n = nodes.length;
  let area = min(rw, 220);
  let cell = (area - 24) / n;
  let ox = rx + 22, oy = gy + 56;
  if (mouseX >= ox && mouseX < ox + n * cell && mouseY >= oy && mouseY < oy + n * cell) {
    let j = Math.floor((mouseX - ox) / cell);
    let i = Math.floor((mouseY - oy) / cell);
    if (i >= 0 && i < n && j >= 0 && j < n && i !== j) {
      adj[i][j] = adj[i][j] ? 0 : 1;
      adj[j][i] = adj[i][j];
    }
  }
}

function highlightActiveModeButton() {
  let active = '#3949ab', activeTxt = 'white';
  let idle = '', idleTxt = 'black';
  if (!rawButton) return;
  rawButton.style('background-color', normMode === 'raw' ? active : '#eeeeee');
  rawButton.style('color', normMode === 'raw' ? activeTxt : idleTxt);
  meanButton.style('background-color', normMode === 'mean' ? active : '#eeeeee');
  meanButton.style('color', normMode === 'mean' ? activeTxt : idleTxt);
  symButton.style('background-color', normMode === 'sym' ? active : '#eeeeee');
  symButton.style('color', normMode === 'sym' ? activeTxt : idleTxt);
}

// ---------- responsive ----------

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  featSlider.size(canvasWidth - sliderLeftMargin - margin);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
