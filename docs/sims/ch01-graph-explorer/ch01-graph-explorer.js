// Graph Property Explorer - Interactive Graph Builder with Live Property Updates
// CANVAS_HEIGHT: 570
// Learning objective (Apply, Bloom L3): construct graphs interactively and
// observe how global properties (degree distribution, path length, clustering,
// components, bipartiteness) respond to structural edits.
// MicroSim template version 2026.03

// ----- responsive canvas globals -----
let containerWidth;
let canvasWidth = 900;
let drawHeight = 520;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

const PANEL_W = 290;   // right-side readout panel width

// ----- controls -----
let presetSelect, clearButton, shotButton;

// ----- model -----
let nodes = [];   // {x, y, vx, vy}
let edges = [];   // [i, j] with i < j

// ----- interaction state -----
let draggingNode = -1;
let pressNode = -1;
let pressX = 0, pressY = 0;
let movedSincePress = false;
let firstSel = -1;     // first node chosen for an edge toggle
let hoverNode = -1;

// component palette (tints)
const COMP_TINTS = ['#1565c0', '#2e7d32', '#c62828', '#6a1b9a', '#ef6c00'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);

  // suppress browser context menu so right-click can delete nodes
  canvas.elt.addEventListener('contextmenu', (e) => e.preventDefault());

  presetSelect = createSelect();
  presetSelect.position(10, drawHeight + 10);
  ['Karate Club (10-node sample)', 'Empty', 'Complete K5', 'Cycle C6',
   'Star K₁₋₆', 'Path P8', 'Random ER (n=12, p=0.3)'].forEach(o => presetSelect.option(o));
  presetSelect.changed(loadPreset);

  clearButton = createButton('Clear');
  clearButton.position(250, drawHeight + 10);
  clearButton.mousePressed(() => { nodes = []; edges = []; firstSel = -1; });

  shotButton = createButton('Save PNG');
  shotButton.position(320, drawHeight + 10);
  shotButton.mousePressed(() => saveCanvas('graph-property-explorer', 'png'));

  describe('Interactive graph builder. Left-click empty space to add a node, ' +
    'click one node then another to toggle an edge, right-click a node to delete ' +
    'it, drag to reposition. A live panel reports degree distribution, path length, ' +
    'clustering, connected components, and bipartiteness.', LABEL);

  loadPreset();   // start with a populated graph so the concept is visible on load
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

  let graphW = canvasWidth - PANEL_W;

  runPhysics(graphW);

  // title
  fill('black');
  noStroke();
  textAlign(CENTER, TOP);
  textSize(22);
  text('Graph Property Explorer', graphW / 2, 8);

  let comp = components();
  let bridgeSet = bridgeEdgeSet();

  drawEdges(bridgeSet);
  hoverNode = nodeAt(mouseX, mouseY, graphW);
  drawNodes(comp);
  drawHint(graphW);
  drawPanel(graphW, comp);

  if (hoverNode >= 0) drawTooltip(hoverNode, comp);

  // control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(13);
  text('Preset', 10, drawHeight + 38);
}

// ---------- rendering ----------

function drawEdges(bridgeSet) {
  strokeWeight(2.5);
  for (let e of edges) {
    let key = e[0] + '-' + e[1];
    stroke(bridgeSet[key] ? '#ef6c00' : 'lightgray');
    line(nodes[e[0]].x, nodes[e[0]].y, nodes[e[1]].x, nodes[e[1]].y);
  }
}

function drawNodes(comp) {
  let deg = nodeDegrees();
  for (let i = 0; i < nodes.length; i++) {
    let d = deg[i];
    let r = nodeRadius(d);
    // component tint stroke
    let tint = COMP_TINTS[comp.label[i] % COMP_TINTS.length];
    if (i === firstSel) { stroke('gold'); strokeWeight(4); }
    else { stroke(tint); strokeWeight(3); }
    fill(degreeColor(d));
    circle(nodes[i].x, nodes[i].y, r * 2);

    noStroke();
    fill(d >= 6 ? 'white' : 'black');
    textAlign(CENTER, CENTER);
    textSize(13);
    text(i, nodes[i].x, nodes[i].y);
  }
}

function drawHint(graphW) {
  noStroke();
  fill(90);
  textAlign(LEFT, TOP);
  textSize(12);
  text('click empty = add node  •  click two nodes = toggle edge', 10, drawHeight - 34);
  text('right-click = delete  •  drag = move', 10, drawHeight - 18);
  if (firstSel >= 0) {
    fill('#b8860b');
    textAlign(RIGHT, TOP);
    text('edge start: node ' + firstSel + ' (click another node)', graphW - 10, 12);
  }
}

function drawPanel(graphW, comp) {
  let px = graphW;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(px, 0, PANEL_W, drawHeight);

  let tx = px + 14;
  let ty = 16;
  let lh = 19;
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);

  textSize(16);
  text('Live Graph Properties', tx, ty); ty += lh + 6;

  textSize(14);
  let n = nodes.length;
  let m = edges.length;
  text('Nodes: ' + n + '     Edges: ' + m, tx, ty); ty += lh;

  let avgDeg = n > 0 ? (2 * m / n) : 0;
  text('Average degree: ' + nf(avgDeg, 1, 1), tx, ty); ty += lh;

  // components
  text('Connected components: ' + comp.count, tx, ty); ty += lh;
  if (comp.count > 0) {
    let sizes = comp.sizes.slice().sort((a, b) => b - a).slice(0, 3);
    text('  top sizes: ' + sizes.join(', '), tx, ty); ty += lh;
  }

  // avg shortest path
  let asp = avgShortestPath(comp);
  text('Avg shortest path: ' + asp, tx, ty); ty += lh;

  // clustering
  text('Clustering (transitivity): ' + nf(transitivity(), 1, 2), tx, ty); ty += lh;

  // bipartite
  text('Bipartite: ' + (isBipartite() ? 'Yes' : 'No'), tx, ty); ty += lh + 6;

  // degree distribution
  textSize(14);
  text('Degree distribution P(k):', tx, ty); ty += lh;
  textSize(13);
  let deg = nodeDegrees();
  if (n === 0) {
    text('  (empty graph)', tx, ty);
  } else {
    let maxK = Math.max(0, ...deg);
    let counts = new Array(maxK + 1).fill(0);
    for (let d of deg) counts[d]++;
    let barX = tx + 34;                       // bar start x
    let barMax = (PANEL_W - 14) - (barX - tx) - 44;  // max bar width
    for (let k = 0; k <= maxK; k++) {
      if (ty > drawHeight - 16) break;
      let p = counts[k] / n;
      // label "k=N"
      noStroke();
      fill('black');
      textAlign(LEFT, CENTER);
      textSize(12);
      text('k=' + k, tx, ty + 6);
      // bar
      fill('#90caf9');
      stroke('#1565c0');
      strokeWeight(1);
      rect(barX, ty + 1, max(1, p * barMax), 11);
      // P(k) value
      noStroke();
      fill('dimgray');
      text(nf(p, 1, 2), barX + barMax + 6, ty + 6);
      ty += 15;
    }
    textAlign(LEFT, TOP);
  }
}

function drawTooltip(i, comp) {
  let deg = nodeDegrees();
  let cv = nodeClustering(i, deg);
  let ecc = eccentricity(i);
  let lines = [
    'Node ' + i,
    'Degree: ' + deg[i],
    'Clustering: ' + nf(cv, 1, 2),
    'Eccentricity: ' + (ecc === null ? 'N/A' : ecc)
  ];
  let w = 150, h = lines.length * 18 + 10;
  let tx = mouseX + 14, ty = mouseY + 4;
  if (tx + w > canvasWidth - PANEL_W) tx = mouseX - w - 14;
  if (ty + h > drawHeight) ty = drawHeight - h;
  fill(255, 255, 255, 240);
  stroke(120);
  strokeWeight(1);
  rect(tx, ty, w, h, 6);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(13);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 18);
}

// ---------- visual encodings ----------

function nodeRadius(d) {
  return constrain(map(d, 0, 10, 12, 30), 12, 30);
}

function degreeColor(d) {
  if (d <= 3) return lerpColor(color('white'), color('#90caf9'), constrain(d / 3, 0, 1));
  return lerpColor(color('#90caf9'), color('#1a237e'), constrain((d - 3) / 7, 0, 1));
}

// ---------- physics layout ----------

function runPhysics(graphW) {
  let n = nodes.length;
  let k = 0.05;        // spring constant
  let charge = -500;   // repulsion
  let restLen = 90;
  let pad = 32;

  // repulsion (all pairs, n is small)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dx = nodes[i].x - nodes[j].x;
      let dy = nodes[i].y - nodes[j].y;
      let d2 = dx * dx + dy * dy;
      let d = Math.sqrt(d2) || 0.01;
      let f = charge / d2;
      let fx = (dx / d) * f;
      let fy = (dy / d) * f;
      nodes[i].vx -= fx; nodes[i].vy -= fy;
      nodes[j].vx += fx; nodes[j].vy += fy;
    }
  }
  // springs along edges
  for (let e of edges) {
    let a = nodes[e[0]], b = nodes[e[1]];
    let dx = b.x - a.x, dy = b.y - a.y;
    let d = Math.sqrt(dx * dx + dy * dy) || 0.01;
    let f = k * (d - restLen);
    let fx = (dx / d) * f, fy = (dy / d) * f;
    a.vx += fx; a.vy += fy;
    b.vx -= fx; b.vy -= fy;
  }
  // integrate with damping
  for (let i = 0; i < n; i++) {
    if (i === draggingNode) { nodes[i].vx = 0; nodes[i].vy = 0; continue; }
    nodes[i].vx *= 0.85; nodes[i].vy *= 0.85;
    nodes[i].vx = constrain(nodes[i].vx, -8, 8);
    nodes[i].vy = constrain(nodes[i].vy, -8, 8);
    nodes[i].x += nodes[i].vx;
    nodes[i].y += nodes[i].vy;
    nodes[i].x = constrain(nodes[i].x, pad, graphW - pad);
    nodes[i].y = constrain(nodes[i].y, 44, drawHeight - pad);
  }
}

// ---------- graph algorithms ----------

function adjacency() {
  let adj = nodes.map(() => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  return adj;
}

function nodeDegrees() {
  let deg = new Array(nodes.length).fill(0);
  for (let e of edges) { deg[e[0]]++; deg[e[1]]++; }
  return deg;
}

function components() {
  let n = nodes.length;
  let label = new Array(n).fill(-1);
  let adj = adjacency();
  let count = 0;
  let sizes = [];
  for (let s = 0; s < n; s++) {
    if (label[s] !== -1) continue;
    let size = 0;
    let queue = [s];
    label[s] = count;
    while (queue.length) {
      let u = queue.shift();
      size++;
      for (let v of adj[u]) if (label[v] === -1) { label[v] = count; queue.push(v); }
    }
    sizes.push(size);
    count++;
  }
  return { label, count, sizes };
}

function bridgeEdgeSet() {
  // an edge is a bridge if removing it increases the component count
  let result = {};
  if (edges.length === 0) return result;
  let baseCount = components().count;
  for (let idx = 0; idx < edges.length; idx++) {
    let e = edges[idx];
    let saved = edges;
    edges = edges.filter((_, k) => k !== idx);
    let c = components().count;
    edges = saved;
    if (c > baseCount) result[e[0] + '-' + e[1]] = true;
  }
  return result;
}

function avgShortestPath(comp) {
  let n = nodes.length;
  if (n < 2) return 'N/A';
  if (comp.count > 1) return 'N/A (disconnected)';
  if (n > 40) return 'N/A (n > 40)';
  let adj = adjacency();
  let total = 0, pairs = 0;
  for (let s = 0; s < n; s++) {
    let dist = new Array(n).fill(-1);
    dist[s] = 0;
    let queue = [s];
    while (queue.length) {
      let u = queue.shift();
      for (let v of adj[u]) if (dist[v] === -1) { dist[v] = dist[u] + 1; queue.push(v); }
    }
    for (let t = 0; t < n; t++) if (t !== s && dist[t] > 0) { total += dist[t]; pairs++; }
  }
  return pairs > 0 ? nf(total / pairs, 1, 2) : 'N/A';
}

function transitivity() {
  let adj = adjacency().map(a => new Set(a));
  let closed = 0, triples = 0;
  for (let v = 0; v < nodes.length; v++) {
    let nbrs = [...adj[v]];
    let d = nbrs.length;
    triples += d * (d - 1) / 2;
    for (let a = 0; a < nbrs.length; a++) {
      for (let b = a + 1; b < nbrs.length; b++) {
        if (adj[nbrs[a]].has(nbrs[b])) closed++;
      }
    }
  }
  return triples > 0 ? closed / triples : 0;
}

function nodeClustering(i, deg) {
  let adj = adjacency().map(a => new Set(a));
  let nbrs = [...adj[i]];
  let d = nbrs.length;
  if (d < 2) return 0;
  let links = 0;
  for (let a = 0; a < nbrs.length; a++)
    for (let b = a + 1; b < nbrs.length; b++)
      if (adj[nbrs[a]].has(nbrs[b])) links++;
  return links / (d * (d - 1) / 2);
}

function eccentricity(s) {
  let adj = adjacency();
  let n = nodes.length;
  let dist = new Array(n).fill(-1);
  dist[s] = 0;
  let queue = [s], maxD = 0;
  while (queue.length) {
    let u = queue.shift();
    for (let v of adj[u]) if (dist[v] === -1) { dist[v] = dist[u] + 1; maxD = max(maxD, dist[v]); queue.push(v); }
  }
  // if not all reachable, eccentricity within component
  return maxD;
}

function isBipartite() {
  let n = nodes.length;
  if (n === 0) return true;
  let adj = adjacency();
  let color = new Array(n).fill(-1);
  for (let s = 0; s < n; s++) {
    if (color[s] !== -1) continue;
    color[s] = 0;
    let queue = [s];
    while (queue.length) {
      let u = queue.shift();
      for (let v of adj[u]) {
        if (color[v] === -1) { color[v] = 1 - color[u]; queue.push(v); }
        else if (color[v] === color[u]) return false;
      }
    }
  }
  return true;
}

// ---------- edge / node editing ----------

function toggleEdge(a, b) {
  if (a === b) return;
  let i = min(a, b), j = max(a, b);
  let idx = edges.findIndex(e => e[0] === i && e[1] === j);
  if (idx >= 0) edges.splice(idx, 1);
  else edges.push([i, j]);
}

function addNode(x, y) {
  nodes.push({ x: x, y: y, vx: 0, vy: 0 });
}

function removeNode(k) {
  nodes.splice(k, 1);
  edges = edges.filter(e => e[0] !== k && e[1] !== k)
               .map(e => [e[0] > k ? e[0] - 1 : e[0], e[1] > k ? e[1] - 1 : e[1]])
               .map(e => [min(e[0], e[1]), max(e[0], e[1])]);
  if (firstSel === k) firstSel = -1;
  else if (firstSel > k) firstSel--;
}

function nodeAt(mx, my, graphW) {
  if (graphW !== undefined && mx > graphW) return -1;
  let deg = nodeDegrees();
  for (let i = nodes.length - 1; i >= 0; i--) {
    let r = nodeRadius(deg[i]);
    if (dist(mx, my, nodes[i].x, nodes[i].y) <= r) return i;
  }
  return -1;
}

// ---------- mouse interaction ----------

function mousePressed() {
  let graphW = canvasWidth - PANEL_W;
  if (mouseX < 0 || mouseX > graphW || mouseY < 40 || mouseY > drawHeight) return;

  if (mouseButton === RIGHT) {
    let hit = nodeAt(mouseX, mouseY, graphW);
    if (hit >= 0) removeNode(hit);
    return false;
  }

  pressNode = nodeAt(mouseX, mouseY, graphW);
  pressX = mouseX; pressY = mouseY;
  movedSincePress = false;
  if (pressNode >= 0) draggingNode = pressNode;
}

function mouseDragged() {
  if (draggingNode >= 0) {
    let graphW = canvasWidth - PANEL_W;
    nodes[draggingNode].x = constrain(mouseX, 20, graphW - 20);
    nodes[draggingNode].y = constrain(mouseY, 44, drawHeight - 20);
    if (dist(mouseX, mouseY, pressX, pressY) > 4) movedSincePress = true;
  }
}

function mouseReleased() {
  let graphW = canvasWidth - PANEL_W;
  let inGraph = (mouseX >= 0 && mouseX <= graphW && mouseY >= 40 && mouseY <= drawHeight);

  if (pressNode >= 0) {
    if (!movedSincePress) handleNodeClick(pressNode);
  } else if (inGraph && !movedSincePress) {
    addNode(mouseX, mouseY);
    firstSel = -1;
  }
  draggingNode = -1;
  pressNode = -1;
}

function handleNodeClick(i) {
  if (firstSel === -1) firstSel = i;
  else if (firstSel === i) firstSel = -1;
  else { toggleEdge(firstSel, i); firstSel = -1; }
}

// ---------- presets ----------

function loadPreset() {
  let name = presetSelect.value();
  nodes = []; edges = []; firstSel = -1;
  let graphW = canvasWidth - PANEL_W;
  let cx = graphW / 2, cy = drawHeight / 2;
  let R = min(graphW, drawHeight) * 0.32;

  function ring(count) {
    for (let i = 0; i < count; i++) {
      let a = TWO_PI * i / count - HALF_PI;
      addNode(cx + R * cos(a), cy + R * sin(a));
    }
  }

  if (name === 'Empty') {
    return;
  } else if (name.startsWith('Complete K5')) {
    ring(5);
    for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) edges.push([i, j]);
  } else if (name.startsWith('Cycle C6')) {
    ring(6);
    for (let i = 0; i < 6; i++) edges.push([min(i, (i + 1) % 6), max(i, (i + 1) % 6)]);
  } else if (name.startsWith('Star')) {
    addNode(cx, cy);
    for (let i = 0; i < 6; i++) {
      let a = TWO_PI * i / 6;
      addNode(cx + R * cos(a), cy + R * sin(a));
      edges.push([0, i + 1]);
    }
  } else if (name.startsWith('Path P8')) {
    for (let i = 0; i < 8; i++) addNode(40 + (graphW - 80) * i / 7, cy);
    for (let i = 0; i < 7; i++) edges.push([i, i + 1]);
  } else if (name.startsWith('Random ER')) {
    ring(12);
    for (let i = 0; i < 12; i++)
      for (let j = i + 1; j < 12; j++)
        if (Math.random() < 0.3) edges.push([i, j]);
  } else {
    // Karate Club (10-node sample)
    ring(10);
    let es = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
              [2, 3], [4, 5], [5, 6], [5, 7], [6, 7], [0, 5], [8, 9], [2, 8]];
    for (let e of es) edges.push([min(e[0], e[1]), max(e[0], e[1])]);
  }
}

// ---------- responsive ----------

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
