// Relational Tables → Heterogeneous Graph
// CANVAS_HEIGHT: 560
// Learning objective (Understand): foreign keys in a relational database (Customers,
// Orders, Products) map to typed nodes and typed edges in a heterogeneous graph.
// Click a row or a node to see the correspondence and the graph neighborhood.
// MicroSim template version 2026.03

let containerWidth;
let canvasWidth = 900;
let drawHeight = 500;
let controlHeight = 60;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 25;
let defaultTextSize = 16;

let reverseToggle, labelToggle, resetButton;

const NC = 8, NO = 12, NP = 5;
// global node ids: customers 0..NC-1, orders NC..NC+NO-1, products NC+NO..
let custName = [], orderCust = [], orderProds = [], prodName = [];
let typeOf = [], pos = [], edges = [];   // edge [a,b,rel] rel:0 places(cust->order),1 includes(order->prod)
let showReverse = false, showLabels = false;
let selNode = -1, hoverNode = -1;
const TYPE_COL = ['#1565c0', '#43a047', '#fb8c00'];
const TYPE_NAME = ['Customer', 'Order', 'Product'];
const REL_NAME = ['places', 'includes'];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  buildData();

  reverseToggle = createCheckbox(' Show reverse edges', false); reverseToggle.position(10, drawHeight + 8); reverseToggle.changed(() => showReverse = reverseToggle.checked());
  labelToggle = createCheckbox(' Show meta-relation labels', false); labelToggle.position(190, drawHeight + 8); labelToggle.changed(() => showLabels = labelToggle.checked());
  resetButton = createButton('Reset'); resetButton.position(420, drawHeight + 8); resetButton.mousePressed(() => selNode = -1);

  describe('A three-table e-commerce database (Customers, Orders, Products) shown beside ' +
    'the heterogeneous graph it becomes. Foreign keys map to typed edges: a customer places ' +
    'an order; an order includes products. Click a table row or a graph node to highlight the ' +
    'correspondence and the 1-hop / 2-hop neighborhood.', LABEL);
}

function draw() {
  updateCanvasSize();
  fill('aliceblue'); stroke('silver'); strokeWeight(1); rect(0, 0, canvasWidth, drawHeight);
  fill('white'); rect(0, drawHeight, canvasWidth, controlHeight);
  noStroke();

  let lw = canvasWidth * 0.40;
  fill('black'); noStroke(); textAlign(CENTER, TOP); textSize(15);
  text('Relational tables', lw / 2, 6); text('Heterogeneous graph', lw + (canvasWidth - lw) / 2, 6);
  stroke('silver'); line(lw, 26, lw, drawHeight - 8);

  hoverNode = -1;
  let hop = neighborhood(selNode);
  drawTables(0, 28, lw, hop);
  drawGraph(lw, 28, canvasWidth - lw, drawHeight - 36, hop);
  if (hoverNode >= 0) drawTip();
}

// ---------- tables ----------

function drawTables(gx, gy, gw, hop) {
  let x = gx + 8, y = gy + 4, rh = 15;
  // Customers
  y = tableHeader(x, y, 'Customers', '#1565c0', 'id   name', gw);
  for (let c = 0; c < NC; c++) { y = tableRow(x, y, gw, rh, 'c' + c + '   ' + custName[c], custNode(c), hop, -1); }
  y += 8;
  // Orders
  y = tableHeader(x, y, 'Orders', '#43a047', 'id   customer_id(FK)', gw);
  for (let o = 0; o < NO; o++) { y = tableRow(x, y, gw, rh, 'o' + o + '   c' + orderCust[o], orderNode(o), hop, 'fkc' + orderCust[o]); }
  y += 8;
  // Products
  y = tableHeader(x, y, 'Products', '#fb8c00', 'id   name', gw);
  for (let p = 0; p < NP; p++) { y = tableRow(x, y, gw, rh, 'p' + p + '   ' + prodName[p], prodNode(p), hop, -1); }
}

function tableHeader(x, y, title, col, cols, gw) {
  noStroke(); fill(col); rect(x, y, gw - 16, 16); fill('white'); textAlign(LEFT, CENTER); textSize(11); textStyle(BOLD);
  text(title + '   (' + cols + ')', x + 4, y + 8); textStyle(NORMAL); return y + 18;
}
function tableRow(x, y, gw, rh, txt, node, hop, fkTag) {
  let sel = node === selNode, h = hop && hop.has(node);
  noStroke(); fill(sel ? '#ffe082' : (h ? '#fff8e1' : 'white')); stroke('#ddd'); strokeWeight(1); rect(x, y, gw - 16, rh);
  if (mouseX > x && mouseX < x + gw - 16 && mouseY > y && mouseY < y + rh) hoverNode = node;
  noStroke(); fill('black'); textAlign(LEFT, CENTER); textSize(11); text(txt, x + 4, y + rh / 2);
  return y + rh;
}

// ---------- graph ----------

function nodeXY(i, gx, gy, gw, gh) { return { x: gx + 28 + pos[i].x * (gw - 56), y: gy + 16 + pos[i].y * (gh - 32) }; }

function drawGraph(gx, gy, gw, gh, hop) {
  for (let e of edges) {
    let a = nodeXY(e[0], gx, gy, gw, gh), b = nodeXY(e[1], gx, gy, gw, gh);
    let inHop = hop && hop.has(e[0]) && hop.has(e[1]);
    stroke(e[2] === 0 ? '#1565c0' : '#fb8c00'); if (!inHop && selNode >= 0) drawingContext.globalAlpha = 0.15;
    strokeWeight(inHop ? 2.5 : 1.2);
    drawArrow(a, b, 12);
    if (showReverse) { stroke(e[2] === 0 ? '#90caf9' : '#ffcc80'); drawArrow(b, a, 12); }
    drawingContext.globalAlpha = 1;
    if (showLabels) { noStroke(); fill('#555'); textAlign(CENTER, CENTER); textSize(9); text(REL_NAME[e[2]], (a.x + b.x) / 2, (a.y + b.y) / 2 - 6); }
  }
  for (let i = 0; i < typeOf.length; i++) {
    let p = nodeXY(i, gx, gy, gw, gh), t = typeOf[i];
    if (dist(mouseX, mouseY, p.x, p.y) <= 12) hoverNode = i;
    let dimmed = selNode >= 0 && hop && !hop.has(i);
    push(); if (dimmed) drawingContext.globalAlpha = 0.2;
    let isSel = i === selNode;
    stroke(isSel ? '#d32f2f' : 50); strokeWeight(isSel ? 3 : 1); fill(TYPE_COL[t]);
    let r = 10;
    if (t === 0) circle(p.x, p.y, r * 2);
    else if (t === 1) { rectMode(CENTER); rect(p.x, p.y, r * 2, r * 2, 2); rectMode(CORNER); }
    else { push(); translate(p.x, p.y); rotate(QUARTER_PI); rectMode(CENTER); rect(0, 0, r * 1.7, r * 1.7); rectMode(CORNER); pop(); }
    noStroke(); fill('white'); textAlign(CENTER, CENTER); textSize(9); text(localLabel(i), p.x, p.y);
    pop();
  }
  // legend: shape + color key for the three node types (drawn bottom-left of graph panel)
  let ly = gy + gh - 12, lx = gx + 12;
  textAlign(LEFT, CENTER); textSize(11);
  // opaque backing so graph nodes/edges don't show through the key
  let legW = 0; for (let t = 0; t < 3; t++) legW += 16 + textWidth(TYPE_NAME[t]) + 16;
  noStroke(); fill(255, 255, 255, 238); stroke('silver'); strokeWeight(1);
  rect(gx + 4, ly - 11, legW + 8, 22, 4);
  noStroke();
  for (let t = 0; t < 3; t++) {
    let sx = lx;
    stroke(50); strokeWeight(1); fill(TYPE_COL[t]);
    if (t === 0) circle(sx, ly, 13);
    else if (t === 1) { rectMode(CENTER); rect(sx, ly, 13, 13, 2); rectMode(CORNER); }
    else { push(); translate(sx, ly); rotate(QUARTER_PI); rectMode(CENTER); rect(0, 0, 11, 11); rectMode(CORNER); pop(); }
    noStroke(); fill('#333'); text(TYPE_NAME[t], sx + 11, ly);
    lx += 16 + textWidth(TYPE_NAME[t]) + 16;
  }
}

function drawArrow(a, b, rad) {
  let ang = atan2(b.y - a.y, b.x - a.x); let ex = b.x - cos(ang) * rad, ey = b.y - sin(ang) * rad;
  line(a.x, a.y, ex, ey); push(); translate(ex, ey); rotate(ang); fill(drawingContext.strokeStyle); noStroke(); triangle(0, 0, -7, 2.5, -7, -2.5); pop();
}

function drawTip() {
  let i = hoverNode, t = typeOf[i];
  let lines = [TYPE_NAME[t] + ' ' + localLabel(i)];
  if (t === 0) lines.push('name: ' + custName[i]);
  else if (t === 1) lines.push('customer_id: c' + orderCust[i - NC]);
  else lines.push('name: ' + prodName[i - NC - NO]);
  let w = 160, h = lines.length * 17 + 12;
  let tx = constrain(mouseX + 14, 4, canvasWidth - w - 4), ty = constrain(mouseY, 28, drawHeight - h - 4);
  fill(255, 255, 255, 248); stroke(100); strokeWeight(1); rect(tx, ty, w, h, 6);
  noStroke(); fill('black'); textAlign(LEFT, TOP); textSize(12);
  for (let k = 0; k < lines.length; k++) text(lines[k], tx + 8, ty + 6 + k * 17);
}

// ---------- helpers ----------

function custNode(c) { return c; }
function orderNode(o) { return NC + o; }
function prodNode(p) { return NC + NO + p; }
function localLabel(i) { let t = typeOf[i]; if (t === 0) return 'c' + i; if (t === 1) return 'o' + (i - NC); return 'p' + (i - NC - NO); }

function neighborhood(s) {
  if (s < 0) return null;
  let adj = Array.from({ length: typeOf.length }, () => []);
  for (let e of edges) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
  let set = new Set([s]); let d = new Set([s]);
  for (let hop = 0; hop < 2; hop++) { let nd = new Set(); for (let u of d) for (let v of adj[u]) if (!set.has(v)) { set.add(v); nd.add(v); } d = nd; }
  return set;
}

function buildData() {
  let rng = mulberry32(7);
  let names = ['Ava', 'Ben', 'Cy', 'Dee', 'Eli', 'Fay', 'Gus', 'Hil'];
  let pnames = ['Pen', 'Mug', 'Lamp', 'Book', 'Hat'];
  custName = names.slice(0, NC); prodName = pnames.slice(0, NP);
  orderCust = []; orderProds = [];
  for (let o = 0; o < NO; o++) { orderCust.push(Math.floor(rng() * NC)); let np = 1 + (rng() < 0.5 ? 1 : 0); let ps = []; for (let k = 0; k < np; k++) ps.push(Math.floor(rng() * NP)); orderProds.push(ps); }
  typeOf = [];
  for (let i = 0; i < NC; i++) typeOf.push(0);
  for (let i = 0; i < NO; i++) typeOf.push(1);
  for (let i = 0; i < NP; i++) typeOf.push(2);
  edges = [];
  for (let o = 0; o < NO; o++) { edges.push([custNode(orderCust[o]), orderNode(o), 0]); for (let p of orderProds[o]) edges.push([orderNode(o), prodNode(p), 1]); }
  pos = springLayout(typeOf.length);
}

function springLayout(n) {
  let p = [];
  // type-grouped init: customers left, orders middle, products right
  for (let i = 0; i < n; i++) { let t = typeOf[i]; let cx = t === 0 ? 0.22 : t === 1 ? 0.5 : 0.78; p.push({ x: cx + (Math.random ? 0 : 0), y: 0.5 }); }
  // deterministic vertical spread
  let counts = [0, 0, 0];
  for (let i = 0; i < n; i++) { let t = typeOf[i]; let tot = [NC, NO, NP][t]; p[i].y = 0.1 + 0.8 * counts[t] / Math.max(tot - 1, 1); counts[t]++; }
  for (let it = 0; it < 160; it++) {
    let fx = new Array(n).fill(0), fy = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, d2 = dx * dx + dy * dy + 1e-4, d = Math.sqrt(d2), r = 0.0005 / d2; fx[i] += dx / d * r; fy[i] += dy / d * r; fx[j] -= dx / d * r; fy[j] -= dy / d * r; }
    for (let e of edges) { let a = p[e[0]], b = p[e[1]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-4, s = (d - 0.18) * 0.015; fx[e[0]] += dx / d * s; fy[e[0]] += dy / d * s; fx[e[1]] -= dx / d * s; fy[e[1]] -= dy / d * s; }
    for (let i = 0; i < n; i++) { p[i].x += constrain(fx[i], -0.015, 0.015); p[i].y += constrain(fy[i], -0.015, 0.015); }
  }
  let xs = p.map(q => q.x), ys = p.map(q => q.y), mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return p.map(q => ({ x: (q.x - mnx) / (mxx - mnx + 1e-9), y: (q.y - mny) / (mxy - mny + 1e-9) }));
}

function mousePressed() {
  let lw = canvasWidth * 0.40;
  // graph node click
  if (mouseX > lw && mouseY > 28 && mouseY < drawHeight) {
    for (let i = 0; i < typeOf.length; i++) { let p = nodeXY(i, lw, 28, canvasWidth - lw, drawHeight - 36); if (dist(mouseX, mouseY, p.x, p.y) <= 13) { selNode = (selNode === i ? -1 : i); return; } }
  }
  // table row click → via hoverNode set during draw
  if (mouseX < lw && hoverNode >= 0) { selNode = (selNode === hoverNode ? -1 : hoverNode); }
}

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function windowResized() { updateCanvasSize(); resizeCanvas(containerWidth, containerHeight); redraw(); }
function updateCanvasSize() { const c = document.querySelector('main').getBoundingClientRect(); containerWidth = Math.floor(c.width); canvasWidth = containerWidth; }
