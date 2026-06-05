// GNN Design Space Comparison (GCN vs GraphSAGE vs GAT) - Chart.js
// CANVAS_HEIGHT: 520
// Learning objective (Evaluate, Bloom L5): compare architecture choices across
// datasets and metrics to build judgment for selecting an appropriate GNN design.
// MicroSim template version 2026.03

(function () {
  const DATASETS = ['Cora', 'CiteSeer', 'ogbn-arxiv', 'QM9 (perf)'];
  const MODELS = [
    { key: 'GCN', color: '#6366f1', params: '~23K', acc: [81.5, 70.3, 71.7, 65.0] },
    { key: 'GraphSAGE', color: '#10b981', params: '~33K', acc: [82.0, 71.3, 71.5, 66.0] },
    { key: 'GAT', color: '#f59e0b', params: '~92K', acc: [83.0, 72.5, 73.9, 70.0] },
  ];

  let metric = 'accuracy';            // 'accuracy' | 'delta'
  const visible = { GCN: true, GraphSAGE: true, GAT: true };
  let chart;

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    const main = document.querySelector('main');
    main.innerHTML = '';
    document.body.style.background = '#ffffff';

    const wrap = document.createElement('div');
    wrap.style.cssText = 'font-family:Arial,Helvetica,sans-serif;padding:8px 12px;background:#ffffff;min-height:512px;box-sizing:border-box;';
    main.appendChild(wrap);

    const h = document.createElement('div');
    h.textContent = 'GNN Design Space: GCN vs GraphSAGE vs GAT';
    h.style.cssText = 'text-align:center;font-size:20px;font-weight:bold;margin:4px 0 8px;';
    wrap.appendChild(h);

    // controls
    const controls = document.createElement('div');
    controls.style.cssText = 'display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:center;margin-bottom:8px;font-size:14px;';
    wrap.appendChild(controls);

    // metric toggle
    const mLabel = document.createElement('span'); mLabel.textContent = 'Metric:'; mLabel.style.fontWeight = 'bold';
    controls.appendChild(mLabel);
    const accBtn = mkBtn('Accuracy', true);
    const dBtn = mkBtn('Δ over GCN', false);
    accBtn.onclick = () => { metric = 'accuracy'; accBtn.dataset.on = '1'; dBtn.dataset.on = '0'; styleBtn(accBtn); styleBtn(dBtn); update(); };
    dBtn.onclick = () => { metric = 'delta'; dBtn.dataset.on = '1'; accBtn.dataset.on = '0'; styleBtn(accBtn); styleBtn(dBtn); update(); };
    controls.appendChild(accBtn); controls.appendChild(dBtn);

    // architecture filter
    const fLabel = document.createElement('span'); fLabel.textContent = '   Show:'; fLabel.style.fontWeight = 'bold';
    controls.appendChild(fLabel);
    MODELS.forEach(m => {
      const lab = document.createElement('label');
      lab.style.cssText = 'display:inline-flex;align-items:center;gap:4px;cursor:pointer;';
      const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = true;
      cb.onchange = () => { visible[m.key] = cb.checked; update(); };
      const sw = document.createElement('span'); sw.style.cssText = `display:inline-block;width:12px;height:12px;border-radius:2px;background:${m.color};`;
      lab.appendChild(cb); lab.appendChild(sw); lab.appendChild(document.createTextNode(m.key));
      controls.appendChild(lab);
    });

    // canvas
    const cWrap = document.createElement('div');
    cWrap.style.cssText = 'position:relative;height:400px;width:100%;';
    const canvas = document.createElement('canvas'); canvas.id = 'designChart';
    cWrap.appendChild(canvas); wrap.appendChild(cWrap);

    chart = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: { labels: DATASETS, datasets: buildDatasets() },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const m = MODELS.find(mm => mm.key === ctx.dataset.label);
                const v = ctx.parsed.y;
                const unit = metric === 'delta' ? ' pts vs GCN' : (ctx.label.startsWith('QM9') ? ' score' : '%');
                return `${ctx.dataset.label} · ${ctx.label}: ${v.toFixed(1)}${unit}  (params ${m.params})`;
              }
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: metric === 'delta' ? 'Δ accuracy vs GCN (points)' : 'Test accuracy (%) / performance score' },
            ticks: { stepSize: 10 }
          },
          x: { title: { display: true, text: 'Dataset' } }
        }
      }
    });
    update();
  }

  function buildDatasets() {
    return MODELS.filter(m => visible[m.key]).map(m => ({
      label: m.key,
      backgroundColor: m.color,
      borderColor: m.color,
      borderWidth: 1,
      data: dataFor(m),
    }));
  }

  function dataFor(m) {
    if (metric === 'accuracy') return m.acc.slice();
    const gcn = MODELS.find(x => x.key === 'GCN').acc;
    return m.acc.map((v, i) => +(v - gcn[i]).toFixed(2));
  }

  function update() {
    chart.data.datasets = buildDatasets();
    chart.options.scales.y.title.text = metric === 'delta' ? 'Δ accuracy vs GCN (points)' : 'Test accuracy (%) / performance score';
    chart.options.scales.y.beginAtZero = metric === 'delta';
    chart.update();
  }

  // ---- button helpers ----
  function mkBtn(label, on) {
    const b = document.createElement('button');
    b.textContent = label; b.dataset.on = on ? '1' : '0';
    b.style.cssText = 'padding:4px 10px;border:1px solid #ccc;border-radius:4px;cursor:pointer;font-size:13px;';
    styleBtn(b); return b;
  }
  function styleBtn(b) {
    if (b.dataset.on === '1') { b.style.background = '#6366f1'; b.style.color = 'white'; }
    else { b.style.background = '#eee'; b.style.color = 'black'; }
  }
})();
