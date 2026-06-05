// MathJax 3 configuration for GNN textbook.
// Uses arithmatex generic mode: display math becomes \[...\], inline becomes \(...\).
// The boldsymbol package is explicitly loaded for bold Greek letters (θ, Λ, Σ, etc.)
// used throughout the textbook's GNN equations.
window.MathJax = {
  loader: {
    load: ['[tex]/boldsymbol']
  },
  tex: {
    packages: {'[+]': ['boldsymbol']},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})
