#!/usr/bin/env bash
# Build the MkDocs site and serve it at /graph-neural-networks-textbook/
# matching the GitHub Pages deployment path.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="/tmp/gnn-serve/graph-neural-networks-textbook"
mkdocs build --config-file "$ROOT/mkdocs.yml" -d "$OUT"
exec python3 -m http.server 8001 --directory /tmp/gnn-serve
