#!/usr/bin/env bash
# Stage a Kaggle Dataset from locally fetched data.  OWNER: Avneet.
#
#   bash scripts/kaggle/stage_dataset.sh [OUT_DIR]
#
# Copies only what the training path reads. Run the fetch steps first:
#   python -m src.data.drivaernet --fetch
#   python -m src.data.drivaerml --runs 30 --download
#   python -m src.data.preprocess --combine
#
# Then upload OUT_DIR as a Kaggle Dataset. See README.md in this directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-/tmp/aero-kaggle}"
mkdir -p "$OUT"

copy() {
  if [ -e "$ROOT/$1" ]; then
    mkdir -p "$OUT/$(dirname "$2")"
    cp -R "$ROOT/$1" "$OUT/$2"
    echo "  + $2"
  else
    echo "  - $1 missing, skipped"
  fi
}

echo "staging into $OUT"
copy data/drivaernet/parametric.csv  parametric.csv
copy data/drivaernet/raw             raw
copy data/drivaerml/combined         combined

echo
echo "size: $(du -sh "$OUT" | cut -f1)"
echo
echo "next:"
echo "  cd $OUT && kaggle datasets init -p ."
echo "  # set title and id in dataset-metadata.json, then:"
echo "  kaggle datasets create -p . --dir-mode zip"
