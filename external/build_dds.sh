#!/bin/bash
# Build the modified DDS library as a shared library for macOS ARM64
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DDS_SRC="$SCRIPT_DIR/dds/library/src"
OUT="$SCRIPT_DIR/libdds_spades.dylib"

# Collect all source files (excluding tests and examples)
SOURCES=$(find "$DDS_SRC" -name "*.cpp" | tr '\n' ' ')

echo "Building DDS with spades_broken support..."
echo "Source dir: $DDS_SRC"
echo "Output: $OUT"

clang++ \
    -std=c++17 \
    -O3 \
    -march=native \
    -shared \
    -fPIC \
    -DDDS_THREADS_STL \
    -I"$DDS_SRC" \
    -I"$DDS_SRC/api" \
    -I"$DDS_SRC/moves" \
    -I"$DDS_SRC/system" \
    -I"$DDS_SRC/trans_table" \
    -I"$DDS_SRC/utility" \
    -I"$DDS_SRC/heuristic_sorting" \
    -I"$DDS_SRC/lookup_tables" \
    -I"$DDS_SRC/solver_context" \
    $SOURCES \
    -o "$OUT" \
    -lpthread

echo "Built successfully: $OUT ($(du -h "$OUT" | cut -f1))"
