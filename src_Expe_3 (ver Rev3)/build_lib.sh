#!/bin/bash
# =============================================================================
# Build script for the fpli_hv shared library (macOS and Linux)
#
# Compiles hv.c and avl.c from the hv-1.3-src distribution into a dynamic
# library placed in src/:
#   - macOS : src/libfpli_hv.dylib
#   - Linux : src/libfpli_hv.so
#
# Windows users: use build_lib.bat (MinGW) or run this script under WSL.
#
# Run once from the project root before launching mainExp1_UKP.jl or
# mainExp1_UFLP.jl:
#   bash build_lib.sh
# =============================================================================

set -e

SRC="src HV/hv-1.3-src"

if [[ "$OSTYPE" == "darwin"* ]]; then
    OUT="src/libfpli_hv.dylib"
else
    OUT="src/libfpli_hv.so"
fi

cc -O3 -DVARIANT=4 -DDEBUG=0 \
   -shared -fPIC \
   "$SRC/hv.c" "$SRC/avl.c" \
   -o "$OUT"

echo "Built: $OUT"
