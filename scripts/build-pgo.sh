#!/bin/sh
# Profile-guided optimized release build of fastroute, trained on the machine that builds it.
# The release workflow uses it for the targets the committed profile does not fit
# (pgo/fastroute.profdata is tied to x86_64-pc-windows-msvc, see scripts/pgo-profile.sh).
#
#   scripts/build-pgo.sh [--native]
#
# 1. builds an instrumented binary (target dir $PGO_DIR/gen),
# 2. runs it on a few benchmark boards (training set below),
# 3. merges the profiles with llvm-profdata and builds the optimized binary
#    ($PGO_DIR/use/release/fastroute).
# --native additionally passes `-C target-cpu=native` (the binary then only runs on CPUs like
# the build machine). Needs `rustup component add llvm-tools-preview` and the benchmark
# fixtures in reference/freerouting. PGO does not change results (same code, different
# optimization decisions; float semantics are unaffected).
#
# Environment: PGO_RUSTFLAGS (more compiler flags, e.g. "-C target-cpu=x86-64-v3"),
# PGO_FEATURES (cargo features of the fastroute crate, e.g. "gpu"), PGO_DIR.
set -eu
# cargo and rustc are native programs: on Windows (Git Bash) they need C:/... paths
native_path() {
  if command -v cygpath > /dev/null; then cygpath -m "$1"; else echo "$1"; fi
}
ROOT=$(native_path "$(cd "$(dirname "$0")/.." && pwd)")
PGO_DIR=$(native_path "${PGO_DIR:-$ROOT/target/pgo}")
FX=$ROOT/reference/freerouting/scripts/benchmark/fixtures
EXTRA=${PGO_RUSTFLAGS:-}
FEATURES=${PGO_FEATURES:-}
if [ "${1:-}" = "--native" ]; then EXTRA="$EXTRA -C target-cpu=native"; fi
PROFDATA=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-profdata
[ -x "$PROFDATA" ] || [ -x "$PROFDATA.exe" ] || { echo "llvm-profdata not found: rustup component add llvm-tools-preview" >&2; exit 1; }
[ -d "$FX" ] || { echo "benchmark fixtures not found: $FX" >&2; exit 1; }

rm -rf "$PGO_DIR/profiles"; mkdir -p "$PGO_DIR/profiles"
echo "== instrumented build"
RUSTFLAGS="-C profile-generate=$PGO_DIR/profiles $EXTRA" CARGO_TARGET_DIR="$PGO_DIR/gen" \
  cargo build --release -p fastroute --features "$FEATURES" --manifest-path "$ROOT/Cargo.toml"
BIN="$PGO_DIR/gen/release/fastroute"
OUT=$(native_path "$(mktemp -d)")
train() {
  echo "== training: $*"
  "$BIN" -do "$OUT/x.ses" "$@" > "$OUT/train.log" 2>&1 < /dev/null || { cat "$OUT/train.log" >&2; echo "training run failed: $*" >&2; exit 1; }
}
train --parity -de "$FX/DAC2020_boards/DAC2020_bm06.dsn"
train --parity -de "$FX/DAC2020_boards/DAC2020_bm11.dsn"
train --parity -de "$FX/DAC2020_boards/DAC2020_bm11.dsn" --optimizer-mode=parallel --router.optimizer.max_threads=4
train --parity -de "$FX/KiCad_10_demos/interf_u.dsn" --router.optimizer.enabled=false
train --parity -de "$FX/KiCad_10_demos/CM5_MINIMA_3.dsn" --router.optimizer.enabled=false -mp 1
train --parity -de "$FX/KiCad_10_demos/multichannel_mixer.dsn" --router.optimizer.enabled=false
# the maze search with a guide field (the CPU solver stands in for the GPU, which build machines lack)
train -de "$FX/DAC2020_boards/DAC2020_bm06.dsn" --guide=cpu --router.optimizer.enabled=false -mp 2
rm -rf "$OUT"
"$PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR/profiles"
echo "== optimized build"
RUSTFLAGS="-C profile-use=$PGO_DIR/merged.profdata -Cllvm-args=-pgo-warn-missing-function=false $EXTRA" \
  CARGO_TARGET_DIR="$PGO_DIR/use" cargo build --release -p fastroute --features "$FEATURES" --manifest-path "$ROOT/Cargo.toml"
echo "built $PGO_DIR/use/release/fastroute"
