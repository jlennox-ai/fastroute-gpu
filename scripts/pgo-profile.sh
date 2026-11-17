#!/bin/sh
# Retrains the committed PGO profile (pgo/fastroute.profdata) that the release workflow and the
# local builds (.cargo/config.toml) use. Run it before pushing code changes: the profile is
# matched to functions by a hash of their control flow, so it loses its effect on code that
# changed after it was trained.
#
#   scripts/pgo-profile.sh          (Git Bash; needs an NVIDIA GPU with NVRTC on the PATH,
#                                    `rustup component add llvm-tools-preview`, and the
#                                    gametank test boards in $PGO_BOARDS)
#
# 1. builds an instrumented binary with the gpu feature,
# 2. runs it on the 4-layer gametank boards and the synthetic test boards (the 6-layer board is
#    held out for benchmarking); the runs are time-capped, the profile only needs coverage,
# 3. merges the profiles into pgo/fastroute.profdata and records the rustc that wrote it in
#    pgo/rustc-version (the workflow installs that version: a profile is only read correctly
#    by the compiler version it was made with).
# Every build that uses the profile must pass an explicit --target, as this one does: cargo
# hashes it into the symbol names, and a profile only applies to functions of the same name.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BOARDS=${PGO_BOARDS:-$ROOT/../gametank/test-boards}
SYNTH=$ROOT/crates/fr-io/testdata/dsn
WORK=${PGO_WORK:-${TEMP:-/tmp}/fastroute-pgo}
HOST=$(rustc -vV | sed -n 's/^host: //p')
PROFDATA=$(rustc --print sysroot)/lib/rustlib/$HOST/bin/llvm-profdata
PLACED=$BOARDS/4layer-placed-with-plane-vias/gametank_compact.dsn
PARTS=$BOARDS/4layer-parts-only/gametank_compact.dsn
GAMETANK="--router.automatic_neckdown=false --router.copper_to_edge_clearance_um=320 --router.neck_width_um=127 --no-neckdown-classes=Input"

[ -x "$PROFDATA" ] || [ -x "$PROFDATA.exe" ] || { echo "llvm-profdata not found: rustup component add llvm-tools-preview" >&2; exit 1; }
[ -f "$PLACED" ] || { echo "training boards not found in $BOARDS (set PGO_BOARDS)" >&2; exit 1; }

rm -rf "$WORK/profiles" "$WORK/out"
mkdir -p "$WORK/profiles" "$WORK/out"

echo "== instrumented build"
RUSTFLAGS="-C profile-generate=$WORK/profiles -C target-cpu=x86-64-v3" CARGO_TARGET_DIR="$WORK/gen" \
  cargo build --release -p fastroute --features gpu --target "$HOST" --manifest-path "$ROOT/Cargo.toml"
BIN=$WORK/gen/$HOST/release/fastroute

FAILED=$WORK/out/failed
train() {
  name=$1
  shift
  start=$(date +%s)
  if "$BIN" -do "$WORK/out/$name.ses" "$@" > "$WORK/out/$name.log" 2>&1 < /dev/null; then
    echo "$name $(( $(date +%s) - start ))s"
  else
    echo "$name FAILED, see $WORK/out/$name.log" >&2
    echo "$name" >> "$FAILED"
  fi
}

echo "== training"
# the gametank routing setup: parallel autorouter, no fanout/optimizer, without and with the GPU guide
train ar_par -de "$PLACED" $GAMETANK --router.autorouter.max_threads=3 --router.optimizer.enabled=false --router.fanout.enabled=false --max-time=240 &
train ar_gpu -de "$PLACED" $GAMETANK --router.autorouter.max_threads=3 --router.optimizer.enabled=false --router.fanout.enabled=false --gpu --guide-cell-um=150 --max-time=180 &
# default pipeline: fanout + sequential autorouter
train fanout_seq -de "$PARTS" $GAMETANK --router.autorouter.max_threads=1 --router.fanout.max_passes=3 --max-time=240 &
wait
# optimizer from a finished route: parallel and java-compat
train opt_par -de "$PLACED" $GAMETANK --initial-session="$BOARDS/reference-results/pass2_finished.ses" --router.fanout.enabled=false --router.optimizer.max_threads=3 --max-time=120 &
train opt_seq -de "$PLACED" $GAMETANK --initial-session="$BOARDS/reference-results/pass1_from_scratch.ses" --router.fanout.enabled=false --router.autorouter.max_passes=1 --router.optimizer.max_threads=1 --max-time=120 &
# full pipeline on the small synthetic boards (default and parity modes)
for board in synth_rules synth_order synth_order_placed synth_planes; do
  train "$board" -de "$SYNTH/$board.dsn"
  train "parity_$board" -de "$SYNTH/$board.dsn" --parity
done
wait
[ ! -f "$FAILED" ] || { echo "training runs failed: $(tr '\n' ' ' < "$FAILED")" >&2; exit 1; }

mkdir -p "$ROOT/pgo"
"$PROFDATA" merge -o "$ROOT/pgo/fastroute.profdata" "$WORK/profiles"
rustc -V | sed 's/^rustc \([0-9.]*\).*/\1/' > "$ROOT/pgo/rustc-version"
echo "wrote pgo/fastroute.profdata ($(du -h "$ROOT/pgo/fastroute.profdata" | cut -f1)), rustc $(cat "$ROOT/pgo/rustc-version")"
