# fastroute-gpu

A fork of [fastroute](https://github.com/parisxmas/fastroute), the Rust port of the
[Freerouting](https://github.com/freerouting/freerouting) PCB autorouter, that uses an NVIDIA
GPU to make the maze search smaller. Everything else (options, KiCad plugin, documentation) is
fastroute's: see [its README](https://github.com/parisxmas/fastroute#readme).

## What is different from fastroute

- **`--gpu`**: for each connection a cost-to-go field is computed on the GPU (CUDA) and guides
  the maze search, which then expands far fewer rooms. Routing, shoving and DRC stay on the CPU
  and unchanged. How it works and what was measured: [docs/GPU.md](docs/GPU.md).
- **`--guide=cpu`**: the same field from a CPU reference solver (slow; for testing).
- **`FASTROUTE_MAZE_STATS=FILE`**: per-connection statistics of the maze search.
- **Session files KiCad's DRC accepts after resuming** (`--initial-session`): no zero-length or
  sub-minimum-width bridge stubs, and KiCad's rule for via-to-pad contact.
- **Release builds are profile-guided (PGO)** for every target, and come in two x86-64
  variants: `x86-64` for any processor, `AVX256` for x86-64-v3 (AVX2; Haswell / Excavator and
  newer).

## Comparison

GameTank compact board, 6 layers (3 signal layers), 362 parts, 1754 connections of which 1053
are left to route, 3 router threads, no fanout or optimizer. Intel Core i9-10900K, NVIDIA
GeForce RTX 3080, Windows 11. One run each; all three stop at pass 70 (no progress in the last
10 passes) with one connection left unrouted (a different one with `--gpu`).

| | fastroute 0.1.14 ([release download](https://github.com/parisxmas/fastroute/releases/tag/v0.1.14)) | fastroute-gpu, CPU only | fastroute-gpu `--gpu` |
|---|---|---|---|
| First pass | 211 s | 180 s | 62 s |
| Down to 1 unrouted | 1014 s | 866 s | 301 s |
| Whole run | 1058 s | 901 s | **509 s** |
| Unrouted | 1 | 1 | 1 |
| Vias | 2001 | 2001 | 2006 |
| Track length | 33.40 m | 33.40 m | 33.49 m |

The CPU-only column is the AVX256 PGO build without `--gpu`: same result as fastroute, 15 %
faster. With `--gpu` the run takes half the time (2.1× faster; 3.4× to the last connection it
routes) for 0.3 % more track.

## Download and run

[Releases](https://github.com/jlennox-ai/fastroute-gpu/releases): Windows (x86-64, AVX256) and
Linux (x86-64, AVX256, ARM64). Windows on ARM and macOS have no NVIDIA driver; use
[fastroute's builds](https://github.com/parisxmas/fastroute/releases) there.

```sh
fastroute -de board.dsn -do board.ses --gpu
```

### What you need for `--gpu`

1. An NVIDIA GPU with a current NVIDIA driver.
2. NVIDIA's NVRTC library.

   **Windows**: install [CUDA Toolkit 12](https://developer.nvidia.com/cuda-toolkit-archive)

   **Ubuntu 24.04 or newer** (multiverse, on by default): `sudo apt install libnvrtc12`

   **Debian 13 or newer**:
   1. Open `/etc/apt/sources.list.d/debian.sources` as root (if it does not exist, the system
      was upgraded from Debian 12: use `/etc/apt/sources.list`).
   2. Add `non-free` to the end of every `Components:` line (old file: every `deb` line), for
      example `Components: main non-free-firmware non-free`.
   3. `sudo apt update`
   4. `sudo apt install libnvrtc12`

   **Other Linux** (and older Ubuntu/Debian, which only package CUDA 11): install the CUDA
   Toolkit 12 from the same page. If fastroute then reports that NVRTC was not found, run
   `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`. Use CUDA 12;
   CUDA 13 (for example Arch's `cuda` package) is untested.

If you do not want the whole toolkit on Windows, copy `nvrtc64_120_0.dll` and
`nvrtc-builtins64_12*.dll` from a machine that has it (`C:\Program Files\NVIDIA GPU Computing
Toolkit\CUDA\v12.x\bin`) into the folder of `fastroute.exe`.

Without `--gpu`, fastroute needs none of this and routes on the CPU.

## Build

```sh
cargo build --release -p fastroute --features gpu    # needs no CUDA toolkit to build
```

## License

GPL-3.0-or-later ([LICENSE](LICENSE)), as fastroute and Freerouting. See [NOTICE](NOTICE).

🤖 These enhancements were created by AI.