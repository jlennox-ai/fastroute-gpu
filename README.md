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

Board with 6 layers (3 signal layers), 362 parts, 1754 connections of which 1053
are left to route, 3 router threads, no fanout or optimizer. Intel Core i9-10900K,
GeForce RTX 3080, Windows 11.

| | fastroute 0.1.14 (original) | fastroute-gpu, CPU only | fastroute-gpu `--gpu` |
|---|---|---|---|
| First pass | 211 s | 180 s | 62 s |
| Down to 1 unrouted | 1014 s | 866 s | 301 s |
| Whole run | 1058 s | 901 s | **509 s** |
| Unrouted | 1 | 1 | 1 |
| Vias | 2001 | 2001 | 2006 |
| Track length | 33.40 m | 33.40 m | 33.49 m |

*\* CPU only is AVX256 w/ PGO without `--gpu`.*

## Download and run

Download: [Releases](https://github.com/jlennox-ai/fastroute-gpu/releases)

```sh
fastroute -de board.dsn -do board.ses --gpu
```

### What you need for `--gpu`

1. An NVIDIA GPU with a current NVIDIA driver.
2. NVIDIA's NVRTC library.

   **Windows**: download
   [NVIDIA's NVRTC zip](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.9.86-archive.zip)
   and extract the `.dll` files from its `bin` folder next to `fastroute.exe`.

   **Linux**: download NVIDIA's NVRTC archive
   ([x86-64](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz),
   [ARM64](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-sbsa/cuda_nvrtc-linux-sbsa-12.9.86-archive.tar.xz))
   and extract the files from its `lib` folder next to `fastroute`. In the folder of
   `fastroute`: `tar -xJf cuda_nvrtc-linux-*.tar.xz --strip-components=2 --wildcards '*/lib/*'`

Without `--gpu`, fastroute needs none of this and routes on the CPU.

Sorry, licenses stopped us from including nvidia's libraries in our releases.

## Build

```sh
cargo build --release -p fastroute --features gpu    # needs no CUDA toolkit to build
```

## License

GPL-3.0-or-later ([LICENSE](LICENSE)), as fastroute and Freerouting. See [NOTICE](NOTICE).

🤖 These enhancements were created by AI.