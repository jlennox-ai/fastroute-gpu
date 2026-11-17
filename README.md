# fastroute-gpu

A fork of [fastroute](https://github.com/parisxmas/fastroute) that uses NVIDIA CUDA, **routing ~2x faster.**

## What is different from fastroute

- **`--gpu`**: Use CUDA to help path routes. Details: [docs/GPU.md](docs/GPU.md).
- **PGO builds**: Even the CPU parts of this build are faster than the originals.

## Comparison

Board with 6 layers (3 signal layers), 362 parts, 1754 connections of which 1053
are left to route, 3 router threads, no fanout or optimizer.
Intel Core i9-10900K, GeForce RTX 3080, Windows 11.

| | fastroute 0.1.14 (original) | fastroute-gpu (CPU only) | fastroute-gpu (`--gpu`) |
|---|---|---|---|
| First pass | 211 s | 168 s | 58 s |
| Down to 1 unrouted | 1014 s | 819 s | 268 s |
| Whole run | 1058 s | 824 s | **439 s** |
| Vias | 2001 | 2001 | 2006 |
| Track length | 33.40 m | 33.40 m | 33.49 m |

*\* CPU only is AVX256 w/ PGO*

## Download and run

- Download: [Releases](https://github.com/jlennox-ai/fastroute-gpu/releases)
- `fastroute -de board.dsn -do board.ses --gpu`

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