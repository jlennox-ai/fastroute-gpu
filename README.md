# fastroute-gpu

A fork of [fastroute](https://github.com/parisxmas/fastroute) that uses NVIDIA CUDA which **routes ~2x faster.**

## What is different from fastroute

- **`--gpu`**: Use CUDA to help path routes. Details: [docs/GPU.md](docs/GPU.md).
- **CPU optimization**: Additional optimizations to the required CPU code paths.
- **PGO builds**: Further optimizing the CPU paths.

## Comparison

Board with 6 layers (3 signal layers), 362 parts, 1754 connections of which 1053
are left to route, 3 router threads, no fanout or optimizer.
Intel Core i9-10900K, GeForce RTX 3080, Windows 11.

| | fastroute 0.1.14 (original) | fastroute-gpu (CPU only) | fastroute-gpu (`--gpu`) |
|---|---|---|---|
| First pass | 211 s | 168 s | 58 s |
| Down to 1 unrouted | 1014 s | 819 s | 268 s |
| **Whole run** | **1058 s** | **824 s** | **439 s** |

*\* CPU only is AVX256 w/ PGO*

## Download and run

1. Download: [Releases](https://github.com/jlennox-ai/fastroute-gpu/releases)
2. NVIDIA's NVRTC library.

   **Windows**: Download [NVIDIA's NVRTC zip](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.9.86-archive.zip)
   and extract the `.dll` files from its `bin` directory next to `fastroute.exe`.

   **Linux**: Download NVIDIA's NVRTC archive ([x86-64](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz), [ARM64](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-sbsa/cuda_nvrtc-linux-sbsa-12.9.86-archive.tar.xz))
   and extract the files from its `lib` directory next to `fastroute`. In the directory of
   `fastroute`: `tar -xJf cuda_nvrtc-linux-*.tar.xz --strip-components=2 --wildcards '*/lib/*'`
3. Run on your project: `fastroute -de board.dsn -do board.ses --gpu`

Without `--gpu`, fastroute-gpu does not need NVRTC and routes on the CPU.

Sorry, licenses stopped us from including nvidia's libraries in our releases.

## Build

```sh
cargo build --release -p fastroute --features gpu    # needs no CUDA toolkit to build
```

## Upstream?

Anyone is welcome to upstream these changes into the main project. I wanted to shotgun blast LLM changes to speedup
a personal project and not take on codebase management.

## License

GPL-3.0-or-later ([LICENSE](LICENSE)), as fastroute and Freerouting. See [NOTICE](NOTICE).

🤖 These enhancements were created by AI.