# uniq-tui

Terminal UI and CLI for a local **CUDA** vanity miner. The GPU program and its build live in `cuda-miner/`.

## Requirements

- Rust (stable)
- NVIDIA GPU and driver
- CUDA toolkit (`nvcc`) and GNU `make` on `PATH`

## Quick start

```bash
git clone https://github.com/bavtika/uniq-tui.git
cd uniq-tui
cargo run --release
```

- **No arguments** — full-screen TUI (menu, prefix/suffix, live log).
- **With arguments** — passed through to `cuda_ed25519_vanity`:

```bash
cargo run --release -- --prefix Abc --case-sensitive true --stop-after-keys 1
```

On first run (or after CUDA sources change), `uniq-tui` runs `make` in `cuda-miner/src/` and produces `cuda-miner/src/debug/cuda_ed25519_vanity` and `libcuda-crypt.so`. `LD_LIBRARY_PATH` is set for that directory.

## Layout

| Path | Purpose |
|------|---------|
| `src/` | Rust UI (`ratatui`) and process launcher |
| `cuda-miner/` | CUDA sources and Makefile |
| `cuda-miner/src/debug/` | Local build output (not committed) |
| `bin/` | Optional install target for `scripts/install-gpu-binary.sh` |
| `scripts/` | Helper scripts |

## Build the miner only

```bash
cd cuda-miner/src
make -j"$(nproc)" V=debug debug/cuda_ed25519_vanity
```

Binary: `cuda-miner/src/debug/cuda_ed25519_vanity`.

## Optional: install binary into `bin/`

```bash
bash scripts/install-gpu-binary.sh --auto
```

Uses `./cuda-miner` in this tree by default; override with `UNIQ_GPU_SOURCE_DIR` or `UNIQ_GPU_BINARY_URL` if needed.

## License

Rust sources at the repository root (`src/`, `Cargo.toml`, etc.) are under the [MIT License](LICENSE). Files under `cuda-miner/` include their own copyright and license notices where applicable.
