//! CUDA miner in `cuda-miner/`: build with `make` on first run; optional passthrough of CLI args to the binary.

use std::path::PathBuf;
use std::process::Command;

pub fn cuda_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("cuda-miner")
        .join("src")
}

pub fn cuda_bin() -> PathBuf {
    cuda_src_dir().join("debug").join("cuda_ed25519_vanity")
}

fn cuda_miner_needs_build(src: &std::path::Path, bin: &std::path::Path) -> bool {
    if !bin.is_file() {
        return true;
    }
    match Command::new("make")
        .current_dir(src)
        .args(["-q", "V=debug", "debug/cuda_ed25519_vanity"])
        .status()
    {
        Ok(s) => !s.success(),
        Err(_) => true,
    }
}

/// Build if missing or stale. Panics with message to stderr on fatal error.
pub fn ensure_cuda_miner() {
    let src = cuda_src_dir();
    if !src.join("Makefile").is_file() {
        eprintln!(
            "CUDA miner sources not found (expected Makefile under {}).",
            src.display()
        );
        std::process::exit(1);
    }
    let bin = cuda_bin();
    if !cuda_miner_needs_build(&src, &bin) {
        return;
    }
    eprintln!("Building CUDA miner (first run or sources changed)…");
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get().max(1))
        .unwrap_or(1);
    let st = Command::new("make")
        .current_dir(&src)
        .args([
            "-j",
            &jobs.to_string(),
            "V=debug",
            "debug/cuda_ed25519_vanity",
        ])
        .status()
        .unwrap_or_else(|e| {
            eprintln!("failed to start `make`: {e}");
            std::process::exit(1);
        });
    if !st.success() || !bin.is_file() {
        eprintln!(
            "CUDA build failed. Need `nvcc` + `make`. Try:\n  cd {}\n  make V=debug debug/cuda_ed25519_vanity",
            src.display()
        );
        std::process::exit(1);
    }
}

pub fn ld_library_path_value() -> String {
    let bin = cuda_bin();
    let lib_dir = bin.parent().expect("cuda bin has parent");
    match std::env::var("LD_LIBRARY_PATH") {
        Ok(prev) if !prev.is_empty() => format!("{}:{}", lib_dir.display(), prev),
        _ => lib_dir.display().to_string(),
    }
}

/// Forward argv to miner; returns process exit code.
pub fn run_passthrough(args: &[String]) -> i32 {
    let bin = cuda_bin();
    let ld = ld_library_path_value();
    Command::new(&bin)
        .args(args)
        .env("LD_LIBRARY_PATH", ld)
        .status()
        .map(|s| s.code().unwrap_or(1))
        .unwrap_or_else(|e| {
            eprintln!("failed to spawn {}: {e}", bin.display());
            1
        })
}
