# Kernel Integration Guide

## Generated Rust Kernel Templates

This directory contains auto-generated Rust kernel templates based on PTX analysis.

## Next Steps

1. **Review Parameter Types**: Verify pointer types (const vs mut, element types)
2. **Add Kernel Logic**: Implement the actual computation
3. **Add Cargo.toml**: Configure nvptx64-nvidia-cuda target
4. **Build PTX**: `cargo build --release --target nvptx64-nvidia-cuda`
5. **Test Kernels**: Create unit tests with cudarc

## Example Cargo.toml

```toml
[package]
name = "foundation-cuda-kernels"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
cuda-std = "0.3"

[profile.release]
lto = true
opt-level = 3
codegen-units = 1
```

## Build Command

```bash
rustup target add nvptx64-nvidia-cuda
cargo build --release --target nvptx64-nvidia-cuda
```
