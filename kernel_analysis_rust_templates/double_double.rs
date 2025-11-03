//! GPU Kernels: double_double
//! Auto-generated from: double_double.ptx
//!
//! IMPORTANT: This is a template. Verify types and add implementation.

#![no_std]
#![feature(abi_ptx)]

use cuda_std::*;

#[kernel]
pub unsafe fn _Z12dd_array_addP7dd_realPKS_S2_i(
) {
    // Kernel implementation
    let tid = thread::index_1d();
    // TODO: Add kernel logic
}

#[kernel]
pub unsafe fn _Z20dd_matrix_vector_mulP10dd_complexPKS_S2_i(
) {
    // Kernel implementation
    let tid = thread::index_1d();
    // TODO: Add kernel logic
}

#[kernel]
pub unsafe fn _Z23dd_deterministic_reduceP7dd_realPKS_i(
) {
    // Kernel implementation
    let tid = thread::index_1d();
    // TODO: Add kernel logic
}

#[kernel]
pub unsafe fn _Z18test_dd_arithmeticv(
) {
    // Kernel implementation
    let tid = thread::index_1d();
    // TODO: Add kernel logic
}

