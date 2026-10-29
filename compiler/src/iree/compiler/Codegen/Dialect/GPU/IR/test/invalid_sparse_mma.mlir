// RUN: iree-opt -verify-diagnostics %s

module {
  func.func @invalid_sparse_mma_selector() attributes {
    // expected-error @+1 {{sparsity selector must be 0 or 1}}
    mma_types = #iree_gpu.sparse_mma_layout<NV_MMA_SP_F32_16x8x32_F16, sparsity_selector = 2>
  } {
    return
  }
}
