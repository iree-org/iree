// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=mi355x@hip \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Verify MI355X-specific (CDNA4) heuristic seed selection produces expected
// configs for large GEMM operations. MI355X targets gfx950 with chip info
// (wgpCount=256), enabling utilization-aware MNT boosting for balanced large
// GEMMs.

// ============================================================================
// LargeGemm — symmetric (4096x4096x4096)
// Balanced K (K == M == N), so MNT gets boosted to 32.
// ============================================================================

func.func @matmul_large_symmetric_f16(
    %arg0: tensor<4096x4096xf16>,
    %arg1: tensor<4096x4096xf16>) -> tensor<4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4096x4096xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
                          outs(%fill : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %result : tensor<4096x4096xf32>
}

// CHECK-LABEL: func.func @matmul_large_symmetric_f16
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  CHECK-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 1]
//  CHECK-SAME:     subgroup = [4, 8, 0]
//  CHECK-SAME:     workgroup = [128, 256, 0]

// -----

// ============================================================================
// LargeGemm — tall-M (21760x3840x3840)
// Balanced K (K == N < M), MNT boost applies.
// ============================================================================

func.func @matmul_large_tall_m_f16(
    %arg0: tensor<21760x3840xf16>,
    %arg1: tensor<3840x3840xf16>) -> tensor<21760x3840xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<21760x3840xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<21760x3840xf32>) -> tensor<21760x3840xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<21760x3840xf16>, tensor<3840x3840xf16>)
                          outs(%fill : tensor<21760x3840xf32>) -> tensor<21760x3840xf32>
  return %result : tensor<21760x3840xf32>
}

// CHECK-LABEL: func.func @matmul_large_tall_m_f16
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  CHECK-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 1]
//  CHECK-SAME:     subgroup = [4, 8, 0]
//  CHECK-SAME:     workgroup = [128, 256, 0]

// -----

// ============================================================================
// LargeGemm — wide-N (4096x8192x2048)
// Balanced K (K < max(M, N)), MNT boost applies.
// ============================================================================

func.func @matmul_large_wide_n_f16(
    %arg0: tensor<4096x2048xf16>,
    %arg1: tensor<2048x8192xf16>) -> tensor<4096x8192xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4096x8192xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4096x8192xf32>) -> tensor<4096x8192xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<4096x2048xf16>, tensor<2048x8192xf16>)
                          outs(%fill : tensor<4096x8192xf32>) -> tensor<4096x8192xf32>
  return %result : tensor<4096x8192xf32>
}

// CHECK-LABEL: func.func @matmul_large_wide_n_f16
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  CHECK-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 1]
//  CHECK-SAME:     subgroup = [4, 8, 0]
//  CHECK-SAME:     workgroup = [128, 256, 0]

// -----

// ============================================================================
// LargeGemm — very tall-M with large K (150000x4096x16384)
// K > max(M, N) so K-dominated — MNT boost does NOT apply.
// Requires padding since 150000 is not a multiple of tile size.
// ============================================================================

func.func @matmul_large_very_tall_m_f16(
    %arg0: tensor<150000x16384xf16>,
    %arg1: tensor<16384x4096xf16>) -> tensor<150000x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<150000x4096xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<150000x4096xf32>) -> tensor<150000x4096xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<150000x16384xf16>, tensor<16384x4096xf16>)
                          outs(%fill : tensor<150000x4096xf32>) -> tensor<150000x4096xf32>
  return %result : tensor<150000x4096xf32>
}

// CHECK-LABEL: func.func @matmul_large_very_tall_m_f16
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//  CHECK-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  CHECK-SAME:     padding = [128, 256, 32]
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 1]
//  CHECK-SAME:     subgroup = [4, 8, 0]
//  CHECK-SAME:     workgroup = [128, 256, 0]
