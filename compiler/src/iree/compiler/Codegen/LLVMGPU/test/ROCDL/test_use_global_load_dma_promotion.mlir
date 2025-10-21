// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-llvmgpu-use-direct-load \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Test that --iree-llvmgpu-use-direct-load generates use_global_load_dma in promotion_types
// This verifies the ConfigUtils.cpp changes that calculate subgroup sizes automatically.

func.func @simple_matmul(%lhs: tensor<256x128xf16>, %rhs: tensor<128x256xf16>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<256x256xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<256x128xf16>, tensor<128x256xf16>)
                          outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func.func @simple_matmul
//       CHECK:   linalg.matmul
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     promotion_types = [#iree_gpu.use_global_load_dma<subgroup = [4, 128]>, #iree_gpu.use_global_load_dma<subgroup = [4, 128]>]

// -----

// Test with f32 elements and exact fit for innermost dimension (64 elements)
// For gfx942:
//   - subgroup_size = 64
//   - dma_sizes = [32] bits
//   - element type = f32 (32 bits)
//   - innermost dim = 64 elements = 2048 bits
//   - Required size for 32-bit DMA: 64 * (32 / 32) = 64 elements = 2048 bits
//   - Since 2048 == 2048 and is a multiple, this should generate subgroup = [4, 64]

func.func @matmul_f32_exact(%lhs: tensor<64x64xf32>, %rhs: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x128xf32>)
                          outs(%fill : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @matmul_f32_exact
//       CHECK:   linalg.matmul
//  CHECK-SAME:     promotion_types = [#iree_gpu.use_global_load_dma<subgroup = [4, 64]>, #iree_gpu.use_global_load_dma<subgroup = [4, 64]>]
