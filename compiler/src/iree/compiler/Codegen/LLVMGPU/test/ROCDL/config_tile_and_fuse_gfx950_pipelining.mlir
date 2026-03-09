// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Test: Large matmul on gfx950 should produce valid tile sizes
// (Verifies the adjustSeeds refactor doesn't regress existing behavior)
func.func @matmul_large_f16(%lhs: tensor<4096x4096xf16>, %rhs: tensor<4096x4096xf16>) -> tensor<4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4096x4096xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
                          outs(%fill : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %result : tensor<4096x4096xf32>
}

// CHECK-LABEL: func.func @matmul_large_f16
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_num_stages = 2
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  CHECK-SAME:     subgroup = [4, 8, 0]
//  CHECK-SAME:     workgroup = [256, 256, 0]
