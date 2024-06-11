// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 \
// RUN:   --iree-codegen-llvmgpu-use-tile-and-fuse --iree-codegen-llvmgpu-use-vector-distribution=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @expanded_matmul_transpose_b(%lhs: tensor<2x64x2048xf16>, %rhs: tensor<10x64x2048xf16>) -> tensor<2x10x64x64xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<2x10x64x64xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf16>) -> tensor<2x10x64x64xf16>
  %7 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %8, %out : f16
    linalg.yield %9 : f16
  } -> tensor<2x10x64x64xf16>
  return %7 : tensor<2x10x64x64xf16>
}

// CHECK: #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>
// CHECK-LABEL: func.func @expanded_matmul_transpose_b

// Verify that the fill does not have the lowering config propagated to it.
//       CHECK:   linalg.fill ins

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
//  CHECK-SAME:     reduction = [0 : index, 0 : index, 0 : index, 0 : index, 8 : index]
//  CHECK-SAME:     subgroup = [0 : index, 0 : index, 4 : index, 1 : index, 0 : index]
//  CHECK-SAME:     workgroup = [1 : index, 1 : index, 64 : index, 64 : index, 0 : index]

// -----

func.func @mfma_matmul_1024x1024x1024(%lhs: tensor<1024x1024xf16>, %rhs: tensor<1024x1024xf16>) -> tensor<1024x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %5 = tensor.empty() : tensor<1024x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %7 = linalg.matmul ins(%lhs, %rhs : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %7 : tensor<1024x1024xf32>
}

// CHECK: #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 2, 1] subgroup_size = 64>
// CHECK-LABEL: func.func @mfma_matmul_1024x1024x1024

// Verify that the fill does not have the lowering config propagated to it.
//       CHECK:   linalg.fill ins

//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
//  CHECK-SAME:     reduction = [0 : index, 0 : index, 4 : index]
//  CHECK-SAME:     subgroup = [2 : index, 4 : index, 0 : index]
//  CHECK-SAME:     workgroup = [64 : index, 128 : index, 0 : index]
