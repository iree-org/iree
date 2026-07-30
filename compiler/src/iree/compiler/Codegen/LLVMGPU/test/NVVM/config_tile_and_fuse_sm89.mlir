// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=sm_89 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN:   | FileCheck %s

// Tests for Ada (sm_89) FP8 mma.sync strategy selection. F16/BF16 coverage
// is shared with sm_80 via config_tile_and_fuse_sm80.mlir.

// -----

// Test that F8E4M3FN matmul selects NV_MMA_SYNC_F32_16x8x32_F8E4M3FN.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f8e4m3fn_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>> -> tensor<256x256xf8E4M3FN>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>> -> tensor<256x256xf8E4M3FN>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf8E4M3FN>, tensor<256x256xf8E4M3FN>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: func.func @matmul_256x256x256_f8e4m3fn_f32(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [128, 1, 1] subgroup_size = 32
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x32_F8E4M3FN>
//  CHECK-SAME:     promote_operands = [0, 1]

// -----

// Test that F8E5M2 matmul selects NV_MMA_SYNC_F32_16x8x32_F8E5M2.

#pipeline_layout_e5m2 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f8e5m2_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_e5m2) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E5M2>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_e5m2) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E5M2>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_e5m2) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E5M2>> -> tensor<256x256xf8E5M2>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E5M2>> -> tensor<256x256xf8E5M2>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf8E5M2>, tensor<256x256xf8E5M2>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: func.func @matmul_256x256x256_f8e5m2_f32(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [128, 1, 1] subgroup_size = 32
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x32_F8E5M2>
//  CHECK-SAME:     promote_operands = [0, 1]
