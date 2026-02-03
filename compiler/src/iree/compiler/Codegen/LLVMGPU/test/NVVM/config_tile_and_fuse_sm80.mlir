// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=sm_80 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Test that sm_80 selects LLVMGPUTileAndFuse with NV_MMA_SYNC intrinsics by default.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f16_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: func.func @matmul_256x256x256_f16_f32(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 32

//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 8]
//  CHECK-SAME:     subgroup = [2, 4, 0]
//  CHECK-SAME:     workgroup = [64, 64, 0]

// -----

// Test F16 output matmul also selects NV_MMA_SYNC_F16 intrinsic with TileAndFuse.

#pipeline_layout_f16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f16_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf16>) -> tensor<256x256xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
  return
}

// CHECK-LABEL: func.func @matmul_256x256x256_f16_f16(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 32

//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F16_16x8x16_F16>

// -----

// Test that matmul_accumulate sets convert_acc_gemm for NV_MMA_SYNC.

#pipeline_layout_acc = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_accumulate_256x256x256_f16_f32() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_acc) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_acc) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_acc) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<256x256xf32>> -> tensor<256x256xf32>
  %6 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: func.func @matmul_accumulate_256x256x256_f16_f32(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 32

//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     convert_acc_gemm
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>
