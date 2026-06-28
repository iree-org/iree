// RUN: iree-opt --mlir-print-local-scope --iree-gpu-test-target=sm_120 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=CONFIG
// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_120 \
// RUN:   --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-nvvm-lowering-pipeline %s | FileCheck %s --check-prefix=NVVM

// Test that sm_120 selects and lowers BF16 matmuls through NVIDIA mma.sync,
// mirroring the sm_80 BF16 coverage in bf16_mma_sm80.mlir.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
// Drives BF16xbf16xf32 matmul selection through the NVIDIA mma.sync path on sm_120.
func.func @matmul_256x256x256_bf16_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xbf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xbf16>> -> tensor<256x256xbf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xbf16>> -> tensor<256x256xbf16>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xbf16>, tensor<256x256xbf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// CONFIG-LABEL: func.func @matmul_256x256x256_bf16_f32(
//       CONFIG:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CONFIG-SAME:     mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_BF16>
//  CONFIG-SAME:     promote_operands = [0, 1]

// NVVM-LABEL: llvm.func @matmul_256x256x256_bf16_f32(
//       NVVM:   nvvm.mma.sync
//  NVVM-SAME:     multiplicandAPtxType = #nvvm.mma_type<bf16>
//  NVVM-SAME:     multiplicandBPtxType = #nvvm.mma_type<bf16>
//  NVVM-SAME:     shape = #nvvm.shape<m = 16, n = 8, k = 16>
//  NVVM-SAME:     : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
