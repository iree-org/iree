// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=WMMA

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

// WMMA:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// WMMA-SAME: workgroup_size = [128, 1, 1]
// WMMA-SAME: subgroup_size = 32

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @wmma_matmul_1024x1024x1024() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
  %5 = tensor.empty() : tensor<1024x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  return
}

// WMMA-LABEL: func.func @wmma_matmul_1024x1024x1024()
// WMMA: linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
// WMMA-SAME:                           mma_kind = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>
// WMMA-SAME:                           reduction =  [0, 0, 64]
// WMMA-SAME:                           subgroup_m_count = 2
// WMMA-SAME:                           subgroup_n_count = 2
// WMMA-SAME:                           workgroup =  [64, 128, 0]
