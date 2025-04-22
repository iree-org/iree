// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline)" --iree-gpu-test-target=gfx942 \
// RUN:   --split-input-file %s | FileCheck %s

// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-rocdl-configuration-pipeline)" --iree-gpu-test-target=gfx942 \
// RUN:   --split-input-file %s | FileCheck %s

// Make sure that the GPU configuration pipelines generalize named ops, e.g., linalg.matmul_transpose_b to linalg.generic.

// CHECK:      linalg.fill
// CHECK-NEXT: linalg.generic
// CHECK-NOT:  linalg.matmul_transpose_b

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @warp_reduction_large_vector() {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c394240 = arith.constant 394240 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c128) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1280xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x1280xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c394240) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1280xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1280xf32>> -> tensor<1x1280xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x1280xf32>> -> tensor<1280x1280xf32>
  %5 = tensor.empty() : tensor<1x1280xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1280xf32>) -> tensor<1x1280xf32>
  %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<1x1280xf32>, tensor<1280x1280xf32>) outs(%6 : tensor<1x1280xf32>) -> tensor<1x1280xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : tensor<1x1280xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1280xf32>>
  return
}
