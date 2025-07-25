// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-softmax), iree-llvmgpu-select-lowering-strategy,  func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-softmax), iree-llvmgpu-select-lowering-strategy,  func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s --check-prefix=CDNA3

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -3.40282347E+38 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<12x128x40960xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<12x128x40960xf32>> -> tensor<12x128x40960xf32>
  %3 = tensor.empty() : tensor<12x128x40960xf32>
  %4 = linalg.softmax dimension(2) ins(%2 : tensor<12x128x40960xf32>) outs(%3 : tensor<12x128x40960xf32>) -> tensor<12x128x40960xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : tensor<12x128x40960xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
  return
}

//          CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [1024, 1, 1] subgroup_size = 32
//    CHECK-LABEL: func.func @softmax
//     CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//     CHECK:    gpu.subgroup_reduce  maxnumf {{.*}} cluster(size = 32) : (f32) -> f32
//     CHECK:    gpu.subgroup_reduce  maxnumf {{.*}} cluster(size = 32) : (f32) -> f32
//     CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32

// On CDNA, we prefer wave64 with subgroup size 64.

//          CDNA3: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [1024, 1, 1] subgroup_size = 64
//          CDNA3: func.func @softmax
//     CDNA3-SAME:      translation_info = #[[$TRANSLATION]]
//     CDNA3:    gpu.subgroup_reduce  maxnumf {{.*}} cluster(size = 64) : (f32) -> f32
//     CDNA3:    gpu.subgroup_reduce  maxnumf {{.*}} cluster(size = 16) : (f32) -> f32
//     CDNA3:    gpu.subgroup_reduce  add {{.*}} cluster(size = 16) : (f32) -> f32
