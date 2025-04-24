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

//          CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [1024, 1, 1] subgroup_size = 32>
//    CHECK-LABEL: func.func @softmax
//     CHECK-SAME:     translation_info = #[[$TRANSLATION]]
// CHECK-COUNT-20:   gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}}

// -----

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

// On CDNA, we prefer wave64 with subgroup size 64.

//          CDNA3: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [1024, 1, 1] subgroup_size = 64>
//          CDNA3: func.func @softmax
//     CDNA3-SAME:      translation_info = #[[$TRANSLATION]]
// CDNA3-COUNT-20:   gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_softmax() {
  %c32_i64 = arith.constant 32 : i64
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.extui %0 : i32 to i64
  %3 = arith.extui %1 : i32 to i64
  %4 = arith.shli %3, %c32_i64 : i64
  %5 = arith.ori %2, %4 : i64
  %6 = arith.index_castui %5 : i64 to index
  %7 = iree_tensor_ext.dispatch.workload.ordinal %6, 0 : index
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xf16>>{%7}
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x?xf16>>{%7}
  %10 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0], sizes = [32, %7], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xf16>>{%7} -> tensor<32x?xf16>
  %11 = tensor.empty(%7) : tensor<32x?xf16>
  %12 = linalg.softmax dimension(1) ins(%10 : tensor<32x?xf16>) outs(%11 : tensor<32x?xf16>) -> tensor<32x?xf16>
  iree_tensor_ext.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [32, %7], strides = [1, 1] : tensor<32x?xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x?xf16>>{%7}
  return
}


// Finer details of this lowering are captured by the spirv pipeline test. Just
// verify that warp reduction triggers.
//    CHECK-LABEL: func.func @dynamic_softmax
// CHECK-COUNT-10: gpu.shuffle  xor {{.*}} : i32
