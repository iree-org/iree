// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
func.func @pad_conv_2d_nchw_fchw_1x320x64x64x320x3x3() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320x64x64xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x320x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x320x64x64xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 320, 64, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320x64x64xf32>> -> tensor<1x320x64x64xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [320, 320, 3, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x320x3x3xf32>> -> tensor<320x320x3x3xf32>
  %5 = tensor.empty() : tensor<1x320x64x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
  %padded = tensor.pad %3 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x320x64x64xf32> to tensor<1x320x66x66xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %4 : tensor<1x320x66x66xf32>, tensor<320x320x3x3xf32>) outs(%6 : tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 320, 64, 64], strides = [1, 1, 1, 1] : tensor<1x320x64x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x320x64x64xf32>>
  return
}

// CHECK-LABEL:  func.func @pad_conv_2d_nchw_fchw_1x320x64x64x320x3x3
//
// Check that the stack buffer is bounded by tiling sizes.
//
// CHECK:          memref.alloca() {alignment = 64 : i64} : memref<1x8x1x8xf32>
// CHECK:          vector.fma
