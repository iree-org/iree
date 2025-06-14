// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
func.func @pad_conv_2d_nchw_fchw_1x320x64x64x320x3x3() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c5243520 = arith.constant 5243520 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
  %5 = arith.index_castui %0 {stream.alignment = 128 : index, stream.values = [10486400 : index, 15729280 : index]} : i32 to index
  %6 = arith.index_castui %1 {stream.alignment = 256 : index, stream.values = [1273222400 : index, 1280618240 : index]} : i32 to index
  %7 = arith.index_castui %2 {stream.alignment = 256 : index, stream.values = [10507520 : index, 21488640 : index]} : i32 to index
  %8 = arith.index_castui %3 {stream.alignment = 256 : index, stream.values = [10508800 : index, 21489920 : index]} : i32 to index
  %9 = arith.index_castui %4 {stream.alignment = 128 : index, stream.values = [10486400 : index, 10487680 : index]} : i32 to index
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c5243520) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320x64x64xf32>>
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%6) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x320x3x3xf32>>
  %12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%7) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>>
  %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%8) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>>
  %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%5) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>>
  %15 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%9) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x320x64x64xf32>>
  %16 = iree_tensor_ext.dispatch.tensor.load %10, offsets = [0, 0, 0, 0], sizes = [1, 320, 64, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320x64x64xf32>> -> tensor<1x320x64x64xf32>
  %17 = iree_tensor_ext.dispatch.tensor.load %11, offsets = [0, 0, 0, 0], sizes = [320, 320, 3, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x320x3x3xf32>> -> tensor<320x320x3x3xf32>
  %18 = iree_tensor_ext.dispatch.tensor.load %12, offsets = [0, 0], sizes = [1, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>> -> tensor<1x320xf32>
  %19 = iree_tensor_ext.dispatch.tensor.load %13, offsets = [0, 0], sizes = [1, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>> -> tensor<1x320xf32>
  %20 = iree_tensor_ext.dispatch.tensor.load %14, offsets = [0, 0], sizes = [1, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>> -> tensor<1x320xf32>
  %21 = tensor.empty() : tensor<1x320x64x64xf32>
  %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
  %padded = tensor.pad %16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x320x64x64xf32> to tensor<1x320x66x66xf32>
  %23 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %17 : tensor<1x320x66x66xf32>, tensor<320x320x3x3xf32>) outs(%22 : tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
  iree_tensor_ext.dispatch.tensor.store %23, %15, offsets = [0, 0, 0, 0], sizes = [1, 320, 64, 64], strides = [1, 1, 1, 1] : tensor<1x320x64x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x320x64x64xf32>>
  return
}

// CHECK-LABEL:  func.func @pad_conv_2d_nchw_fchw_1x320x64x64x320x3x3
//
// Check that the stack buffer is bounded by tiling sizes.
//
// CHECK:          memref.alloca() {alignment = 64 : i64} : memref<1x8x1x8xf32>
// CHECK:          vector.fma
