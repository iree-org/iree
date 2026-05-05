// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_1d() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
  %4 = tensor.empty() : tensor<f32>
  %5 = tensor.empty() : tensor<i32>
  %6:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%3 : tensor<4096xf32>)
    outs(%4, %5 : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
  return
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-LABEL: func.func @argcompare_1d
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:        partial_reduction
// CHECK-SAME:        workgroup

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_2d_reduce_last() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xf32>> -> tensor<32x4096xf32>
  %4 = tensor.empty() : tensor<32xf32>
  %5 = tensor.empty() : tensor<32xi32>
  %6:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%3 : tensor<32x4096xf32>)
    outs(%4, %5 : tensor<32xf32>, tensor<32xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<32xf32>, tensor<32xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [32], strides = [1] : tensor<32xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xi32>>
  return
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-LABEL: func.func @argcompare_2d_reduce_last
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:        partial_reduction = [0,
// CHECK-SAME:        workgroup = [1, 0]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_f16() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2048xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2048xf16>> -> tensor<8x2048xf16>
  %4 = tensor.empty() : tensor<8xf16>
  %5 = tensor.empty() : tensor<8xi32>
  %6:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%3 : tensor<8x2048xf16>)
    outs(%4, %5 : tensor<8xf16>, tensor<8xi32>) {
    ^bb0(%a: f16, %b: f16):
      %cmp = arith.cmpf ogt, %a, %b : f16
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<8xf16>, tensor<8xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf16>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [8], strides = [1] : tensor<8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  return
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-LABEL: func.func @argcompare_f16
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:        partial_reduction
// CHECK-SAME:        thread

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_non_divisible_reduction() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x1024xf32>> -> tensor<16x1024xf32>
  %4 = tensor.empty() : tensor<16xf32>
  %5 = tensor.empty() : tensor<16xi32>
  %6:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%3 : tensor<16x1024xf32>)
    outs(%4, %5 : tensor<16xf32>, tensor<16xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<16xf32>, tensor<16xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [16], strides = [1] : tensor<16xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  return
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-LABEL: func.func @argcompare_non_divisible_reduction
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_i32() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x512xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x512xi32>> -> tensor<16x512xi32>
  %4 = tensor.empty() : tensor<16xi32>
  %5 = tensor.empty() : tensor<16xi32>
  %6:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%3 : tensor<16x512xi32>)
    outs(%4, %5 : tensor<16xi32>, tensor<16xi32>) {
    ^bb0(%a: i32, %b: i32):
      %cmp = arith.cmpi sgt, %a, %b : i32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<16xi32>, tensor<16xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [16], strides = [1] : tensor<16xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [16], strides = [1] : tensor<16xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  return
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-LABEL: func.func @argcompare_i32
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_argmin() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1024xf32>> -> tensor<32x1024xf32>
  %4 = tensor.empty() : tensor<32xf32>
  %5 = tensor.empty() : tensor<32xi32>
  %6:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%3 : tensor<32x1024xf32>)
    outs(%4, %5 : tensor<32xf32>, tensor<32xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf olt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<32xf32>, tensor<32xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [32], strides = [1] : tensor<32xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xi32>>
  return
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-LABEL: func.func @argcompare_argmin
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:        partial_reduction
// CHECK-SAME:        workgroup

// -----

// f64 (64-bit) is not supported by vector distribution, falls back to default.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_f64_fallback() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x256xf64>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf64>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x256xf64>> -> tensor<8x256xf64>
  %4 = tensor.empty() : tensor<8xf64>
  %5 = tensor.empty() : tensor<8xi32>
  %6:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%3 : tensor<8x256xf64>)
    outs(%4, %5 : tensor<8xf64>, tensor<8xi32>) {
    ^bb0(%a: f64, %b: f64):
      %cmp = arith.cmpf ogt, %a, %b : f64
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<8xf64>, tensor<8xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf64>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [8], strides = [1] : tensor<8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  return
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<Distribute>
// CHECK-LABEL: func.func @argcompare_f64_fallback
