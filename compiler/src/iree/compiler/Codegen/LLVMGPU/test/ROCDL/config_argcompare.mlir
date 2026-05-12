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

// Reduction size 2 is smaller than the subgroup size, so setReductionConfig
// rejects this op and it falls through to setArgCompareConfig (TileAndFuse).
// This is the canonical small-reduction case that motivated routing
// arg_compare through TileAndFuse.

func.func @argcompare_small_reduction_f32(
    %input: tensor<2x3x4xf32>,
    %out_val: tensor<3x4xf32>,
    %out_idx: tensor<3x4xi64>) -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input : tensor<2x3x4xf32>)
    outs(%out_val, %out_idx : tensor<3x4xf32>, tensor<3x4xi64>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
// CHECK-SAME:    workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @argcompare_small_reduction_f32
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config<{thread = [0, 1, 1], workgroup = [0, 1, 4]}>

// -----

// Reduction size 100 is not a multiple of any subgroup size choice
// (32 or 64), so setReductionConfig rejects this op even though the
// element type (f32) is supported. It falls through to
// setArgCompareConfig (TileAndFuse).

func.func @argcompare_unaligned_reduction_f32(
    %input: tensor<8x100xf32>,
    %out_val: tensor<8xf32>,
    %out_idx: tensor<8xi32>) -> (tensor<8xf32>, tensor<8xi32>) {
  %0:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input : tensor<8x100xf32>)
    outs(%out_val, %out_idx : tensor<8xf32>, tensor<8xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<8xf32>, tensor<8xi32>
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xi32>
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
// CHECK-SAME:    workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @argcompare_unaligned_reduction_f32
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [8, 0]}>

// -----

// f64 (64-bit) is not supported by vector distribution, falls back to default.

func.func @argcompare_f64_fallback(
    %input: tensor<8x256xf64>,
    %out_val: tensor<8xf64>,
    %out_idx: tensor<8xi32>) -> (tensor<8xf64>, tensor<8xi32>) {
  %0:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input : tensor<8x256xf64>)
    outs(%out_val, %out_idx : tensor<8xf64>, tensor<8xi32>) {
    ^bb0(%a: f64, %b: f64):
      %cmp = arith.cmpf ogt, %a, %b : f64
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<8xf64>, tensor<8xi32>
  return %0#0, %0#1 : tensor<8xf64>, tensor<8xi32>
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
// CHECK-SAME:    workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @argcompare_f64_fallback
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [8, 0]}>

// -----

// Rank-1 input reducing to a scalar: the only iteration loop is the reduction
// dim, so there are no parallel loops to distribute across threads.
// setReductionConfig rejects (reduction size 5 below subgroup), and
// setArgCompareConfig emits a degenerate single-thread TileAndFuse config
// (workgroup_size = [1, 1, 1], tile = [0]) so the lone thread runs the full
// scan once.

func.func @argcompare_1d_rank_to_scalar_f32(
    %input: tensor<5xf32>,
    %out_val: tensor<f32>,
    %out_idx: tensor<i32>) -> (tensor<f32>, tensor<i32>) {
  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input : tensor<5xf32>)
    outs(%out_val, %out_idx : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>
  return %0#0, %0#1 : tensor<f32>, tensor<i32>
}

// CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
// CHECK-SAME:    workgroup_size = [1, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @argcompare_1d_rank_to_scalar_f32
// CHECK:         iree_linalg_ext.arg_compare
// CHECK-SAME:      lowering_config = #iree_gpu.lowering_config<{thread = [0], workgroup = [0]}>
