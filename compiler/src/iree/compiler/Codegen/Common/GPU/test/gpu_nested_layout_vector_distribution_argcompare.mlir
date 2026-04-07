// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

// Tests for DistributeArgCompare pattern with explicit index mode.

#layout_2d_element_only = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

#layout_1d_element_only = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [1],
  element_tile = [16],

  subgroup_strides = [0],
  thread_strides   = [0]
>

// Test: Element-tile-only argmax with explicit index. No thread shuffles needed
// since thread_tile = [1, 1].
// CHECK-LABEL: func @argmax_element_only
// CHECK-SAME: %[[INPUT:.*]]: vector<16x8xf16>
// CHECK-SAME: %[[INPUT_IDX:.*]]: vector<16x8xi32>
// CHECK-SAME: %[[INIT_VAL:.*]]: vector<16xf16>
// CHECK-SAME: %[[INIT_IDX:.*]]: vector<16xi32>
func.func @argmax_element_only(
    %input: vector<16x8xf16>,
    %input_idx: vector<16x8xi32>,
    %init_val: vector<16xf16>,
    %init_idx: vector<16xi32>) -> (vector<16xf16>, vector<16xi32>) {

  %input_layout = iree_vector_ext.to_layout %input to layout(#layout_2d_element_only) : vector<16x8xf16>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_2d_element_only) : vector<16x8xi32>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_1d_element_only) : vector<16xf16>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_1d_element_only) : vector<16xi32>

  // Local inline reduction within element tile using scalar extract/cmpf/select.
  // No shuffles needed since thread_tile[1] = 1.
  // CHECK-DAG: %[[DIS_INPUT:.*]] = iree_vector_ext.to_simt %[[INPUT]] : vector<16x8xf16> -> vector<1x1x1x1x16x8xf16>
  // CHECK: %[[ELEM0:.*]] = vector.extract %[[DIS_INPUT]][0, 0, 0, 0, 0, 0]
  // CHECK: %[[CMP:.*]] = arith.cmpf ogt, %[[ELEM0]], %{{.*}} : f16
  // CHECK: arith.select %[[CMP]],
  // CHECK-NOT: gpu.shuffle
  %result:2 = iree_vector_ext.arg_compare dimension(1)
      ins(%input_layout, %input_idx_layout : vector<16x8xf16>, vector<16x8xi32>)
      inits(%init_val_layout, %init_idx_layout : vector<16xf16>, vector<16xi32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f16
      iree_vector_ext.yield %cmp : i1
  } -> vector<16xf16>, vector<16xi32>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_1d_element_only) : vector<16xf16>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_1d_element_only) : vector<16xi32>

  func.return %result_val_layout, %result_idx_layout : vector<16xf16>, vector<16xi32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_2d_thread_reduce = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

#layout_1d_thread_reduce = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [2],
  outer_tile = [1],
  thread_tile = [16],
  element_tile = [1],

  subgroup_strides = [1],
  thread_strides = [1]
>

// Test: Ballot-based thread reduction with i64 indices.
// CHECK-LABEL: func @argmax_i64_index
func.func @argmax_i64_index(
    %input: vector<32x32xf32>,
    %input_idx: vector<32x32xi64>,
    %init_val: vector<32xf32>,
    %init_idx: vector<32xi64>) -> (vector<32xf32>, vector<32xi64>) {

  %input_layout = iree_vector_ext.to_layout %input to layout(#layout_2d_thread_reduce) : vector<32x32xf32>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_2d_thread_reduce) : vector<32x32xi64>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_1d_thread_reduce) : vector<32xf32>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_1d_thread_reduce) : vector<32xi64>

  // Local reduction.
  // CHECK: %[[ELEM0:.*]] = vector.extract %{{.*}}[0, 0, 0, 0, 0, 0]
  // CHECK: %[[CMP:.*]] = arith.cmpf ogt, %[[ELEM0]], %{{.*}}
  // CHECK: arith.select %[[CMP]],
  // Ballot-based thread reduction (4 threads on dim 1).
  // CHECK: %[[REDUCED:.*]] = gpu.subgroup_reduce maxnumf %[[LOCAL_VAL:[a-z0-9]+]]
  // CHECK: %[[IS_WINNER:.*]] = arith.cmpf oeq, %[[LOCAL_VAL]], %[[REDUCED]] : f32
  // CHECK: %[[BALLOT:.*]] = gpu.ballot %[[IS_WINNER]]
  // CHECK: %[[MASKED:.*]] = arith.andi %[[BALLOT]],
  // CHECK: %[[WINNER_LANE_I64:.*]] = math.cttz %[[MASKED]]
  // CHECK: %[[WINNER_LANE:.*]] = arith.trunci %[[WINNER_LANE_I64]]
  // CHECK: gpu.shuffle idx {{.*}}, %[[WINNER_LANE]], {{.*}} : i64
  %result:2 = iree_vector_ext.arg_compare dimension(1)
      ins(%input_layout, %input_idx_layout : vector<32x32xf32>, vector<32x32xi64>)
      inits(%init_val_layout, %init_idx_layout : vector<32xf32>, vector<32xi64>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<32xf32>, vector<32xi64>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_1d_thread_reduce) : vector<32xf32>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_1d_thread_reduce) : vector<32xi64>

  func.return %result_val_layout, %result_idx_layout : vector<32xf32>, vector<32xi64>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// Test: Custom comparator falls back to butterfly shuffles.
// CHECK-LABEL: func @argmax_custom_comparator_i64_index
func.func @argmax_custom_comparator_i64_index(
    %input: vector<32x32xf32>,
    %input_idx: vector<32x32xi64>,
    %init_val: vector<32xf32>,
    %init_idx: vector<32xi64>) -> (vector<32xf32>, vector<32xi64>) {

  %input_layout = iree_vector_ext.to_layout %input to layout(#layout_2d_thread_reduce) : vector<32x32xf32>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_2d_thread_reduce) : vector<32x32xi64>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_1d_thread_reduce) : vector<32xf32>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_1d_thread_reduce) : vector<32xi64>

  // CHECK: gpu.shuffle xor {{.*}} : f32
  // CHECK: gpu.shuffle xor {{.*}} : i64
  %result:2 = iree_vector_ext.arg_compare dimension(1)
      ins(%input_layout, %input_idx_layout : vector<32x32xf32>, vector<32x32xi64>)
      inits(%init_val_layout, %init_idx_layout : vector<32xf32>, vector<32xi64>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %lhs2 = arith.mulf %lhs, %lhs : f32
      %rhs2 = arith.mulf %rhs, %rhs : f32
      %cmp = arith.cmpf ogt, %lhs2, %rhs2 : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<32xf32>, vector<32xi64>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_1d_thread_reduce) : vector<32xf32>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_1d_thread_reduce) : vector<32xi64>

  func.return %result_val_layout, %result_idx_layout : vector<32xf32>, vector<32xi64>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// Test: Integer argmax with ballot-based thread reduction (arith.cmpi + maxsi).
// CHECK-LABEL: func @argmax_integer_thread_reduction
func.func @argmax_integer_thread_reduction(
    %input: vector<32x32xi32>,
    %input_idx: vector<32x32xi32>,
    %init_val: vector<32xi32>,
    %init_idx: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {

  %input_layout = iree_vector_ext.to_layout %input to layout(#layout_2d_thread_reduce) : vector<32x32xi32>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_2d_thread_reduce) : vector<32x32xi32>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_1d_thread_reduce) : vector<32xi32>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_1d_thread_reduce) : vector<32xi32>

  // CHECK: %[[ELEM0:.*]] = vector.extract %{{.*}}[0, 0, 0, 0, 0, 0]
  // CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[ELEM0]], %{{.*}}
  // CHECK: arith.select %[[CMP]],
  // CHECK: vector.extract %{{.*}}[0, 1, 0, 0, 0, 0]
  // CHECK: vector.extract %{{.*}}[1, 0, 0, 0, 0, 0]
  // CHECK: %[[REDUCED:.*]] = gpu.subgroup_reduce maxsi %[[LOCAL_VAL:[a-z0-9]+]]
  // CHECK: %[[IS_WINNER:.*]] = arith.cmpi eq, %[[LOCAL_VAL]], %[[REDUCED]] : i32
  // CHECK: %[[BALLOT:.*]] = gpu.ballot %[[IS_WINNER]]
  // CHECK: %[[MASKED:.*]] = arith.andi %[[BALLOT]],
  // CHECK: %[[WINNER_LANE_I64:.*]] = math.cttz %[[MASKED]]
  // CHECK: %[[WINNER_LANE:.*]] = arith.trunci %[[WINNER_LANE_I64]]
  // CHECK: gpu.shuffle idx {{.*}}, %[[WINNER_LANE]],
  %result:2 = iree_vector_ext.arg_compare dimension(1)
      ins(%input_layout, %input_idx_layout : vector<32x32xi32>, vector<32x32xi32>)
      inits(%init_val_layout, %init_idx_layout : vector<32xi32>, vector<32xi32>) {
    ^bb0(%lhs: i32, %rhs: i32):
      %cmp = arith.cmpi sgt, %lhs, %rhs : i32
      iree_vector_ext.yield %cmp : i1
  } -> vector<32xi32>, vector<32xi32>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_1d_thread_reduce) : vector<32xi32>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_1d_thread_reduce) : vector<32xi32>

  func.return %result_val_layout, %result_idx_layout : vector<32xi32>, vector<32xi32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_2d_explicit = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 4],
  element_tile = [1, 8],

  subgroup_strides = [0, 0],
  thread_strides = [4, 1]
>

#layout_1d_explicit = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [4],
  element_tile = [1],

  subgroup_strides = [0],
  thread_strides = [4]
>

// Test: Argmin with explicit index mode using ballot-based thread reduction.
// CHECK-LABEL: func @argmin_explicit_index
// CHECK-SAME: %[[INPUT_VAL:.*]]: vector<4x32xf32>
// CHECK-SAME: %[[INPUT_IDX:.*]]: vector<4x32xi32>
// CHECK-SAME: %[[INIT_VAL:.*]]: vector<4xf32>
// CHECK-SAME: %[[INIT_IDX:.*]]: vector<4xi32>
func.func @argmin_explicit_index(
    %input_val: vector<4x32xf32>,
    %input_idx: vector<4x32xi32>,
    %init_val: vector<4xf32>,
    %init_idx: vector<4xi32>) -> (vector<4xf32>, vector<4xi32>) {

  %input_val_layout = iree_vector_ext.to_layout %input_val to layout(#layout_2d_explicit) : vector<4x32xf32>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_2d_explicit) : vector<4x32xi32>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_1d_explicit) : vector<4xf32>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_1d_explicit) : vector<4xi32>

  // Inline local element reduction with explicit index input.
  // CHECK: %[[ELEM0:.*]] = vector.extract %{{.*}}[0, 0, 0, 0, 0, 0]
  // CHECK: %[[CMP:.*]] = arith.cmpf olt, %[[ELEM0]], %{{.*}}
  // CHECK: arith.select %[[CMP]],
  // Thread reduction with ballot-based approach.
  // CHECK: %[[REDUCED:.*]] = gpu.subgroup_reduce minnumf %[[LOCAL_VAL:[a-z0-9]+]]
  // CHECK: %[[IS_WINNER:.*]] = arith.cmpf oeq, %[[LOCAL_VAL]], %[[REDUCED]] : f32
  // CHECK: %[[BALLOT:.*]] = gpu.ballot %[[IS_WINNER]]
  // CHECK: %[[WINNER_LANE_I64:.*]] = math.cttz %[[BALLOT]]
  // CHECK: %[[WINNER_LANE:.*]] = arith.trunci %[[WINNER_LANE_I64]]
  // CHECK: gpu.shuffle idx {{.*}}, %[[WINNER_LANE]],
  %result:2 = iree_vector_ext.arg_compare dimension(1)
      ins(%input_val_layout, %input_idx_layout : vector<4x32xf32>, vector<4x32xi32>)
      inits(%init_val_layout, %init_idx_layout : vector<4xf32>, vector<4xi32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %cmp = arith.cmpf olt, %lhs, %rhs : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_1d_explicit) : vector<4xf32>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_1d_explicit) : vector<4xi32>

  func.return %result_val_layout, %result_idx_layout : vector<4xf32>, vector<4xi32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_2d_subgroup = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 2],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [8, 4],
  element_tile = [1, 4],

  subgroup_strides = [0, 1],
  thread_strides = [1, 8]
>

#layout_1d_subgroup = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [8],
  element_tile = [1],

  subgroup_strides = [0],
  thread_strides = [1]
>

// Test: Subgroup reduction via shared memory with explicit index.
// 2 subgroups participate in reduction on dim 1.
// CHECK-LABEL: func @argmax_subgroup_reduction
// CHECK-SAME: %[[INPUT:.*]]: vector<8x32xf32>
// CHECK-SAME: %[[INPUT_IDX:.*]]: vector<8x32xi32>
// CHECK-SAME: %[[INIT_VAL:.*]]: vector<8xf32>
// CHECK-SAME: %[[INIT_IDX:.*]]: vector<8xi32>
func.func @argmax_subgroup_reduction(
    %input: vector<8x32xf32>,
    %input_idx: vector<8x32xi32>,
    %init_val: vector<8xf32>,
    %init_idx: vector<8xi32>) -> (vector<8xf32>, vector<8xi32>) {

  %input_layout = iree_vector_ext.to_layout %input to layout(#layout_2d_subgroup) : vector<8x32xf32>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_2d_subgroup) : vector<8x32xi32>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_1d_subgroup) : vector<8xf32>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_1d_subgroup) : vector<8xi32>

  // Inline local element reduction + thread reduction (ballot-based) + subgroup reduction via shared memory.
  // CHECK: %[[ELEM0:.*]] = vector.extract %{{.*}}[0, 0, 0, 0, 0, 0]
  // CHECK: %[[CMP:.*]] = arith.cmpf ogt, %[[ELEM0]], %{{.*}}
  // CHECK: arith.select %[[CMP]],
  // Thread reduction with ballot-based approach (4 threads on dim 1).
  // CHECK: %[[REDUCED:.*]] = gpu.subgroup_reduce maxnumf %[[LOCAL_VAL:[a-z0-9]+]]
  // CHECK: %[[IS_WINNER:.*]] = arith.cmpf oeq, %[[LOCAL_VAL]], %[[REDUCED]] : f32
  // CHECK: %[[BALLOT:.*]] = gpu.ballot %[[IS_WINNER]]
  // CHECK: %[[MASKED:.*]] = arith.andi %[[BALLOT]],
  // CHECK: %[[WINNER_LANE_I64:.*]] = math.cttz %[[MASKED]]
  // CHECK: %[[WINNER_LANE:.*]] = arith.trunci %[[WINNER_LANE_I64]]
  // CHECK: gpu.shuffle idx {{.*}}, %[[WINNER_LANE]],
  // Shared memory operations for subgroup reduction.
  // CHECK: %[[VAL_ALLOC:.*]] = memref.alloc
  // CHECK: %[[IDX_ALLOC:.*]] = memref.alloc
  // CHECK: gpu.barrier
  // CHECK: vector.transfer_write {{.*}}, %[[VAL_ALLOC]]
  // CHECK: vector.transfer_write {{.*}}, %[[IDX_ALLOC]]
  // CHECK: gpu.barrier
  // CHECK: vector.transfer_read %[[VAL_ALLOC]]
  // CHECK: vector.transfer_read %[[IDX_ALLOC]]
  // Second reduction across subgroups uses ballot-based approach.
  // CHECK: gpu.subgroup_reduce maxnumf
  // CHECK: gpu.shuffle idx
  %result:2 = iree_vector_ext.arg_compare dimension(1)
      ins(%input_layout, %input_idx_layout : vector<8x32xf32>, vector<8x32xi32>)
      inits(%init_val_layout, %init_idx_layout : vector<8xf32>, vector<8xi32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<8xf32>, vector<8xi32>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_1d_subgroup) : vector<8xf32>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_1d_subgroup) : vector<8xi32>

  func.return %result_val_layout, %result_idx_layout : vector<8xf32>, vector<8xi32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// Test: Full reduction to 0-d (scalar) result with thread reduction.
// This tests the special handling for 0-d vectors where:
// 1. Init values are unwrapped from ToSIMD (identity op for 0-d)
// 2. reshapeFlatToTarget uses extract+broadcast instead of shape_cast

#layout_1d_full_reduce = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [4],
  element_tile = [8],

  subgroup_strides = [0],
  thread_strides = [1]
>

#layout_0d = #iree_vector_ext.nested_layout<
  subgroup_tile = [],
  batch_tile = [],
  outer_tile = [],
  thread_tile = [],
  element_tile = [],

  subgroup_strides = [],
  thread_strides = []
>

// CHECK-LABEL: func @argmax_full_reduce_to_scalar
// CHECK-SAME: %[[INPUT:.*]]: vector<32xf32>
// CHECK-SAME: %[[INPUT_IDX:.*]]: vector<32xi32>
// CHECK-SAME: %[[INIT_VAL:.*]]: vector<f32>
// CHECK-SAME: %[[INIT_IDX:.*]]: vector<i32>
func.func @argmax_full_reduce_to_scalar(
    %input: vector<32xf32>,
    %input_idx: vector<32xi32>,
    %init_val: vector<f32>,
    %init_idx: vector<i32>) -> (vector<f32>, vector<i32>) {

  %input_layout = iree_vector_ext.to_layout %input to layout(#layout_1d_full_reduce) : vector<32xf32>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_1d_full_reduce) : vector<32xi32>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_0d) : vector<f32>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_0d) : vector<i32>

  // Full reduction to scalar. Init values are 0-d vectors (scalars).
  // The pattern should:
  // 1. Unwrap ToSIMD for 0-d init values (identity op)
  // 2. Use extract+broadcast instead of shape_cast for 0-d reshape
  // Distributed shape: vector<1x1x8xf32> (batch=1, outer=1, element=8)
  // CHECK: %[[ELEM0:.*]] = vector.extract %{{.*}}[0, 0, 0] : f32 from vector<1x1x8xf32>
  // CHECK: %[[CMP:.*]] = arith.cmpf ogt, %[[ELEM0]], %{{.*}}
  // CHECK: arith.select %[[CMP]],
  // Thread reduction with ballot-based approach (4 threads).
  // CHECK: %[[REDUCED:.*]] = gpu.subgroup_reduce maxnumf %[[LOCAL_VAL:[a-z0-9]+]]
  // CHECK: %[[IS_WINNER:.*]] = arith.cmpf oeq, %[[LOCAL_VAL]], %[[REDUCED]] : f32
  // CHECK: %[[BALLOT:.*]] = gpu.ballot %[[IS_WINNER]]
  // CHECK: %[[WINNER_LANE_I64:.*]] = math.cttz %[[BALLOT]]
  // CHECK: %[[WINNER_LANE:.*]] = arith.trunci %[[WINNER_LANE_I64]]
  // CHECK: gpu.shuffle idx {{.*}}, %[[WINNER_LANE]],
  // For 0-d result, use extract+broadcast instead of shape_cast.
  // CHECK: vector.broadcast {{.*}} : f32 to vector<f32>
  %result:2 = iree_vector_ext.arg_compare dimension(0)
      ins(%input_layout, %input_idx_layout : vector<32xf32>, vector<32xi32>)
      inits(%init_val_layout, %init_idx_layout : vector<f32>, vector<i32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<f32>, vector<i32>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_0d) : vector<f32>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_0d) : vector<i32>

  func.return %result_val_layout, %result_idx_layout : vector<f32>, vector<i32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// Test: Full reduction to 0-d with subgroup reduction.
// Exercises the combination of 0-d result + subgroup_tile > 1, which requires
// shared memory exchange between subgroups before producing a scalar output.

#layout_1d_full_reduce_subgroup = #iree_vector_ext.nested_layout<
  subgroup_tile = [2],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [4],
  element_tile = [8],

  subgroup_strides = [1],
  thread_strides = [1]
>

#layout_0d_subgroup = #iree_vector_ext.nested_layout<
  subgroup_tile = [],
  batch_tile = [],
  outer_tile = [],
  thread_tile = [],
  element_tile = [],

  subgroup_strides = [],
  thread_strides = []
>

// CHECK-LABEL: func @argmax_full_reduce_with_subgroup
// CHECK-SAME: %[[INPUT:.*]]: vector<64xf16>
// CHECK-SAME: %[[INPUT_IDX:.*]]: vector<64xi32>
// CHECK-SAME: %[[INIT_VAL:.*]]: vector<f16>
// CHECK-SAME: %[[INIT_IDX:.*]]: vector<i32>
func.func @argmax_full_reduce_with_subgroup(
    %input: vector<64xf16>,
    %input_idx: vector<64xi32>,
    %init_val: vector<f16>,
    %init_idx: vector<i32>) -> (vector<f16>, vector<i32>) {

  %input_layout = iree_vector_ext.to_layout %input to layout(#layout_1d_full_reduce_subgroup) : vector<64xf16>
  %input_idx_layout = iree_vector_ext.to_layout %input_idx to layout(#layout_1d_full_reduce_subgroup) : vector<64xi32>
  %init_val_layout = iree_vector_ext.to_layout %init_val to layout(#layout_0d_subgroup) : vector<f16>
  %init_idx_layout = iree_vector_ext.to_layout %init_idx to layout(#layout_0d_subgroup) : vector<i32>

  // CHECK: %[[ELEM0:.*]] = vector.extract %{{.*}}[0, 0, 0] : f16 from vector<1x1x8xf16>
  // CHECK: %[[CMP:.*]] = arith.cmpf ogt, %[[ELEM0]], %{{.*}}
  // CHECK: arith.select %[[CMP]],
  // CHECK: %[[REDUCED:.*]] = gpu.subgroup_reduce maxnumf %[[LOCAL_VAL:[a-z0-9]+]]
  // CHECK: %[[IS_WINNER:.*]] = arith.cmpf oeq, %[[LOCAL_VAL]], %[[REDUCED]] : f16
  // CHECK: %[[BALLOT:.*]] = gpu.ballot %[[IS_WINNER]]
  // CHECK: gpu.shuffle idx {{.*}} : i32
  // CHECK: %[[VAL_ALLOC:.*]] = memref.alloc
  // CHECK: %[[IDX_ALLOC:.*]] = memref.alloc
  // CHECK: gpu.barrier
  // CHECK: vector.transfer_write {{.*}}, %[[VAL_ALLOC]]
  // CHECK: vector.transfer_write {{.*}}, %[[IDX_ALLOC]]
  // CHECK: gpu.barrier
  // CHECK: vector.transfer_read %[[VAL_ALLOC]]
  // CHECK: vector.transfer_read %[[IDX_ALLOC]]
  // CHECK: gpu.subgroup_reduce maxnumf
  // CHECK: gpu.shuffle idx
  // CHECK: vector.broadcast {{.*}} : f16 to vector<f16>
  %result:2 = iree_vector_ext.arg_compare dimension(0)
      ins(%input_layout, %input_idx_layout : vector<64xf16>, vector<64xi32>)
      inits(%init_val_layout, %init_idx_layout : vector<f16>, vector<i32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f16
      iree_vector_ext.yield %cmp : i1
  } -> vector<f16>, vector<i32>

  %result_val_layout = iree_vector_ext.to_layout %result#0 to layout(#layout_0d_subgroup) : vector<f16>
  %result_idx_layout = iree_vector_ext.to_layout %result#1 to layout(#layout_0d_subgroup) : vector<i32>

  func.return %result_val_layout, %result_idx_layout : vector<f16>, vector<i32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
