// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse --mlir-print-local-scope %s | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [4],
  outer_tile    = [1],
  thread_tile   = [4],
  element_tile  = [1],

  subgroup_strides = [0],
  thread_strides   = [16]
>

func.func @step_1() -> vector<16xindex> {
  %step = vector.step : vector<16xindex>
  %stepl = iree_vector_ext.to_layout %step to layout(#nested) : vector<16xindex>
  return %stepl : vector<16xindex>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @step_1
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 4, 8, 12]> : vector<4xindex>
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 16)
// CHECK: %[[TIDB:.+]] = vector.broadcast %[[TID]]#1 : index to vector<4xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[TIDB]], %[[CST]] : vector<4xindex>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [1],
  outer_tile    = [3],
  thread_tile   = [4],
  element_tile  = [2],

  subgroup_strides = [0],
  thread_strides   = [2]
>

func.func @step_2() -> vector<24xindex> {
  %step = vector.step : vector<24xindex>
  %stepl = iree_vector_ext.to_layout %step to layout(#nested) : vector<24xindex>
  return %stepl : vector<24xindex>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @step_2
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 1, 8, 9, 16, 17]> : vector<6xindex>
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 2)
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]]#1, %c2 : index
// CHECK: %[[TID_STRIDEV:.+]] = vector.broadcast %[[TID_STRIDE]] : index to vector<6xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[TID_STRIDEV]], %[[CST]] : vector<6xindex>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [3],
  batch_tile    = [1],
  outer_tile    = [2],
  thread_tile   = [4],
  element_tile  = [2],

  subgroup_strides = [8],
  thread_strides   = [2]
>

func.func @step_3() -> vector<48xindex> {
  %step = vector.step : vector<48xindex>
  %stepl = iree_vector_ext.to_layout %step to layout(#nested) : vector<48xindex>
  return %stepl : vector<48xindex>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @step_3
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 1, 8, 9]> : vector<4xindex>
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[WID:.+]]:4 = affine.delinearize_index %[[IDX]] into (3, 8, 64)
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 2)
// CHECK: %[[WID_STRIDE:.+]] = arith.muli %[[WID]]#1, %c16 : index
// CHECK: %[[WID_STRIDEV:.+]] = vector.broadcast %[[WID_STRIDE]] : index to vector<4xindex>
// CHECK: %[[OFFSET0:.+]] = arith.addi %[[WID_STRIDEV]], %[[CST]] : vector<4xindex>
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]]#1, %c2 : index
// CHECK: %[[TID_STRIDEV:.+]] = vector.broadcast %[[TID_STRIDE]] : index to vector<4xindex>
// CHECK: %[[OFFSET1:.+]] = arith.addi %[[OFFSET0]], %[[TID_STRIDEV]] : vector<4xindex>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [1],
  outer_tile    = [1],
  thread_tile   = [16],
  element_tile  = [8],

  subgroup_strides = [0],
  thread_strides   = [1]
>

func.func @step_4() -> vector<128xindex> {
  %step = vector.step : vector<128xindex>
  %stepl = iree_vector_ext.to_layout %step to layout(#nested) : vector<128xindex>
  return %stepl : vector<128xindex>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @step_4
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[TID:.+]]:2 = affine.delinearize_index %[[IDX]] into (16)
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]]#1, %c8 : index
// CHECK: %[[TID_STRIDEV:.+]] = vector.broadcast %[[TID_STRIDE]] : index to vector<8xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[TID_STRIDEV]], %[[CST]] : vector<8xindex>
