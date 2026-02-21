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
// CHECK-DAG: %[[CST:.+]] = arith.constant dense<{{\[\[}}0], [4], [8], [12]]> : vector<4x1xindex>
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 16)
// CHECK: %[[STEP:.+]] = vector.step : vector<1xindex>
// CHECK: %[[STEP_BCAST:.+]] = vector.broadcast %[[STEP]] : vector<1xindex> to vector<4x1xindex>
// CHECK: %[[BASE:.+]] = arith.addi %[[STEP_BCAST]], %[[CST]] : vector<4x1xindex>
// CHECK: %[[TIDB:.+]] = vector.broadcast %[[TID]]#1 : index to vector<4x1xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[BASE]], %[[TIDB]] : vector<4x1xindex>

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
// CHECK-DAG: %[[CST:.+]] = arith.constant dense<{{\[\[}}0], [8], [16]]> : vector<3x1xindex>
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 2)
// CHECK: %[[STEP:.+]] = vector.step : vector<2xindex>
// CHECK: %[[STEP_BCAST:.+]] = vector.broadcast %[[STEP]] : vector<2xindex> to vector<3x2xindex>
// CHECK: %[[OUTER_BCAST:.+]] = vector.broadcast %[[CST]] : vector<3x1xindex> to vector<3x2xindex>
// CHECK: %[[BASE:.+]] = arith.addi %[[OUTER_BCAST]], %[[STEP_BCAST]] : vector<3x2xindex>
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]]#1, %[[C2]] : index
// CHECK: %[[TID_STRIDEV:.+]] = vector.broadcast %[[TID_STRIDE]] : index to vector<3x2xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[BASE]], %[[TID_STRIDEV]] : vector<3x2xindex>

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
// CHECK-DAG: %[[CST:.+]] = arith.constant dense<{{\[\[}}0], [8]]> : vector<2x1xindex>
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[WID:.+]]:4 = affine.delinearize_index %[[IDX]] into (3, 8, 64)
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 2)
// CHECK: %[[STEP:.+]] = vector.step : vector<2xindex>
// CHECK: %[[STEP_BCAST:.+]] = vector.broadcast %[[STEP]] : vector<2xindex> to vector<2x2xindex>
// CHECK: %[[OUTER_BCAST:.+]] = vector.broadcast %[[CST]] : vector<2x1xindex> to vector<2x2xindex>
// CHECK: %[[BASE:.+]] = arith.addi %[[OUTER_BCAST]], %[[STEP_BCAST]] : vector<2x2xindex>
// CHECK: %[[WID_STRIDE:.+]] = arith.muli %[[WID]]#1, %[[C16]] : index
// CHECK: %[[WID_STRIDEV:.+]] = vector.broadcast %[[WID_STRIDE]] : index to vector<2x2xindex>
// CHECK: %[[OFFSET0:.+]] = arith.addi %[[BASE]], %[[WID_STRIDEV]] : vector<2x2xindex>
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]]#1, %[[C2]] : index
// CHECK: %[[TID_STRIDEV:.+]] = vector.broadcast %[[TID_STRIDE]] : index to vector<2x2xindex>
// CHECK: %[[OFFSET1:.+]] = arith.addi %[[OFFSET0]], %[[TID_STRIDEV]] : vector<2x2xindex>

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
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK: %[[IDX:.+]] = gpu.thread_id x
// CHECK: %[[TID:.+]]:2 = affine.delinearize_index %[[IDX]] into (16)
// CHECK: %[[STEP:.+]] = vector.step : vector<8xindex>
// CHECK: %[[STEP_2D:.+]] = vector.shape_cast %[[STEP]] : vector<8xindex> to vector<1x8xindex>
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]]#1, %[[C8]] : index
// CHECK: %[[TID_STRIDEV:.+]] = vector.broadcast %[[TID_STRIDE]] : index to vector<1x8xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[STEP_2D]], %[[TID_STRIDEV]] : vector<1x8xindex>
