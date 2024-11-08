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
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK: %[[TID:.+]] = affine.apply affine_map<()[s0] -> ((s0 floordiv 16) mod 4)>()[%thread_id_x]
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]], %c16 : index
// CHECK: %[[TID_STRIDEV:.+]] = vector.broadcast %[[TID_STRIDE]] : index to vector<4xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[TID_STRIDEV]], %[[CST]] : vector<4xindex>

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
// CHECK: %[[TID:.+]] = affine.apply affine_map<()[s0] -> ((s0 floordiv 2) mod 4)>()[%thread_id_x]
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]], %c2 : index
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
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 1, 24, 25]> : vector<4xindex>
// CHECK: %[[WID:.+]] = affine.apply affine_map<()[s0] -> ((s0 floordiv 512) mod 3)>()[%thread_id_x]
// CHECK: %[[TID:.+]] = affine.apply affine_map<()[s0] -> ((s0 floordiv 2) mod 4)>()[%thread_id_x]
// CHECK: %[[WID_STRIDE:.+]] = arith.muli %[[WID]], %c8 : index
// CHECK: %[[WID_STRIDEV:.+]] = vector.broadcast %[[WID_STRIDE]] : index to vector<4xindex>
// CHECK: %[[OFFSET0:.+]] = arith.addi %[[WID_STRIDEV]], %[[CST]] : vector<4xindex>
// CHECK: %[[TID_STRIDE:.+]] = arith.muli %[[TID]], %c2 : index
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
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112]> : vector<8xindex>
// CHECK: %[[TID:.+]] = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%thread_id_x]
// CHECK: %[[TIDV:.+]] = vector.broadcast %[[TID]] : index to vector<8xindex>
// CHECK: %[[OFFSET:.+]] = arith.addi %[[TIDV]], %[[CST]] : vector<8xindex>
