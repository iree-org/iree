// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --cse %s | FileCheck %s

#layout_1d = #iree_vector_ext.layout<<[VECTORX, LANEX], [4, 4]>>
#layout_2d = #iree_vector_ext.layout<<[VECTORY, LANEY], [4, 4]>, <[VECTORX, LANEX], [4, 4]>>

builtin.module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: @distribute_elementwise_f16
  func.func @distribute_elementwise_f16(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<4xf16>
    %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout_2d} dense<0.0> : vector<16x16xf16>
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<4xf16>
    // CHECK-DAG: %[[C:.*]] = arith.mulf %[[ROOT]], %[[B]] : vector<4xf16>
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<4xf16>
    // CHECK-DAG: %[[D:.*]] = arith.addf %[[C]], %[[A]] : vector<4xf16>
    %d = arith.addf %c, %a : vector<16x16xf16>
    // CHECK: iree_vector_ext.to_simd %[[D]] : vector<4xf16> -> vector<16x16xf16>
    return %d : vector<16x16xf16>
  }

  // CHECK-LABEL: @distribute_elementwise_i32
  func.func @distribute_elementwise_i32(%a: vector<16x16xi32>, %b: vector<16x16xi32>) -> vector<16x16xi32> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0 : i32
    // CHECK: %[[ROOT:.*]] = arith.constant dense<0> : vector<4xi32>
    %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout_2d} dense<0> : vector<16x16xi32>
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<4xi32>
    // CHECK-DAG: %[[C:.*]] = arith.muli %[[ROOT]], %[[B]] : vector<4xi32>
    %c = arith.muli %root, %b : vector<16x16xi32>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<4xi32>
    // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] : vector<4xi32>
    %d = arith.addi %c, %a : vector<16x16xi32>
    // CHECK: iree_vector_ext.to_simd %[[D]] : vector<4xi32> -> vector<16x16xi32>
    return %d : vector<16x16xi32>
  }

  // CHECK-LABEL: @distribute_reduction_f32
  // CHECK-SAME:    (%[[ARG0:.+]]: vector<16xf32>, %[[ACC:.+]]: f32) -> f32
  func.func @distribute_reduction_f32(%a: vector<16xf32>, %c: f32) -> f32 {
    // CHECK:     %[[D0:.+]] = iree_vector_ext.to_simt %[[ARG0]] : vector<16xf32> -> vector<4xf32>
    // CHECK-DAG: %[[R0:.+]] = vector.reduction <add>, %[[D0]] : vector<4xf32> into f32
    // CHECK-DAG: %[[R1:.+]] = vector.reduction <mul>, %[[D0]], %[[ACC]] fastmath<reassoc,nnan> : vector<4xf32> into f32
    %r0 = vector.reduction <add>, %a {"__vector_layout_test_anchor_operand_0" = #layout_1d} : vector<16xf32> into f32
    %r1 = vector.reduction <mul>, %a, %c fastmath<reassoc,nnan> : vector<16xf32> into f32
    // CHECK:     %[[R3:.+]] = arith.addf %[[R0]], %[[R1]] : f32
    %r = arith.addf %r0, %r1 : f32
    // CHECK:     return %[[R3]] : f32
    return %r : f32
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
