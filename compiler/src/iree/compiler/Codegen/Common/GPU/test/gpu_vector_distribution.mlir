// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --cse %s | FileCheck %s

#layout = #iree_vector_ext.layout<<[VECTORY, LANEY], [4, 4]>, <[VECTORX, LANEX], [4, 4]>>

builtin.module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: @distribute_elementwise_f16
  func.func @distribute_elementwise_f16(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<4xf16>
    %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout} dense<0.0> : vector<16x16xf16>
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<4xf16>
    // CHECK-DAG: %[[C:.*]] = arith.mulf %[[ROOT]], %[[B]] : vector<4xf16>
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<4xf16>
    // CHECK-DAG: %[[D:.*]] = arith.addf %[[C]], %[[A]] fastmath<reassoc,nnan> : vector<4xf16>
    %d = arith.addf %c, %a fastmath<reassoc,nnan> : vector<16x16xf16>
    // CHECK: iree_vector_ext.to_simd %[[D]] : vector<4xf16> -> vector<16x16xf16>
    return %d : vector<16x16xf16>
  }

  // CHECK-LABEL: @distribute_elementwise_i32
  func.func @distribute_elementwise_i32(%a: vector<16x16xi32>, %b: vector<16x16xi32>) -> vector<16x16xi32> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0 : i32
    // CHECK: %[[ROOT:.*]] = arith.constant dense<0> : vector<4xi32>
    %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout} dense<0> : vector<16x16xi32>
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<4xi32>
    // CHECK-DAG: %[[C:.*]] = arith.muli %[[ROOT]], %[[B]] : vector<4xi32>
    %c = arith.muli %root, %b : vector<16x16xi32>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<4xi32>
    // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] : vector<4xi32>
    %d = arith.addi %c, %a : vector<16x16xi32>
    // CHECK: iree_vector_ext.to_simd %[[D]] : vector<4xi32> -> vector<16x16xi32>
    return %d : vector<16x16xi32>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
