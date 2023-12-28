// RUN: iree-opt -iree-transform-dialect-interpreter --split-input-file --cse %s | FileCheck %s

#layout = #iree_vector_ext.layout<<[VECTORY, LANEY], [4, 4]>, <[VECTORX, LANEX], [4, 4]>>

builtin.module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: @distribute_elementwise
  func.func @distribute_elementwise(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<4x4xf16>
    %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout} dense<0.0> : vector<16x16xf16>
    // CHECK: %[[C:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<4x4xf16>
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // CHECK: %[[D:.*]] = arith.addf %{{.*}}, %{{.*}} : vector<4x4xf16>
    %d = arith.addf %c, %a : vector<16x16xf16>
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield 
  }
}
