// RUN: iree-opt -iree-transform-dialect-interpreter --split-input-file %s --verify-diagnostics

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// Propagate the layout from transfer_read to everyone.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = arith.constant 0.0 : vector<16x16xf16>
    %c = arith.mulf %root, %b : vector<16x16xf16>
    %d = arith.addf %c, %a : vector<16x16xf16>
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield 
  }
}
