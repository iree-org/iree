// RUN: iree-opt -iree-transform-dialect-interpreter --split-input-file %s --verify-diagnostics

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// Propagate the layout from transfer_read to everyone.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// Enforce the layout from the transfer_write to everyone
builtin.module attributes { transform.with_named_sequence } {
  func.func @enforce_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0 = arith.constant dense<0.0> : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %cst0, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    vector.transfer_write %d, %arr[%c0, %c0] {in_bounds = [true, true], "__vector_layout_test_anchor_operand_0" = #layout} : vector<16x16xf16>, memref<16x16xf16>
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// First propagate the layout, and then enforce it up.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_and_enforce(%arr: memref<16x16xf16>, %arr2: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %root2 = vector.transfer_read %arr2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %e = arith.divf %d, %root2 : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    func.return %e : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %root_red = vector.multi_reduction<add>, %root, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through transpose and then reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @transpose_and_reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %root_transpose = vector.transpose %root, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>, <[ VECTORY], [16]>>}}
    %root_red = vector.multi_reduction<add>, %root_transpose, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}
