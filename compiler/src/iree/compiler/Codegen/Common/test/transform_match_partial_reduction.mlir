// RUN: iree-opt %s --iree-transform-dialect-interpreter --verify-diagnostics --split-input-file

// This can be matched by "reduction_partial" but not by "reduction".

func.func @reduction_with_extra_op_in_func(%arg0: tensor<8x479xf32>, %arg1: tensor<32x32xf32>) -> (tensor<8xf32>, tensor<32xf32>) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  // expected-remark @below {{fill}}
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  // expected-remark @below {{reduction}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg0 : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>

  %empty2 = tensor.empty() : tensor<32xf32>
  %fill2 = linalg.fill ins(%cst : f32) outs(%empty2 : tensor<32xf32>) -> tensor<32xf32>
  return %result, %fill2 : tensor<8xf32>, tensor<32xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.iree.register_match_callbacks

  %leading, %fill, %reduction, %trailing =
    transform.iree.match_callback failures(propagate) "reduction_partial"(%arg0)
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  transform.iree.emit_remark "leading" at %leading : !transform.any_op
  transform.iree.emit_remark "fill" at %fill : !transform.any_op
  transform.iree.emit_remark "reduction" at %reduction : !transform.any_op
  transform.iree.emit_remark "trailing" at %trailing : !transform.any_op

  // expected-error @below {{failed to match}}
  transform.iree.match_callback failures(propagate) "reduction"(%arg0)
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}
