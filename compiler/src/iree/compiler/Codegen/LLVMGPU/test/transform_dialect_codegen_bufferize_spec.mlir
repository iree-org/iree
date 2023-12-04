module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %variant_op: !transform.any_op {transform.consumed}) {
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
} // module
