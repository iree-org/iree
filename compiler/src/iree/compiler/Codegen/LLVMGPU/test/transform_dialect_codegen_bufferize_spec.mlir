module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %variant_op: !transform.any_op) {
    %tensor_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.eliminate_empty_tensors %tensor_func : (!transform.any_op) -> ()
    %memref_func = transform.iree.bufferize %tensor_func : (!transform.any_op) -> !transform.any_op

    // Annotate the exported function as already translated.
    %none = transform.param.constant #iree_codegen.translation_info<None> -> !transform.any_param
    transform.annotate %memref_func "translation_info" = %none : !transform.any_op, !transform.any_param
    transform.yield
  }
} // module
