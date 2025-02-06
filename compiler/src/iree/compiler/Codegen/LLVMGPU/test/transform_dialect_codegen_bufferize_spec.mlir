module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %variant_op: !transform.any_op) {
    %tensor_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.eliminate_empty_tensors %tensor_func : (!transform.any_op) -> ()
    %memref_func = transform.iree.bufferize %tensor_func : (!transform.any_op) -> !transform.any_op
    %func_op_bufferized = transform.structured.match ops{["func.func"]} in %memref_func : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_bufferized {
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // Annotate the exported function as already translated.
    %none = transform.param.constant #iree_codegen.translation_info<pipeline = None> -> !transform.any_param
    transform.annotate %func_op_bufferized "translation_info" = %none : !transform.op<"func.func">, !transform.any_param
    transform.yield
  }
} // module
