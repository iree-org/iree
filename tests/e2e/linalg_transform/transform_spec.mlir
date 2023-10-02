
module attributes { transform.with_named_sequence } {

  // Dispatch.
  transform.named_sequence @dispatch(
      %graph: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %graph 
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %foreach_op = transform.structured.tile_using_forall %0 num_threads [13, 33]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %dispatch_op = transform.iree.forall_to_flow %foreach_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  } // @dispatch


  // Codegen.
  transform.named_sequence @codegen(
      %variant_op: !transform.any_op {transform.consumed}) {
    %variant_op_2 = transform.iree.bufferize %variant_op
      : (!transform.any_op) -> !transform.any_op
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  } // codegen

  // Match `func.func`s that are not nested under a `hal.executable.variant` and 
  // only those for codegen.
  transform.named_sequence @match_func_for_dispatch(%root: !transform.any_op {transform.readonly}) 
    -> !transform.any_op {
    transform.match.operation_name %root ["func.func"] : !transform.any_op
    %variant = transform.get_parent_op %root { allow_empty_results, op_name = "hal.executable.variant" } : (!transform.any_op) -> (!transform.any_op)
    transform.match.operation_empty %variant : !transform.any_op
    transform.yield %root : !transform.any_op
  }

  // Find `hal.executable.variant`.
  transform.named_sequence @match_variant_for_codegen(%root: !transform.any_op {transform.readonly}) 
    -> !transform.any_op {
    transform.match.operation_name %root ["hal.executable.variant"] : !transform.any_op
    transform.yield %root : !transform.any_op
  }

  // Transform entry-point
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %root
        @match_variant_for_codegen -> @codegen,
        @match_func_for_dispatch -> @dispatch
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield 
  }
} // module
