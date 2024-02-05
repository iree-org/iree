module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {

    %if_op = transform.structured.match ops{["scf.if"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
      : (!transform.any_op) -> !transform.any_op

    // Late canonicalizations to cleanup and pass the checks.
    %func_op = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func_op : !transform.any_op
    transform.apply_cse to %func_op : !transform.any_op
    transform.yield
  }
} // module
