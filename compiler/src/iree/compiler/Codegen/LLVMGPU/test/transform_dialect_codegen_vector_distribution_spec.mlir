transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op 
    : (!transform.any_op) -> !transform.any_op
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
    : (!transform.any_op) -> !transform.any_op
  %isolated = transform.get_closest_isolated_parent %warp 
    : (!transform.any_op) -> !transform.any_op
  transform.iree.vector.warp_distribute %isolated
    : (!transform.any_op) -> ()

  // Late canonicalizations to cleanup and pass the checks.
  transform.apply_patterns to %variant_op {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
  } : !transform.any_op
  transform.iree.apply_patterns %variant_op
    { canonicalization, licm, cse } : (!transform.any_op) -> ()
}
