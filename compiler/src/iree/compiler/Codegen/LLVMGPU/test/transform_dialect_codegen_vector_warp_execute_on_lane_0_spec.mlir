transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op 
    : (!transform.any_op) -> !transform.any_op
  transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
    : (!transform.any_op) -> !transform.any_op

  // Late canonicalizations to cleanup and pass the checks.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()
}
