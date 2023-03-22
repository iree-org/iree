transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  %isolated = transform.get_closest_isolated_parent %warp 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.vector.warp_distribute %isolated
    : (!pdl.operation) -> ()

  // Late canonicalizations to cleanup and pass the checks.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()
}
