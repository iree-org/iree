transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %if_op = transform.structured.match ops{["scf.if"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  %isolated = transform.get_closest_isolated_parent %warp : (!pdl.operation) -> !pdl.operation
  transform.iree.vector.warp_distribute %isolated
}
