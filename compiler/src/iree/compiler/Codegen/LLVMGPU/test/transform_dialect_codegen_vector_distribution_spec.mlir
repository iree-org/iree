transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %if_op = transform.structured.match ops{["scf.if"]} in %arg1
    %warp = transform.iree.vector.warp_execute_on_lane_0 %if_op { warp_size = 32 }
    %isolated = transform.get_closest_isolated_parent %warp
    transform.iree.vector.warp_distribute %isolated
  }
}
