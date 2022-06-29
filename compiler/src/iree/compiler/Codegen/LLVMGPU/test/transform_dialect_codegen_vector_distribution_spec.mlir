transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):

  pdl.pattern @pdl_if_op_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "scf.if"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %if_op = pdl_match @pdl_if_op_target in %arg1
    %warp = transform.iree.vector.warp_execute_on_lane_0 %if_op { warp_size = 32 }
    %isolated = transform.get_closest_isolated_parent %warp
    transform.iree.vector.warp_distribute %isolated
  }
}
