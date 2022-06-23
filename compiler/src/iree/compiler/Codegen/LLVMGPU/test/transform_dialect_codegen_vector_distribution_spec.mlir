// RUN: iree-opt

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):

  pdl.pattern @pdl_if_op_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "scf.if"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  // Step 3: tile to scf.for reduction by 1x32, vectorize 1x32, 
  // distribute vector along 32 on threadidx.x.
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %if_op = pdl_match @pdl_if_op_target in %arg1
    %isolated = transform.get_closest_isolated_parent %if_op
    transform.iree.vector_distribution %isolated { warp_size = 32 }
  }
}
