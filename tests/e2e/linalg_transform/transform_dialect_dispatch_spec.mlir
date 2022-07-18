transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_matmul_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    transform.iree.decide_fusion_roots %arg1
    transform.print %arg1 { name = "AFTER FUSION GROUPS!!!" }

    %0 = pdl_match @pdl_matmul_target in %arg1
    %foreach_op, %tiled_op = tile_to_foreach_thread_op %0 {num_threads = [13, 33]}
    %dispatch_op = transform.iree.foreach_thread_to_flow %foreach_op
  }
}
