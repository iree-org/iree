transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_fill_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.fill"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }
  pdl.pattern @pdl_matmul_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_fill_target in %arg1
    %foreach_thread, %tiled_fill = tile_to_foreach_thread_op %0 {num_threads = [5, 1], thread_dim_mapping = [1, 0, 2]}

    %1 = pdl_match @pdl_matmul_target in %arg1
    %foreach_thread_2, %tiled_matmul = tile_to_foreach_thread_op %1 {num_threads = [7, 9]}

    transform.iree.bufferize

    // Get the function to which to apply to.
    %2 = pdl_match @pdl_matmul_target in %arg1
    %func = transform.get_closest_isolated_parent %2
    transform.iree.foreach_thread_to_gpu_and_translation_info %func { workgroup_size = [10, 11]}
  }
}
