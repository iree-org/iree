transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1
    %foreach_thread, %tiled_fill = tile_to_foreach_thread_op %0 {num_threads = [5, 1], thread_dim_mapping = [1, 0, 2]}

    %1 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %foreach_thread_2, %tiled_matmul = tile_to_foreach_thread_op %1 {num_threads = [7, 9]}

    transform.iree.bufferize

    // Get the function to which to apply to.
    %2 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %func = transform.get_closest_isolated_parent %2
    transform.iree.foreach_thread_to_gpu_and_translation_info %func { workgroup_size = [10, 11]}
  }
}
