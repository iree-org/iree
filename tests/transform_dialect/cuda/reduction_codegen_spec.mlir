// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op
  %fused_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
  // Note: split by 32 to vector-distribute the tail combiner_op, but
  // split by 2 to vector-distribute the meaty %more_parallel_op
  %init_or_alloc_op, %fill_op, %more_parallel_op, %combiner_op =
    transform.structured.split_reduction %0
      { split_factor = 2, insert_split_dimension = 1, use_alloc }

  %1 = transform.structured.match ops{["linalg.generic"]} in %variant_op
  %foreach_thread_1, %tiled_fill =
    transform.structured.tile_to_foreach_thread_op %fill_op num_threads [4, 2] (mapped to dims [2, 1, 0])
  %foreach_thread_2, %tiled_more_parallel_op =
      transform.structured.tile_to_foreach_thread_op %more_parallel_op num_threads [4, 2] (mapped to dims [2, 1, 0])
  %foreach_thread_3, %tiled_combiner_op =
    transform.structured.tile_to_foreach_thread_op %combiner_op num_threads [4] (mapped to dims [2, 1, 0])
  %foreach_thread_4, %tiled_fused_fill_op =
    transform.structured.tile_to_foreach_thread_op %fused_fill num_threads [4] (mapped to dims [2, 1, 0])

  %isolated_handle_1 = transform.get_closest_isolated_parent %foreach_thread_2
  %isolated_handle_2 = transform.structured.vectorize %isolated_handle_1
  %isolated_handle_3 = transform.iree.apply_patterns %isolated_handle_2 { rank_reducing }

  %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op

  %funcop = transform.structured.match ops{["func.func"]} in %variant_op_2
  %isolated_handle_4 =
    transform.iree.foreach_thread_to_gpu_and_translation_info %funcop
      { workgroup_size = [32, 2, 4] }

  // Vector distribution needs to happen on buffers.
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  transform.iree.vector.warp_distribute %isolated_handle_4
}
