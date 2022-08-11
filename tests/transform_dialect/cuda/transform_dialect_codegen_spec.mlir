// RUN: iree-opt %s 

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    
    // Note: split by 32 to vector-distribute the tail combiner_op, but
    // split by 2 to vector-distribute the meaty %more_parallel_op
    %init_or_alloc_op, %fill_op, %more_parallel_op, %combiner_op = 
      transform.structured.split_reduction %0 
        { split_factor = 2, insert_split_dimension = 1, use_alloc }
    
    %1 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %foreach_thread_1, %tiled_fill = 
      transform.structured.tile_to_foreach_thread_op %fill_op num_threads [8, 2] (mapped to dims [2, 1, 0])
    %foreach_thread_2, %tiled_more_parallel_op = 
       transform.structured.tile_to_foreach_thread_op %more_parallel_op num_threads [8, 2] (mapped to dims [2, 1, 0])
    %foreach_thread_3, %tiled_combiner_op = 
      transform.structured.tile_to_foreach_thread_op %combiner_op num_threads [8] (mapped to dims [2, 1, 0])

    %isolated_handle_1 = transform.get_closest_isolated_parent %foreach_thread_2
    %isolated_handle_2 = transform.structured.vectorize %isolated_handle_1
    %isolated_handle_3 = transform.iree.apply_patterns %isolated_handle_2 { rank_reducing }

    transform.iree.bufferize { target_gpu }
    %isolated_handle_4 = 
      transform.iree.foreach_thread_to_gpu_and_translation_info %isolated_handle_3 
        { workgroup_size = [32, 2, 8] }
    
    // Vector distribution needs to happen on buffers.
    %if_op = transform.structured.match ops{["scf.if"]} in %arg1
    %warp = transform.iree.vector.warp_execute_on_lane_0 %if_op { warp_size = 32 }
    transform.iree.vector.warp_distribute %isolated_handle_4
    
    // transform.print { name = "after codegen"}
  }
}
