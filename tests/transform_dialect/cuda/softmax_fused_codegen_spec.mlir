// RUN: iree-opt %s 
 
// Codegen
transform.structured.canonicalized_sequence failures(suppress) {
^bb1(%variant_op: !pdl.operation):
  // First level of tiling + fusion parallelizes to blocks.
  // The mapping  to block ids can only happen after bufferization atm 
  %root = transform.structured.match interface{LinalgOp} 
    attributes{iterator_types = ["parallel", "parallel", "parallel"]} in %variant_op
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
  %red = transform.structured.match interface{LinalgOp} 
    attributes{iterator_types = ["parallel", "parallel", "reduction"]} in %variant_op
  %not_root = merge_handles %fill, %red : !pdl.operation
  // This must be used with the custom dispatch region formation because IREE's
  // pulls in tensor.empty by default. This results in threadprivate allocations
  // and prevents vector distribution down the line.
  %foreach_thread, %tiled_generic = 
    transform.structured.tile_to_foreach_thread_op %root tile_sizes [1, 1]
      (mapped to dims [0, 1, 2])
  // %foreach_thread, %tiled_generic = 
    // transform.iree.tile_to_foreach_thread_and_workgroup_count_region %root tile_sizes [1, 1]
    //   (mapped to dims [0, 1, 2])
  transform.structured.fuse_into_containing_op %not_root into %foreach_thread

  // Second level of tiling + fusion parallelizes to threads.
  // Leaving the reduction untiled on threadIdx.x makes it sequential on 
  // threadIdx.x. After distribution, predication by if (threadIdx.x == 0) is
  // introduced and opportunities for distributing vector ops across warps
  // appear.
  %fill_linalg = transform.structured.match ops{["linalg.fill"]} in %variant_op
  %reduction_linalg = transform.structured.match ops{["linalg.generic"]} 
    attributes{iterator_types = ["parallel", "parallel", "reduction"]} in %variant_op
  %not_root_2 = merge_handles %fill_linalg, %reduction_linalg : !pdl.operation
  %parallel_linalg = transform.structured.match ops{["linalg.generic"]} 
    attributes{iterator_types = ["parallel", "parallel", "parallel"]} in %variant_op
  %foreach_thread_2, %parallel_linalg_2 = 
    transform.structured.tile_to_foreach_thread_op %parallel_linalg tile_sizes [1, 1, 0]
      (mapped to dims [2, 1, 0])
  transform.structured.fuse_into_containing_op %not_root_2 into %foreach_thread_2
  
  // Rank-reduce and vectorize.
  %funcx = transform.structured.match ops{["func.func"]} in %variant_op
  %funcx_2 = transform.iree.apply_patterns %funcx { rank_reducing }
  transform.structured.vectorize %funcx_2
  
  // Bufferization is necessary for:
  //   1. lowering scf.foreach_thread to workgroup (block level parallelism)
  //   2. lowering scf.foreach_thread to gpu (thread level parallelism)
  //   3. introducing predication (due to 1. + 2.) which enables rewriting to
  //      warp_execute_on_lane_0 and later vector distribution.
  %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op
  %func = transform.structured.match ops{["func.func"]} in %variant_op_2
  %func_2 = transform.iree.foreach_thread_to_workgroup %func
  transform.iree.map_nested_foreach_thread_to_gpu_threads %func_2
    { workgroup_size = [32, 1, 1] }
  
  // Vector distribution needs to happen on buffers.
  %end_func = transform.structured.match ops{["func.func"]} in %variant_op_2
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  transform.iree.vector.warp_distribute %end_func
}
