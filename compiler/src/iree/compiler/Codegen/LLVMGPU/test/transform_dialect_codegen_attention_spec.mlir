transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):

    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!pdl.operation) -> !pdl.operation

    // Tile and distribute to workgroups
    // ==========================================
    %forall_grid, %tiled_attention =
    transform.structured.tile_to_forall_op %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )

    // Tile and decompose attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %outer_loop, %max_fill, %sum_fill, %inner_loop, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %reduce_sum, %update,
    %softmax, %scale_acc, %second_matmul = transform.iree.tile_and_decompose_attention %attention2 :
       (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

    // Tile and fuse attention
    // ==========================================
    // TODO: Currently, we cannot fuse all of these ops together using tile and fuse
    %forall2, %_ = transform.structured.tile_to_forall_op %second_matmul tile_sizes [32] ( mapping = [#gpu.warp<x>])
    transform.structured.fuse_into_containing_op %scale_acc into %forall2
    transform.structured.tile_to_forall_op %softmax tile_sizes [32] ( mapping = [#gpu.warp<x>])
    transform.structured.tile_to_forall_op %update tile_sizes [32] ( mapping = [#gpu.warp<x>])
    transform.structured.tile_to_forall_op %reduce_sum tile_sizes [32] ( mapping = [#gpu.warp<x>])
    transform.structured.tile_to_forall_op %partial_softmax tile_sizes [32] ( mapping = [#gpu.warp<x>])
    transform.structured.tile_to_forall_op %reduce_max tile_sizes [32] ( mapping = [#gpu.warp<x>])
    %forall, %grid = transform.structured.tile_to_forall_op %first_matmul tile_sizes [32] ( mapping = [#gpu.warp<x>])
    transform.structured.fuse_into_containing_op %fill_op into %forall

    // Tile fill ops
    // ==========================================
    transform.structured.tile_to_forall_op %max_fill tile_sizes [32] ( mapping = [#gpu.warp<x>])
    transform.structured.tile_to_forall_op %sum_fill tile_sizes [32] ( mapping = [#gpu.warp<x>])

    // Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
    %func_3 = transform.structured.vectorize %func

    // Bufferization
    // ==========================================
    transform.iree.apply_patterns %func_3
      { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse } : (!pdl.operation) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
    transform.iree.apply_patterns %func_3 { erase_unnecessary_tensor_operands } : (!pdl.operation) -> ()
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> (!pdl.operation)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.forall_to_workgroup %func_7 : (!pdl.operation) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [128] : (!pdl.operation) -> ()

    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func_8 { canonicalization } : (!pdl.operation) -> ()
    transform.iree.apply_patterns %func_8 { cse } : (!pdl.operation) -> ()
    transform.iree.apply_buffer_optimizations %func_8 : (!pdl.operation) -> ()
}
