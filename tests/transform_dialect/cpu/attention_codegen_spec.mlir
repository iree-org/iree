transform.sequence failures(propagate) {
  ^bb0(%variant_op: !transform.any_op):

    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %forall_grid, %tiled_attention =
    transform.structured.tile_to_forall_op %attention num_threads [2]
      ( mapping = [#gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid
    : (!transform.any_op) -> ()

    // Tile and decompose attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %outer_loop, %max_fill, %sum_fill, %inner_loop, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %reduce_sum, %update,
    %softmax, %scale_acc, %second_matmul = tile_and_decompose_attention %attention2 :
       (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op,!transform.any_op,  !transform.any_op, !transform.any_op)

    // Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!transform.any_op) -> ()
    %func_3 = transform.structured.vectorize %func : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %variant_op
        { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()

    // Bufferization
    // ==========================================
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.iree.apply_patterns %func_3 { erase_unnecessary_tensor_operands } : (!transform.any_op) -> ()
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> (!transform.any_op)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %func_8 { canonicalization } : (!transform.any_op) -> ()
    transform.iree.apply_patterns %func_8 { cse } : (!transform.any_op) -> ()
    transform.iree.apply_buffer_optimizations %func_8 : (!transform.any_op) -> ()
}
