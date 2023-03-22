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
    %outer_loop, %mid_loop, %inner_loop, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %reduce_sum, %update,
    %softmax, %scale_acc, %second_matmul = transform.iree.tile_and_decompose_attention %attention2 :
       (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

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
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!pdl.operation) -> !pdl.operation
}
