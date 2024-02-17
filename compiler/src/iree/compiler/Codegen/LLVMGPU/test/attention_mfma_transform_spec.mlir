// First Matmul
// A: 64x128
// B: 32x128 MMT
// C: 64x32
#layout_a1 = #iree_vector_ext.layout<
  <[BATCHX, LANEX], [4, 16]>,
  <[BATCHY, LANEY, VECTORX], [8, 4, 4]>
>
#layout_b1 = #iree_vector_ext.layout<
  <[BATCHX, LANEX], [2, 16]>,
  <[BATCHY, LANEY, VECTORX], [8, 4, 4]>
>
#layout_c1 = #iree_vector_ext.layout<
  <[BATCHX, LANEY, VECTORX], [4, 4, 4]>,
  <[BATCHY, LANEX], [2, 16]>
>
 
// Second Matmul
// A: 128x64
// B: 64x32  MM
// C: 128x32
#layout_a2 = #iree_vector_ext.layout<
  <[BATCHX, LANEX], [8, 16]>,
  <[BATCHY, LANEY, VECTORX], [4, 4, 4]>
>
#layout_b2 = #iree_vector_ext.layout<
  <[BATCHX, LANEY, VECTORX], [4, 4, 4]>,
  <[BATCHY, LANEX], [2, 16]>
>
#layout_c2 = #iree_vector_ext.layout<
  <[BATCHX, LANEY, VECTORX], [8, 4, 4]>,
  <[BATCHY, LANEX], [2, 16]>
>

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.consumed}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_attention, %forall_grid =
    transform.structured.tile_using_forall %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Tile batch dimensions of attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %top_level_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %top_level_func : !transform.any_op

    // Promote query and output operands
    // ==========================================
    //%attention3 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    //%promoted_attention, %alloc_a0, %alloc_a1 = transform.iree.promote_operands %attention3 [0, 3]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile and decompose attention
    // ==========================================
    %attention4 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill, %inner_loop, %final_scaling, %last_truncate, %blocked_attention = transform.tile_attention %attention4 {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %fill_op, %first_matmul, %reduce_max, %partial_softmax, %scale_factor, %update, %reduce_sum, %truncate, %scale_acc, %second_matmul
        = transform.decompose_tiled_attention %blocked_attention {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Promote key and value operands
    // ==========================================
    %promoted_first_matmul, %alloc0 = transform.iree.promote_operands %first_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %promoted_second_matmul, %alloc1 = transform.iree.promote_operands %second_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile and fuse attention ops
    // ==========================================
    %tiled_matmul, %forall = transform.structured.tile_using_forall %promoted_second_matmul tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_reduce_sum, %forall_reduce = transform.structured.tile_using_forall %reduce_sum tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    %f0, %loop0 = transform.structured.fuse_into_containing_op %scale_acc into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f1, %loop1 = transform.structured.fuse_into_containing_op %truncate into %loop0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %loop4 = transform.loop.fuse_sibling %forall_reduce into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f5_1, %loop5_1 = transform.structured.fuse_into_containing_op %update into %loop4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f5, %loop5 = transform.structured.fuse_into_containing_op %scale_factor into %loop5_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f6, %loop6 = transform.structured.fuse_into_containing_op %partial_softmax into %loop5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f7, %loop7 = transform.structured.fuse_into_containing_op %reduce_max into %loop6 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f8, %loop8 = transform.structured.fuse_into_containing_op %promoted_first_matmul into %loop7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f9, %loop9 = transform.structured.fuse_into_containing_op %fill_op into %loop8 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Distribute fills
    // ==========================================
    %fills = transform.merge_handles %acc_fill, %max_fill, %sum_fill : !transform.any_op
    %tiled_fill, %fill_grid = transform.structured.tile_using_forall %fills tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Distribute last_truncate and fuse final_scaling into it
    // ==========================================
    %tiled_truncate, %loop_truncate = transform.structured.tile_using_forall %last_truncate tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %final_scaling into %loop_truncate : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.iree.fold_arith_ext_into_contraction
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %func_7 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %func_7 : !transform.any_op
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_7 : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %func_7
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    // Get the vector.contract ops.
    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op_3 :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %layoutA1 = transform.param.constant #layout_a1 -> !transform.any_param
    %layoutB1 = transform.param.constant #layout_b1 -> !transform.any_param
    %layoutC1 = transform.param.constant #layout_c1 -> !transform.any_param

    %layoutA2 = transform.param.constant #layout_a2 -> !transform.any_param
    %layoutB2 = transform.param.constant #layout_b2 -> !transform.any_param
    %layoutC2 = transform.param.constant #layout_c2 -> !transform.any_param

    transform.annotate %contract1 "__vector_layout_test_anchor_operand_0" = %layoutA1 : !transform.any_op, !transform.any_param
    transform.annotate %contract1 "__vector_layout_test_anchor_operand_1" = %layoutB1 : !transform.any_op, !transform.any_param
    transform.annotate %contract1 "__vector_layout_test_anchor_operand_2" = %layoutC1 : !transform.any_op, !transform.any_param

    transform.annotate %contract2 "__vector_layout_test_anchor_operand_0" = %layoutA2 : !transform.any_op, !transform.any_param
    transform.annotate %contract2 "__vector_layout_test_anchor_operand_1" = %layoutB2 : !transform.any_op, !transform.any_param
    transform.annotate %contract2 "__vector_layout_test_anchor_operand_2" = %layoutC2 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.amdgpu_distribute_vectors %distribute_func : !transform.any_op

    transform.apply_patterns to %distribute_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func_10 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    transform.yield
  }
} ////  module
