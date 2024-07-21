// Transform dialect specification for attention on MI300 with MFMA.
// This script only supports variants of attention with a sequence
// length that is a multiple of 64. There are two near duplicate
// because we need different tile sizes when the head dimension is 512.
// TODO: Figure out how to parameterize the tile sizes without duplicating
// the attention function.

#layout_16 = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
#layout = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>

module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Attention
//===----------------------------------------------------------------------===//

  // Utility matching for finding all undistributed fills.
  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["linalg.fill"] : !transform.any_op
    %0 = transform.get_parent_op %arg0 {allow_empty_results, nth_parent = 2 : i64, op_name = "scf.forall"} : (!transform.any_op) -> !transform.any_op
    transform.match.operation_empty %0 : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @get_undistributed_fills(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield %0 : !transform.any_op
  }

  // Script for FA2 transform pipeline when head_dim % 64 = 0.
  transform.named_sequence @__attention_main(%variant_op: !transform.any_op {transform.readonly}) {
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
    %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    %acc_fill, %max_fill, %sum_fill, %inner_loop, %final_scaling, %last_truncate, %blocked_attention = transform.iree.tile_attention %attention4 {tile_size = 32} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %scale_q, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %scale_factor, %update, %reduce_sum, %truncate, %scale_acc, %second_matmul
        = transform.iree.decompose_tiled_attention %blocked_attention {tile_size = 32} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

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

    %f10, %loop10 = transform.structured.fuse_into_containing_op %scale_q into %loop9 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Distribute fills
    // ==========================================

    // Get all fills that haven't been distributed to warps.
    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
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
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    // Get the vector.contract ops.
    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract1, %layout16x16x16 { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %layout16x16x16 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %distribute_func_2 = transform.iree.amdgpu_distribute_vectors %distribute_func : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    transform.iree.gpu_distribute_shared_memory_copy %distribute_func_2 : (!transform.any_op) -> ()
    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    transform.iree.reduce_shared_memory_bank_conflicts %distribute_func_2 : (!transform.any_op) -> ()

    transform.yield
  }

  // Script for FA2 transform pipeline for head_dim = 512.
  // For head_dim = 512, since the matmul is so big, and just try to do a single wave big load + big mfma.
  transform.named_sequence @__attention_main_len_512(%variant_op: !transform.any_op {transform.readonly}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_attention, %forall_grid =
    transform.structured.tile_using_forall %attention tile_sizes [1, 64]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Tile batch dimensions of attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    %acc_fill, %max_fill, %sum_fill, %inner_loop, %final_scaling, %last_truncate, %blocked_attention = transform.iree.tile_attention %attention4 {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %scale_q, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %scale_factor, %update, %reduce_sum, %truncate, %scale_acc, %second_matmul
        = transform.iree.decompose_tiled_attention %blocked_attention {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Promote key and value operands
    // ==========================================
    // %promoted_first_matmul, %alloc0 = transform.iree.promote_operands %first_matmul [1]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // %promoted_second_matmul, %alloc1 = transform.iree.promote_operands %second_matmul [1]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile and fuse attention ops
    // ==========================================
    %tiled_matmul, %forall = transform.structured.tile_using_forall %second_matmul tile_sizes [16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_reduce_sum, %forall_reduce = transform.structured.tile_using_forall %reduce_sum tile_sizes [16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


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
    %f8, %loop8 = transform.structured.fuse_into_containing_op %first_matmul into %loop7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f9, %loop9 = transform.structured.fuse_into_containing_op %fill_op into %loop8 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f10, %loop10 = transform.structured.fuse_into_containing_op %scale_q into %loop9 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Distribute fills
    // ==========================================

    // Get all fills that haven't been distributed to warps.
    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    %tiled_fill, %fill_grid = transform.structured.tile_using_forall %fills tile_sizes[16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Distribute last_truncate and fuse final_scaling into it
    // ==========================================
    %tiled_truncate, %loop_truncate = transform.structured.tile_using_forall %last_truncate tile_sizes[16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    // transform.print %memref_func : !transform.any_op

    // Get the vector.contract ops.
    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %layout16x16x16 = transform.param.constant #layout_16 -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract1, %layout16x16x16 { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %layout16x16x16 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %distribute_func_2 = transform.iree.amdgpu_distribute_vectors %distribute_func : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    %func_10 = transform.structured.match ops{["func.func"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func_10 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    transform.iree.reduce_shared_memory_bank_conflicts %func_11 : (!transform.any_op) -> ()

    transform.yield
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention_len_512(%attention: !transform.any_op {transform.readonly}) {
    %func = transform.get_parent_op %attention {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main_len_512, { llvm_func_attrs = { "amdgpu-waves-per-eu" = "1" } }> -> !transform.any_param
    transform.annotate %func "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention_len_512(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x512xf16> : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention(%attention: !transform.any_op {transform.readonly}) {
    %func = transform.get_parent_op %attention {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main, { llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" } }> -> !transform.any_param
    transform.annotate %func "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?xf16> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[2], 64 : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_mmt_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    // transform.print %root {name = "Generic"} : !transform.any_op
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %out: tensor<?x?xf32>):
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                            affine_map<(d0, d1, d2) -> (d1, d2)>,
                                            affine_map<(d0, d1, d2) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?xf32>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f32):
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %acc, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_mmt_f16_f16_f16(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    // transform.print %root {name = "Generic"} : !transform.any_op
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %out: tensor<?x?xf16>):
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                            affine_map<(d0, d1, d2) -> (d1, d2)>,
                                            affine_map<(d0, d1, d2) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f16):
          %10 = arith.mulf %in, %in_0 : f16
          %11 = arith.addf %acc, %10 : f16
          linalg.yield %11 : f16
        } -> tensor<?x?xf16>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    // transform.print %op {name = "Applied"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_mmt_2048x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x1280xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 320, 32]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 1, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
           subgroup_m_count = 1, subgroup_n_count = 2>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_2048x1280x5120(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x5120xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 80, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
           subgroup_m_count = 2, subgroup_n_count = 1>
         , workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">,
         llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_2048x1280x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x1280xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x1280xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
           subgroup_m_count = 2, subgroup_n_count = 1>
         , workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">,
         llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_8192x5120x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<8192x640xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<5120x640xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 32]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
           subgroup_m_count = 2, subgroup_n_count = 1>
         }>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_8192x640x2560(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<8192x2560xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<640x2560xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 160, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
           subgroup_m_count = 2, subgroup_n_count = 2>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_8192x640x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<8192x640xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<640x640xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 160, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
           subgroup_m_count = 4, subgroup_n_count = 1>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Convolution tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x640(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x640xf16>, %rhs: tensor<3x3x640x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x640xf16>, tensor<3x3x640x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 496, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           ,  workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x1280xf16>, %rhs: tensor<3x3x1280x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x1280xf16>, tensor<3x3x1280x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 288, 256, 1, 1, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [256, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>,
              subgroup_m_count = 1, subgroup_n_count = 4>
           ,  workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1920(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x1920xf16>, %rhs: tensor<3x3x1920x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x1920xf16>, tensor<3x3x1920x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 384, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           ,  workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x2560xf16>, %rhs: tensor<3x3x2560x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x2560xf16>, tensor<3x3x2560x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 512, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           ,  workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x320xf16>, %rhs: tensor<3x3x320x320xf16>, %out: tensor<2x128x128x320xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x320xf16>, tensor<3x3x320x320xf16>)
        outs(%out : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 512, 160, 1, 1, 16]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [128, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 4, subgroup_n_count = 2>
           ,  workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64x640x3x3x640(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x640xf16>, %rhs: tensor<3x3x640x640xf16>, %out: tensor<2x64x64x640xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x640xf16>, tensor<3x3x640x640xf16>)
        outs(%out : tensor<2x64x64x640xf32>) -> tensor<2x64x64x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 464, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           ,  workgroup_reorder = #iree_gpu.workgroup_reorder<reorder_option = "transpose">}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Batch matmul tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_batch_matmul_64x968x320x640(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x968x640xf16>, %rhs: tensor<64x640x320xf16>, %out: tensor<64x968x320xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x968x640xf16>, tensor<64x640x320xf16>)
        outs(%out : tensor<64x968x320xf32>) -> tensor<64x968x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 64, 64]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [64, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 4, subgroup_n_count = 1>
          , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x968x640x640(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x968x640xf16>, %rhs: tensor<64x640x640xf16>, %out: tensor<64x968x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x968x640xf16>, tensor<64x640x640xf16>)
        outs(%out : tensor<64x968x640xf32>) -> tensor<64x968x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 256, 128, 16]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [64, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 4, subgroup_n_count = 1>
          }>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x968x320x960(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x968x960xf16>, %rhs: tensor<64x960x320xf16>, %out: tensor<64x968x320xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x968x960xf16>, tensor<64x960x320xf16>)
        outs(%out : tensor<64x968x320xf32>) -> tensor<64x968x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 64, 64]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [64, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>,
              subgroup_m_count = 4, subgroup_n_count = 1>
           , llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x640x960(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x960xf16>, %rhs: tensor<64x960x640xf16>, %out: tensor<64x242x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x960xf16>, tensor<64x960x640xf16>)
        outs(%out : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 2, subgroup_n_count = 2>
           , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x1280x1280(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x1280xf16>, %rhs: tensor<64x1280x1280xf16>, %out: tensor<64x242x1280xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x1280xf16>, tensor<64x1280x1280xf16>)
        outs(%out : tensor<64x242x1280xf32>) -> tensor<64x242x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 256, 16]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          }>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x640x1280(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x1280xf16>, %rhs: tensor<64x1280x640xf16>, %out: tensor<64x242x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x1280xf16>, tensor<64x1280x640xf16>)
        outs(%out : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          }>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x640x1920(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x1920xf16>, %rhs: tensor<64x1920x640xf16>, %out: tensor<64x242x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x1920xf16>, tensor<64x1920x640xf16>)
        outs(%out : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Contraction tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_contract_3x2x20x1024x64x1280(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x1024x1280xf16>, %rhs: tensor<3x20x64x1280xf16>, %out: tensor<3x2x20x1024x64xf32>):
      %20 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<2x1024x1280xf16>, tensor<3x20x64x1280xf16>)
          outs(%out : tensor<3x2x20x1024x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %22 = arith.extf %in : f16 to f32
        %23 = arith.extf %in_0 : f16 to f32
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %acc, %24 : f32
        linalg.yield %25 : f32
      } -> tensor<3x2x20x1024x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 256, 384, 32]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
          subgroup_m_count = 4, subgroup_n_count = 1>
        , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_3x2x10x4096x64x640(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x4096x640xf16>, %rhs: tensor<3x10x64x640xf16>, %out: tensor<3x2x10x4096x64xf32>):
      %20 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<2x4096x640xf16>, tensor<3x10x64x640xf16>)
          outs(%out : tensor<3x2x10x4096x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %22 = arith.extf %in : f16 to f32
        %23 = arith.extf %in_0 : f16 to f32
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %acc, %24 : f32
        linalg.yield %25 : f32
      } -> tensor<3x2x10x4096x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128, 160, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
          subgroup_m_count = 4, subgroup_n_count = 1>
        , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_2x10x64x64x2048(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x64x2048xf16>, %rhs: tensor<10x64x2048xf16>, %out: tensor<2x10x64x64xf32>):
        %14 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
        } ins(%lhs, %rhs : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>)
          outs(%out : tensor<2x10x64x64xf32>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f32):
          %16 = arith.extf %in : f16 to f32
          %17 = arith.extf %in_0 : f16 to f32
          %18 = arith.mulf %16, %17 : f32
          %19 = arith.addf %acc, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<2x10x64x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 128, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
          subgroup_m_count = 2, subgroup_n_count = 2>
        }>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_2x20x64x64x2048(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x64x2048xf16>, %rhs: tensor<20x64x2048xf16>, %out: tensor<2x20x64x64xf32>):
        %14 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
        } ins(%lhs, %rhs : tensor<2x64x2048xf16>, tensor<20x64x2048xf16>)
          outs(%out : tensor<2x20x64x64xf32>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f32):
          %16 = arith.extf %in : f16 to f32
          %17 = arith.extf %in_0 : f16 to f32
          %18 = arith.mulf %16, %17 : f32
          %19 = arith.addf %acc, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<2x20x64x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 160, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
          subgroup_m_count = 2, subgroup_n_count = 2>
        }>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_2x20x1024x64x1280(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x1024x1280xf16>, %rhs: tensor<20x64x1280xf16>, %out: tensor<2x20x1024x64xf32>):
      %20 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
          outs(%out : tensor<2x20x1024x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %22 = arith.extf %in : f16 to f32
        %23 = arith.extf %in_0 : f16 to f32
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %acc, %24 : f32
        linalg.yield %25 : f32
      } -> tensor<2x20x1024x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
          subgroup_m_count = 4, subgroup_n_count = 1>
        }>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // Attention.
        @match_attention_len_512 -> @custom_attention_len_512,
        @match_attention -> @custom_attention

        // Matmul.
        , @match_mmt_2048x10240x1280 -> @apply_op_config
        , @match_mmt_2048x1280x5120 -> @apply_op_config
        , @match_mmt_2048x1280x1280 -> @apply_op_config
        , @match_mmt_8192x5120x640 -> @apply_op_config
        , @match_mmt_8192x640x2560 -> @apply_op_config
        , @match_mmt_8192x640x640 -> @apply_op_config

        // Convolution.
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x640 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1920 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64x640x3x3x640 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320 -> @apply_op_config

        // Batch matmul.
        , @match_batch_matmul_64x968x320x640 -> @apply_op_config
        , @match_batch_matmul_64x968x640x640 -> @apply_op_config
        , @match_batch_matmul_64x968x320x960 -> @apply_op_config
        , @match_batch_matmul_64x242x1280x1280 -> @apply_op_config
        , @match_batch_matmul_64x242x640x960 -> @apply_op_config
        , @match_batch_matmul_64x242x640x1280 -> @apply_op_config
        , @match_batch_matmul_64x242x640x1920 -> @apply_op_config

        // Contration.
        , @match_contract_3x2x20x1024x64x1280 -> @apply_op_config
        , @match_contract_3x2x10x4096x64x640 -> @apply_op_config
        , @match_contract_2x10x64x64x2048 -> @apply_op_config
        , @match_contract_2x20x64x64x2048 -> @apply_op_config
        , @match_contract_2x20x1024x64x1280 -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
