module attributes { transform.with_named_sequence } {
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

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %blocked_att, %wg_forall =
    transform.structured.tile_using_forall %attention tile_sizes [1, 128, 0, 0, 0]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Cleanup
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Convert to online attention
    // ==========================================
    transform.iree.convert_to_online_attention %blocked_att : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Tile along K2
    %online_att = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %k2_for = transform.structured.tile_using_for %online_att tile_sizes [0, 0, 0, 64, 0]: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Promote Key and Value operands
    // ==========================================
    %attt = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %promoted_att, %alloc0, %alloc1 = transform.iree.promote_operands %attt [1, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // This is a hack... We distribute the loop and the merging seperately and just assume they are fused.
    %warp_attention = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %warp_merge = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %warp_attention tile_sizes[0, 32, 0, 0, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %warp_merge tile_sizes[0, 32, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill = transform.split_handle %fills : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %acc_fill tile_sizes[0, 32, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %max_fill tile_sizes[0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %sum_fill tile_sizes[0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Decompose attention
    %func2 = transform.apply_registered_pass "iree-linalg-ext-decompose-attention" to %func : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func2 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func2 : (!transform.any_op) -> (!transform.any_op)

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
    %func_4 = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.iree.prepare_vector_to_mma
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [4, 8, 4] subgroup_size = 32 sync_after_distribution = false : (!transform.any_op) -> ()

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
    transform.apply_registered_pass "iree-codegen-optimize-vector-transfer" to %func_8 : (!transform.any_op) -> (!transform.any_op)
    transform.yield

  }
} ////  module
