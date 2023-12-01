// RUN: iree-opt %s

// Codegen
module attributes { transform.with_named_sequence } {
  transform.named_sequence @codegen(
      %variant_op: !transform.any_op {transform.consumed}) {

    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
      in %variant_op : (!transform.any_op) -> !transform.any_op
    %input_max_fill,
    %input_max,
    %exps_sum_fill,
    %exps,
    %exps_sum,
    %div = transform.split_handle %ops
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op,
                            !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 1. First level of tiling + fusion parallelizes to blocks.
    // ==============================================================
    %_, %forall =
      transform.structured.tile_using_forall %div tile_sizes [1, 4]
        ( mapping = [#gpu.block<x>, #gpu.block<y>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall : (!transform.any_op) -> ()

    // TODO: Merging and fusing merged handles does not work properly atm.
    transform.structured.fuse_into_containing_op %exps_sum into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %exps into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %exps_sum_fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %input_max into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %input_max_fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // By default, fusion into scf.forall does not promote captured values
    // to shared as this involves a cross-thread dependence analysis.
    // Instead, we activate it explicitly post-hoc to promote all the extract_slice
    // ops that we find and match the prerequisites
    %forall_with_type = transform.cast %forall : !transform.any_op to !transform.op<"scf.forall">
    transform.iree.share_forall_operands %forall_with_type
      : (!transform.op<"scf.forall">) -> !transform.op<"scf.forall">
    transform.apply_patterns to %variant_op {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_cse %variant_op : !transform.any_op

    // Step 2. Second level of tiling + fusion parallelizes to threads.
    // ================================================================
    %tiled_ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
      in %variant_op : (!transform.any_op) -> !transform.any_op
    %tiled_input_max_fill,
    %tiled_input_max,
    %tiled_exps_sum_fill,
    %tiled_exp_and_exps_sum,
    %tiled_exp_and_exps_sum_2,
    %tiled_div = transform.split_handle %tiled_ops
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op,
                            !transform.any_op, !transform.any_op, !transform.any_op)
    // Leaving the reduction untiled on threadIdx.x makes it sequential on
    // threadIdx.x. After distribution, predication by `if (threadIdx.x == 0)` is
    // introduced and opportunities for distributing vector ops across warps
    // appear.
    %reduction_linalg_ops = transform.merge_handles %tiled_input_max,
                                                    %tiled_exp_and_exps_sum,
                                                    %tiled_exp_and_exps_sum_2
      : !transform.any_op
    transform.structured.tile_using_forall %reduction_linalg_ops tile_sizes [1, 1]
      ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fully parallel ops are tiled and mapped.
    %parallel_linalg_ops = transform.merge_handles %tiled_input_max_fill,
                                                  %tiled_exps_sum_fill,
                                                  %tiled_div
      : !transform.any_op
    transform.structured.tile_using_forall %parallel_linalg_ops num_threads [1, 4, 32]
        ( mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3. Rank-reduce and vectorize.
    // ==================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op

    // Step 4. Bufferize and drop HAL decriptor from memref ops.
    // =========================================================
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> !transform.any_op
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

    // Step 5. Post-bufferization mapping to blocks and threads.
    // =========================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_2 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_2 workgroup_dims = [32, 4, 1] : (!transform.any_op) -> ()

    // Step 6. Post-bufferization vector distribution with rank-reduction.
    // ===================================================================
    %end_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %end_func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 } : (!transform.any_op) -> !transform.any_op
    transform.iree.vector.warp_distribute %end_func : (!transform.any_op) -> ()

    transform.yield
  }
} // module
