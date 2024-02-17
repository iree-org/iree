// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @codegen(
      %variant_op: !transform.any_op {transform.consumed}) {

    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %reduction = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Step 1. First level of tiling + fusion parallelizes to blocks.
    // ===========================================================================
    %grid_reduction, %forall_grid =
      transform.structured.tile_using_forall %reduction tile_sizes [1]
        ( mapping = [#gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()
    transform.structured.fuse_into_containing_op %fill into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 2. Split the reduction to get meatier parallelism.
    // ===========================================================================
    %block_more_parallel_fill_op_2, %block_more_parallel_op_2, %block_combiner_op_2, %forall =
      transform.structured.tile_reduction_using_for %grid_reduction by tile_sizes = [0, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %_1:2 =
      transform.structured.tile_using_forall %block_more_parallel_op_2 num_threads [0, 32]
      ( mapping = [#gpu.thread<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3. Second level of tiling parallelizes to threads.
    // ===========================================================================
    // 1st op is [parallel, parallel], map it to threadIdx.x by 4.
    %_2:2 =
      transform.structured.tile_using_forall %block_more_parallel_fill_op_2 tile_sizes [0, 4]
      ( mapping = [#gpu.thread<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // 2nd op is [parallel, reduction] of 1x128, map the 1-dim to threadIdx.y to
    // trigger mapping of the reduction to threadIdx.x via predication via `if (x==0)`.
    %_3:2 =
      transform.structured.tile_using_forall %block_combiner_op_2 tile_sizes [1]
      ( mapping = [#gpu.thread<y>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 4. Rank-reduce and vectorize.
    // ===========================================================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op

    // Step 5. Bufferize and drop HAL decriptor from memref ops.
    // ===========================================================================
    // Canonicalization/CSE is needed before bufferization otherwise unnecessary
    // allocs will be created.
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %func_5 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_5 {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

    // Step 6. Post-bufferization mapping to blocks and threads.
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7
        workgroup_dims = [32, 1, 1] : (!transform.any_op) -> ()

    // Step 7. Post-bufferization vector distribution with rank-reduction.
    // ===========================================================================
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3
      : (!transform.any_op) -> !transform.any_op
    %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 } : (!transform.any_op) -> !transform.any_op
    transform.iree.vector.warp_distribute %func_7
      : (!transform.any_op) -> ()

    // Late canonicalizations to cleanup and pass the checks
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func_7 : !transform.any_op
    transform.apply_cse to %func_7 : !transform.any_op

    // Annotate the exported function as already translated.
    %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    %none = transform.param.constant #iree_codegen.translation_info<None> -> !transform.any_param
    transform.annotate %exports "translation_info" = %none : !transform.any_op, !transform.any_param

    transform.yield
  }
} // module
