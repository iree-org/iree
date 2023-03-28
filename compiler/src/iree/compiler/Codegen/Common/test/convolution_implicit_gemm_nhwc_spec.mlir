// RUN: iree-opt %s

transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks

  // Step 1. Match convolution and apply img2col patterns.
  // ===========================================================================
  %maybe_fill, %convolution, %maybe_trailing =
        transform.iree.match_callback failures(propagate) "convolution"(%arg0) : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  %img2col_tensor, %transformed = transform.iree.convert_conv2d_to_img2col_and_adjust_workgroup_count_region %convolution : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %1 = get_producer_of_operand %transformed[0] : (!pdl.operation) -> !pdl.operation

  // Bubble the expand_shape op on the output through the trailing elementwise.
  %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %2 {bubble_collapse} : (!pdl.operation) -> ()

  // Step 2. Tile and fuse to workgroups.
  // ===========================================================================
  %first, %rest = transform.iree.take_first %maybe_trailing, %1 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  %forall_op, %tiled_op = transform.structured.tile_to_forall_op %first num_threads [] tile_sizes [1, 32, 32](mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>])
  transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
  %3 = transform.structured.fuse_into_containing_op %rest into %forall_op
  %4 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.fuse_into_containing_op %4 into %forall_op
  %6 = transform.structured.fuse_into_containing_op %img2col_tensor into %forall_op
  transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  // Step 3. Tile the matmul reduction to scf.for and fuse the im2col gather inside the loop.
  // ===========================================================================
  %first_0, %rest_1 = transform.iree.take_first %3, %tiled_op : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  %tiled_linalg_op, %loops = transform.structured.tile_to_scf_for %first_0[0, 0, 0, 16]
  %7 = transform.structured.fuse_into_containing_op %6 into %loops
  transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  // Step 4. Promote the filter to shared memory.
  // We let bufferization promote the input so promotion is only needed for the filter.
  // ===========================================================================
  %8:2 = transform.iree.promote_operands %tiled_linalg_op [1] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

  // Step 5. Tile to threads/warps.
  // ===========================================================================
  %forall_op_2, %tiled_op_3 = transform.structured.tile_to_forall_op %7   num_threads [0, 32] tile_sizes [](mapping = [#gpu.thread<x>])
  transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
  %forall_op_4, %tiled_op_5 = transform.structured.tile_to_forall_op %rest_1   num_threads [0, 32] tile_sizes [](mapping = [#gpu.thread<x>])
  transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
  %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %5   num_threads [0, 32] tile_sizes [](mapping = [#gpu.thread<x>])
  transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
  %forall_op_8, %tiled_op_9 = transform.structured.tile_to_forall_op %8#0   num_threads [0, 1] tile_sizes [](mapping = [#gpu.warp<x>])
  transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  // Step 6. Vectorize and unroll to wmma sizes.
  // ===========================================================================
  %9 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %9 {rank_reducing_linalg, rank_reducing_vector} : (!pdl.operation) -> ()
  %10 = transform.structured.vectorize %9 {vectorize_nd_extract}
  %11 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!pdl.operation) -> !pdl.operation

  // Currently don't do anything beyond here due to the workaround used to unroll to wmma sizes.

  // // Workaround to avoid unrolling on the fused im2col operation.
  // %12:5 = split_handles %11 in[5] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  // transform.iree.apply_patterns_to_nested %12#2 {unroll_vectors_gpu_coop_mat} : (!pdl.operation) -> ()

  // // Step 7. Bufferization.
  // // ===========================================================================
  // transform.iree.apply_patterns %10 {fold_reassociative_reshapes} : (!pdl.operation) -> ()
  // transform.iree.eliminate_empty_tensors %arg0 : (!pdl.operation) -> ()
  // %13 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // transform.iree.apply_patterns %13 {erase_unnecessary_tensor_operands} : (!pdl.operation) -> ()
  // %14 = transform.iree.bufferize {target_gpu} %arg0 : (!pdl.operation) -> !pdl.operation
  // transform.iree.apply_patterns %14 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  // // Step 8. Post bufferization cleanup and mapping to blocks and threads.
  // // ===========================================================================
  // %15 = transform.structured.match ops{["func.func"]} in %14 : (!pdl.operation) -> !pdl.operation
  // transform.iree.forall_to_workgroup %15 : (!pdl.operation) -> ()
  // transform.iree.map_nested_forall_to_gpu_threads %15 workgroup_dims = [32, 1, 1] : (!pdl.operation) -> ()
  // transform.iree.hoist_static_alloc %15 : (!pdl.operation) -> ()
  // // Vectorize and distribute shared memory copies.
  // transform.iree.gpu_distribute_shared_memory_copy %15 : (!pdl.operation) -> ()
  // transform.iree.apply_patterns %14 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  // // Step 9. Convert to wmma.
  // // ===========================================================================
  // transform.iree.vector.vector_to_mma_conversion %15 {use_wmma} : (!pdl.operation) -> ()
  // transform.iree.apply_patterns %14 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
}
