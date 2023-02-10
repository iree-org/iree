// RUN: iree-opt %s


// Pasting an actionnable command here for sanity:
//
// ./build/tools/iree-opt ./tests/transform_dialect/cuda/matmul.mlir \
//   --iree-hal-target-backends=cuda \
//   --iree-abi-transformation-pipeline \
//   --iree-flow-transformation-pipeline \
//   --iree-stream-transformation-pipeline \
//   --iree-hal-configuration-pipeline | \
// ./build/tools/iree-opt \
//    --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//    --iree-codegen-llvmgpu-use-transform-dialect=tests/transform_dialect/cuda/matmul_codegen_spec.mlir \
//    --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
// head -n -30

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to foreach_thread and sequential scf.for.
  // ======================================================
  %foreach_thread_l1, %matmul_l1 =
    transform.iree.tile_to_foreach_thread_and_workgroup_count_region %matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  %matmul_l2, %loops:3 = transform.structured.tile_to_scf_for %matmul_l1 [16, 16, 16]

  
  // Step 2. Rank-reduce and vectorize.
  // ==================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  %func_v_3 = transform.structured.vectorize %func_v_2
  %func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_mma }

  // Step 3. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize %variant_op_2
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_m_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_m

  // This must occur after bufferization because of the fancy CUDA types.
  %func_m_3 = transform.iree.vector.vector_to_mma_conversion %func_m_2

  // Step 3. Post-bufferization mapping workgroup.
  // =============================================
  transform.iree.foreach_thread_to_workgroup %func_m_3
}
