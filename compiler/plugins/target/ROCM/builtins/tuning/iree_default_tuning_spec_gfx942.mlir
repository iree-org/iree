// RUN: iree-opt %s

// This is just an initial tuning spec for gfx942 and is not intended for
// production use.
// TODO(https://github.com/iree-org/iree/issues/19214): Add missing
// configurations to this spec.

!in_ty = tensor<256x?xf16>
!block_in = tensor<256x64xf16>
// M/N = 256, K = 64
!shared = memref<16384xf16, #gpu.address_space<workgroup>>
// 256 -> [16, 16], 64 -> [16, 4]
//         0   2           1   3
!shared_exp = memref<16x16x16x4xf16, #gpu.address_space<workgroup>>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

module @iree_default_tuning_spec_gfx942 attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {

transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  // transform.print %op {name="Apply on"} : !transform.any_op
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  // Add a dummy unit attribute to be sure that the tuning spec applied.
  // Otherwise it would be difficult to tell if the lowering config attribute
  // comes from our tuning spec or if the compiler heuristic happened to produce
  // the same config as this script.
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  transform.yield
}

transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
                                                %config: !transform.any_param {transform.readonly},
                                                %decomposition_config: !transform.any_param {transform.readonly}) {
  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
  transform.annotate %attention "__tuning_spec_applied__" : !transform.any_op
  transform.yield
}

// ============================================================
// * Handwritten Codegen Schedules *
// ============================================================

util.func private @pingpong_large(%lhs: !in_ty, %rhs: !in_ty, %unused_acc: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16
  %lhs_shared = memref.alloc() : !shared
  %rhs_shared = memref.alloc() : !shared

  %lhs_init = tensor.extract_slice %lhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1, 2, 3]] output_shape [16, 16, 16, 4] : !shared into !shared_exp
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1, 2, 3]] output_shape [16, 16, 16, 4] : !shared into !shared_exp

  // Prefetch the data for the first tile into shared memory.
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    %lhs_vec_first = vector.extract_strided_slice %lhs_vec_local {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
    %lhs_vec_second = vector.extract_strided_slice %lhs_vec_local {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
    %ids:3 = affine.delinearize_index %id into (16, 16, 8) : index, index, index
    vector.transfer_write %lhs_vec_first, %lhs_shared_expand[%ids#0, %ids#2, %ids#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
    %p8 = arith.addi %ids#2, %c8 : index
    vector.transfer_write %lhs_vec_second, %lhs_shared_expand[%ids#0, %p8, %ids#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
  } {mapping = [#gpu.thread<linear_dim_0>]}

  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    %rhs_vec_first = vector.extract_strided_slice %rhs_vec_local {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
    %rhs_vec_second = vector.extract_strided_slice %rhs_vec_local {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
    %ids:3 = affine.delinearize_index %id into (16, 16, 8) : index, index, index
    vector.transfer_write %rhs_vec_first, %rhs_shared_expand[%ids#0, %ids#2, %ids#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
    %p8 = arith.addi %ids#2, %c8 : index
    vector.transfer_write %rhs_vec_second, %rhs_shared_expand[%ids#0, %p8, %ids#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %dim = tensor.dim %lhs, %c1 : !in_ty

  %0 = tensor.empty() : tensor<16x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<16x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c4 : index
    %k_id = arith.addi %ids#2, %c0 : index
    %k_id_p4 = arith.addi %k_id, %c4 : index
    %k_id_p8 = arith.addi %k_id_p4, %c4 : index
    %k_id_p12 = arith.addi %k_id_p8, %c4 : index
    %m_outer_id = arith.muli %ids#0, %c8 : index
    %n_outer_id = arith.muli %ids#1, %c4 : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 8 elements.
    %gko = arith.muli %delin#1, %c8 : index
    // Each subgroup loads 32 contiguous rows out of 256.
    // %bpo = arith.muli %wt#0, %c32 : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %delin#0, %c0 : index
    %glb1 = arith.addi %glb0, %c64 : index
    %glb2 = arith.addi %glb1, %c64 : index
    %glb3 = arith.addi %glb2, %c64 : index

    %wglb0:2 = affine.delinearize_index %glb0 into (16, 16) : index, index
    %wglb1:2 = affine.delinearize_index %glb1 into (16, 16) : index, index
    %wglb2:2 = affine.delinearize_index %glb2 into (16, 16) : index, index
    %wglb3:2 = affine.delinearize_index %glb3 into (16, 16) : index, index
    %wgko0 = arith.addi %wt#2, %c0 : index
    %wgko1 = arith.addi %wgko0, %c8 : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index

    // Conditional barrier:
    //  if (subgroup_id \in {0, 1, 2, 3})
    //
    // This pauses half of the waves (importantly all on different SIMDs) for
    // one portion of the pingpong loop (first set of loads). This skews
    // execution of the loop on every other wave, enabling the pingpong pattern.
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c64 to %dim step %c64 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {

      // Global loads of lhs.
      %lhs_block = tensor.extract_slice %lhs [0, %i] [256, 64] [1, 1] : !in_ty to !block_in
      %lhs_thread_0 = tensor.extract_slice %lhs_block [%glb0, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs_block [%glb1, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
      %lhs_thread_2 = tensor.extract_slice %lhs_block [%glb2, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
      %lhs_thread_3 = tensor.extract_slice %lhs_block [%glb3, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>

      %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_gpu.multi_mma %lhs_vec_0, %rhs_vec_0, %iter {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_block = tensor.extract_slice %rhs [0, %i] [256, 64] [1, 1] : !in_ty to !block_in
      %rhs_thread_0 = tensor.extract_slice %rhs_block [%glb0, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs_block [%glb1, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
      %rhs_thread_2 = tensor.extract_slice %rhs_block [%glb2, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
      %rhs_thread_3 = tensor.extract_slice %rhs_block [%glb3, %gko] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>

      %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id_p4, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id_p4, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot1 = iree_gpu.multi_mma %lhs_vec_1, %rhs_vec_1, %dot0 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id_p8, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id_p8, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id_p12, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id_p12, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_gpu.multi_mma %lhs_vec_2, %rhs_vec_2, %dot1 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_first_0 = vector.extract_strided_slice %lhs_vec_local_0 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %lhs_vec_second_0 = vector.extract_strided_slice %lhs_vec_local_0 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %lhs_vec_first_0, %lhs_shared_expand [%wglb0#0, %wgko0, %wglb0#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %lhs_vec_second_0, %lhs_shared_expand [%wglb0#0, %wgko1, %wglb0#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      %lhs_vec_first_1 = vector.extract_strided_slice %lhs_vec_local_1 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %lhs_vec_second_1 = vector.extract_strided_slice %lhs_vec_local_1 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %lhs_vec_first_1, %lhs_shared_expand [%wglb1#0, %wgko0, %wglb1#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %lhs_vec_second_1, %lhs_shared_expand [%wglb1#0, %wgko1, %wglb1#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      %lhs_vec_first_2 = vector.extract_strided_slice %lhs_vec_local_2 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %lhs_vec_second_2 = vector.extract_strided_slice %lhs_vec_local_2 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %lhs_vec_first_2, %lhs_shared_expand [%wglb2#0, %wgko0, %wglb2#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %lhs_vec_second_2, %lhs_shared_expand [%wglb2#0, %wgko1, %wglb2#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      %lhs_vec_first_3 = vector.extract_strided_slice %lhs_vec_local_3 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %lhs_vec_second_3 = vector.extract_strided_slice %lhs_vec_local_3 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %lhs_vec_first_3, %lhs_shared_expand [%wglb3#0, %wgko0, %wglb3#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %lhs_vec_second_3, %lhs_shared_expand [%wglb3#0, %wgko1, %wglb3#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      %rhs_vec_first_0 = vector.extract_strided_slice %rhs_vec_local_0 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %rhs_vec_second_0 = vector.extract_strided_slice %rhs_vec_local_0 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %rhs_vec_first_0, %rhs_shared_expand [%wglb0#0, %wgko0, %wglb0#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %rhs_vec_second_0, %rhs_shared_expand [%wglb0#0, %wgko1, %wglb0#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      %rhs_vec_first_1 = vector.extract_strided_slice %rhs_vec_local_1 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %rhs_vec_second_1 = vector.extract_strided_slice %rhs_vec_local_1 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %rhs_vec_first_1, %rhs_shared_expand [%wglb1#0, %wgko0, %wglb1#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %rhs_vec_second_1, %rhs_shared_expand [%wglb1#0, %wgko1, %wglb1#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      %rhs_vec_first_2 = vector.extract_strided_slice %rhs_vec_local_2 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %rhs_vec_second_2 = vector.extract_strided_slice %rhs_vec_local_2 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %rhs_vec_first_2, %rhs_shared_expand [%wglb2#0, %wgko0, %wglb2#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %rhs_vec_second_2, %rhs_shared_expand [%wglb2#0, %wgko1, %wglb2#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      %rhs_vec_first_3 = vector.extract_strided_slice %rhs_vec_local_3 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      %rhs_vec_second_3 = vector.extract_strided_slice %rhs_vec_local_3 {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}: vector<1x8xf16> to vector<1x4xf16>
      vector.transfer_write %rhs_vec_first_3, %rhs_shared_expand [%wglb3#0, %wgko0, %wglb3#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp
      vector.transfer_write %rhs_vec_second_3, %rhs_shared_expand [%wglb3#0, %wgko1, %wglb3#1, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, !shared_exp

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot3 = iree_gpu.multi_mma %lhs_vec_3, %rhs_vec_3, %dot2 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot3 : vector<8x4x1x4xf32>
    }

    // Conditional barrier:
    //  if (subgroup_id \in {4, 5, 6, 7})
    //
    // This synchronizes all wave slots that passed through on the first
    // conditional barrier, reuniting all waves in the workgroup.
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue.
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot0 = iree_gpu.multi_mma %lhs_vec_0, %rhs_vec_0, %3 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id_p4, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id_p4, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot1 = iree_gpu.multi_mma %lhs_vec_1, %rhs_vec_1, %dot0 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id_p8, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id_p8, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot2 = iree_gpu.multi_mma %lhs_vec_2, %rhs_vec_2, %dot1 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %k_id_p12, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %k_id_p12, %ids#3, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot3 = iree_gpu.multi_mma %lhs_vec_3, %rhs_vec_3, %dot2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %tp = vector.transpose %dot3, [0, 2, 1, 3] : vector<8x4x1x4xf32> to vector<8x1x4x4xf32>
    %empty = tensor.empty() : tensor<8x1x4x4xf32>
    %4 = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x1x4x4xf32>, tensor<8x1x4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%m_outer_id, %ids#3, %n_outer_id, %inner_id] [8, 1, 4, 4] [1, 1, 1, 1] : tensor<8x1x4x4xf32> into tensor<16x16x16x16xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %collapse = tensor.collapse_shape %1 [[0, 1], [2, 3]] : tensor<16x16x16x16xf32> into tensor<256x256xf32>
  util.return %collapse : tensor<256x256xf32>
}

/// Entry point for custom lowering strategy for pingpong. Rewrites a matmul
/// as follows:
///
///  %acc = linalg.fill 0.0 : tensor<256x256xf32>
///  %0 = linalg.mmt {lowering_strategy = "cast_and_call_pingpong_matmul"}
///    ins(%lhs, %rhs: tensor<256x?xf16>, tensor<256x?xf16>)
///    outs(%acc: tensor<256x256xf32>)
///
/// to
///
///  %0 = call @pingpong_large(%lhs, %rhs) -> tensor<256x256xf32>
///
/// And then inlines the call.
transform.named_sequence @cast_and_call_pingpong_matmul(%mm: !transform.any_op {transform.readonly}) {
  %module = transform.util.get_nearest_symbol_table %mm : (!transform.any_op) -> !transform.any_op
  %func = transform.util.lookup_nearest_symbol_from_self @pingpong_large : !transform.any_op
  %ins = transform.get_operand %mm[all] : (!transform.any_op) -> !transform.any_value
  %out = transform.get_result %mm[all] : (!transform.any_op) -> !transform.any_value
  // Replace
  transform.util.cast_and_call inline_call %func(%ins) -> %out after %mm {
        transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
  transform.yield
}

transform.named_sequence @match_mmt_f16_f16_f32_impl(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>):
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<?x?xf32>
    %out = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
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

transform.named_sequence
@match_mmt_f16_f16_f32_ge_4k(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // TODO(qedawkins): Support > 2d
  transform.iree.match.cast_compatible_type %lhs = tensor<?x?xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<?x?xf16> : !transform.any_value

  // TODO(qedawkins): Add a bounds checking match op so we can check alignment
  // and min size separately.
  transform.iree.match.dim_is_multiple_of  %lhs[0], 4096 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[1], 4096 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 4096 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 4096 : !transform.any_value

  // Lowering config for pingpong large. "cast_and_call_pingpong_matmul" refers
  // to the custom lowering strategy to use, which in this case replaces the
  // matmul with a call to the @pingpong_large implementation above.
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [256, 256, 0],
      lowering_strategy = "cast_and_call_pingpong_matmul"}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
      workgroup_size = [512, 1, 1] subgroup_size = 64,
      // This strategy uses the maximum amount of possible shared memory on
      // all gfx942 architectures so shared memory padding to reduce bank
      // conflicts must be disabled. Also prefetching is done manually in the
      // above and is disabled here as well.
      {gpu_pipeline_options =
        #iree_gpu.pipeline_options<
          prefetch_shared_memory = false,
          no_reduce_shared_memory_bank_conflicts = true>,
      // This strategy requires 2 waves per SIMD.
        llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

/// Applies the op config for pingpong_large. This requires importing external
/// symbols needed for the custom lowering (in this case inline + replace).
transform.named_sequence @apply_pingpong_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  %module = transform.util.get_nearest_symbol_table %op : (!transform.any_op) -> !transform.any_op

  // Create and serialize a module with the needed symbols.
  %syms = transform.util.create_serialized_module {
    ^bb0(%m: !transform.any_op):
      transform.util.import_symbol @cast_and_call_pingpong_matmul into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.util.import_symbol @pingpong_large into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.annotate %m "transform.with_named_sequence" : !transform.any_op
  } -> !transform.any_param

  // Annotate the parent function with the serialized module.
  %func = transform.get_parent_op %op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  transform.annotate %func "iree_codegen_external_symbols" = %syms : !transform.any_op, !transform.any_param
  transform.yield
}

// ============================================================
// * Tuning Configurations Start *
// ============================================================

transform.named_sequence @match_attention_f16(%root: !transform.any_op {transform.readonly})
  -> !transform.any_op {
  transform.match.operation_name %root ["iree_linalg_ext.attention"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%query: tensor<?x?x?x?xf16>,
         %key: tensor<?x?x?x?xf16>,
         %value: tensor<?x?x?x?xf16>,
         %softmax_scale: f16,
         %out: tensor<?x?x?x?xf16>):

      %attn = iree_linalg_ext.attention {indexing_maps = [
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, M, K1)>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, K2, K1)>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, N, K2)>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> ()>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, M, N)>]}
        ins(%query, %key, %value, %softmax_scale :
            tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f16)
        outs(%out : tensor<?x?x?x?xf16>){
          ^bb0(%arg0: f32):
            iree_linalg_ext.yield %arg0 : f32
        } -> tensor<?x?x?x?xf16>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)

  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_attention_2x10x4096x64x64x64_f16(%attention: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param, !transform.any_param) {

  %matched = transform.include @match_attention_f16 failures(propagate) (%attention)
    : (!transform.any_op) -> !transform.any_op

  %query = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
  %key = transform.get_operand %attention[1] : (!transform.any_op) -> !transform.any_value
  %value = transform.get_operand %attention[2] : (!transform.any_op) -> !transform.any_value

  transform.iree.match.cast_compatible_type %query = tensor<?x?x?x?xf16> : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %query[2], 128 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %query[3], 16 : !transform.any_value
  transform.iree.match.cast_compatible_type %key = tensor<?x?x64x64xf16> : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %key[2], 64 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %key[3], 16 : !transform.any_value
  transform.iree.match.cast_compatible_type %value = tensor<?x?x64x64xf16> : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %value[2], 16 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %value[3], 64 : !transform.any_value

  // `amdgpu-waves-per-eu`:
  // The gfx942 GPU attention implementation uses a high number of registers.
  // Setting this flag instructs the compiler to be less conservative in register allocation,
  // leading to better performance.

  // `denormal-fp-math-f32`:
  // Disables denormal flushing for `exp2/exp` operations, reducing the number of instructions
  // required for exp/exp2.
  %config = transform.param.constant #iree_codegen.compilation_info<
          lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
          translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                            workgroup_size = [256]
                                                            subgroup_size = 64 ,
            {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
  -> !transform.any_param

  // `promote_operands = [1]`:
  // - Only `K` and `V` tensors are promoted to shared memory.
  // - `Q` is not promoted since the `QK` matrix multiplication uses VMFMA instructions,
  //   which operate efficiently with `vector<8xf16>` from global memory.
  %decomposition_config = transform.param.constant {
    qk_attrs = {attention_qk_matmul,
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F16>,
                                                              subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
    pv_attrs = {attention_pv_matmul,
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                              subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
  } -> !transform.any_param

  transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
}

transform.named_sequence @match_mmt_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
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

transform.named_sequence
@match_mmt_2048x1280x5120_f16_f16_f32(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<2048x5120xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                 subgroup_m_count = 2, subgroup_n_count = 2,
                                                 reduction = [0, 0, 64],
                                                 workgroup = [64, 128, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [256, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence
@__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
  attributes { iree_codegen.tuning_spec_entrypoint } {
  %res = transform.foreach_match in %variant_op
    @match_mmt_f16_f16_f32_ge_4k -> @apply_pingpong_op_config,
    // Expected speedup: 1.22x.
    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config,
    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}

}
