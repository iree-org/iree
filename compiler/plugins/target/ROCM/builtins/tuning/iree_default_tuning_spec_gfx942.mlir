// RUN: iree-opt %s

// This is just an initial tuning spec for gfx942 and is not intended for
// production use.
// TODO(https://github.com/iree-org/iree/issues/19214): Add missing
// configurations to this spec.

!in_ty = tensor<256x?xf16>
!exp_in_ty = tensor<1x256x?xf16>
!block_in = tensor<256x64xf16>
!exp_block_in = tensor<1x256x64xf16>
!flat_shared = memref<16384xf16, #gpu.address_space<workgroup>>
!shared = memref<256x64xf16, #gpu.address_space<workgroup>>
!shared_exp = memref<16x16x4x16xf16, #gpu.address_space<workgroup>>

!mexp_in_ty = tensor<1x128x?xf16>
!mexp_block_in = tensor<1x128x64xf16>
!mflat_shared = memref<8192xf16, #gpu.address_space<workgroup>>
!mshared = memref<128x64xf16, #gpu.address_space<workgroup>>
!mshared_exp = memref<8x16x4x16xf16, #gpu.address_space<workgroup>>

!in_ty_f8 = tensor<256x?xf8E4M3FNUZ>
!exp_in_ty_f8 = tensor<1x256x?xf8E4M3FNUZ>
!block_in_f8 = tensor<256x128xf8E4M3FNUZ>
!exp_block_in_f8 = tensor<1x256x128xf8E4M3FNUZ>
!flat_shared_f8 = memref<32768xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!shared_f8 = memref<256x128xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!shared_exp_f8 = memref<16x16x4x32xf8E4M3FNUZ, #gpu.address_space<workgroup>>

!mexp_in_ty_f8 = tensor<1x128x?xf8E4M3FNUZ>
!mexp_block_in_f8 = tensor<1x128x128xf8E4M3FNUZ>
!mflat_shared_f8 = memref<16384xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!mshared_f8 = memref<128x128xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!mshared_exp_f8 = memref<8x16x4x32xf8E4M3FNUZ, #gpu.address_space<workgroup>>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

module @iree_default_tuning_spec_gfx942 attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {

// ============================================================
// * Handwritten Codegen Schedules *
// ============================================================

// ------------------------------------------------------------
// * Large Pingpong f16 *
// ------------------------------------------------------------

util.func private @pingpong_large(%lhs_base: !in_ty, %rhs_base: !in_ty, %unused_acc: tensor<256x256xf32>) -> tensor<256x256xf32> {
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
  %lhs_shared_base = memref.alloc() : !flat_shared
  %rhs_shared_base = memref.alloc() : !flat_shared

  %dim = tensor.dim %lhs_base, %c1 : !in_ty
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim) : !in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim) : !in_ty

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared

  %lhs_init = tensor.extract_slice %lhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in

  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp

  %0 = tensor.empty() : tensor<16x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<16x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c4 : index
    %m_outer_id = arith.muli %ids#0, %c8 : index
    %n_outer_id = arith.muli %ids#1, %c4 : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 8 elements.
    %gko = arith.muli %wt#2, %c8 : index
    // Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo : index
    %glb1 = arith.addi %glb0, %c8 : index
    %glb2 = arith.addi %glb1, %c8 : index
    %glb3 = arith.addi %glb2, %c8 : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
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

      %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

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

      %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

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

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

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

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_2, %lhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_3, %lhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

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
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot0 = iree_gpu.multi_mma %lhs_vec_0, %rhs_vec_0, %3 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot1 = iree_gpu.multi_mma %lhs_vec_1, %rhs_vec_1, %dot0 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot2 = iree_gpu.multi_mma %lhs_vec_2, %rhs_vec_2, %dot1 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
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

// Expanded Variant
util.func private @pingpong_large_expanded(%lhs_base: !exp_in_ty, %rhs_base: !in_ty, %unused_acc: tensor<1x256x256xf32>) -> tensor<1x256x256xf32> {
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
  %lhs_shared_base = memref.alloc() : !flat_shared
  %rhs_shared_base = memref.alloc() : !flat_shared

  %dim = tensor.dim %rhs_base, %c1 : !in_ty
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim) : !exp_in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim) : !in_ty

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared

  %lhs_init = tensor.extract_slice %lhs [0, 0, 0] [1, 256, 64] [1, 1, 1] : !exp_in_ty to !exp_block_in
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in

  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [0, %delin#0, %vec] [1, 1, 8] [1, 1, 1] : !exp_block_in to tensor<1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp

  %0 = tensor.empty() : tensor<1x16x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x16x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c4 : index
    %m_outer_id = arith.muli %ids#0, %c8 : index
    %n_outer_id = arith.muli %ids#1, %c4 : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 8 elements.
    %gko = arith.muli %wt#2, %c8 : index
    // Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo : index
    %glb1 = arith.addi %glb0, %c8 : index
    %glb2 = arith.addi %glb1, %c8 : index
    %glb3 = arith.addi %glb2, %c8 : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c64 to %dim step %c64 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {

      // Global loads of lhs.
      %lhs_block = tensor.extract_slice %lhs [0, 0, %i] [1, 256, 64] [1, 1, 1] : !exp_in_ty to !exp_block_in
      %lhs_thread_0 = tensor.extract_slice %lhs_block [0, %glb0, %gko] [1, 1, 8] [1, 1, 1] : !exp_block_in to tensor<1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs_block [0, %glb1, %gko] [1, 1, 8] [1, 1, 1] : !exp_block_in to tensor<1x1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
      %lhs_thread_2 = tensor.extract_slice %lhs_block [0, %glb2, %gko] [1, 1, 8] [1, 1, 1] : !exp_block_in to tensor<1x1x8xf16>
      %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
      %lhs_thread_3 = tensor.extract_slice %lhs_block [0, %glb3, %gko] [1, 1, 8] [1, 1, 1] : !exp_block_in to tensor<1x1x8xf16>
      %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>

      %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

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

      %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

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

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

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

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_2, %lhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_3, %lhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

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
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot0 = iree_gpu.multi_mma %lhs_vec_0, %rhs_vec_0, %3 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot1 = iree_gpu.multi_mma %lhs_vec_1, %rhs_vec_1, %dot0 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot2 = iree_gpu.multi_mma %lhs_vec_2, %rhs_vec_2, %dot1 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot3 = iree_gpu.multi_mma %lhs_vec_3, %rhs_vec_3, %dot2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %tp = vector.transpose %dot3, [0, 2, 1, 3] : vector<8x4x1x4xf32> to vector<8x1x4x4xf32>
    %empty = tensor.empty() : tensor<1x8x1x4x4xf32>
    %4 = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x1x4x4xf32>, tensor<1x8x1x4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[0, %m_outer_id, %ids#3, %n_outer_id, %inner_id] [1, 8, 1, 4, 4] [1, 1, 1, 1, 1] : tensor<1x8x1x4x4xf32> into tensor<1x16x16x16x16xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %collapse = tensor.collapse_shape %1 [[0], [1, 2], [3, 4]] : tensor<1x16x16x16x16xf32> into tensor<1x256x256xf32>
  util.return %collapse : tensor<1x256x256xf32>
}

// ------------------------------------------------------------
// * Large Pingpong f8 *
// ------------------------------------------------------------

util.func private @pingpong_large_f8_expanded(%lhs_base: !exp_in_ty_f8, %rhs_base: !in_ty_f8, %unused_acc: tensor<1x256x256xf32>) -> tensor<1x256x256xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ
  %lhs_shared_base = memref.alloc() : !flat_shared_f8
  %rhs_shared_base = memref.alloc() : !flat_shared_f8

  %dim = tensor.dim %rhs_base, %c1 : !in_ty_f8
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim) : !exp_in_ty_f8
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim) : !in_ty_f8

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<128, 8>] : !flat_shared_f8
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<128, 8>] : !flat_shared_f8

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [256, 128] : !flat_shared_f8 into !shared_f8
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 128] : !flat_shared_f8 into !shared_f8

  %lhs_init = tensor.extract_slice %lhs [0, 0, 0] [1, 256, 128] [1, 1, 1] : !exp_in_ty_f8 to !exp_block_in_f8
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 128] [1, 1] : !in_ty_f8 to !block_in_f8

  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c16 : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [0, %delin#0, %vec] [1, 1, 16] [1, 1, 1] : !exp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c16 : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 32] : !shared_f8 into !shared_exp_f8
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 32] : !shared_f8 into !shared_exp_f8

  %0 = tensor.empty() : tensor<1x16x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x16x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c8 : index
    %inner_id_acc = arith.muli %ids#2, %c4 : index
    %m_outer_id = arith.muli %ids#0, %c8 : index
    %n_outer_id = arith.muli %ids#1, %c4 : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 16 elements.
    %gko = arith.muli %wt#2, %c16 : index
    // Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo : index
    %glb1 = arith.addi %glb0, %c8 : index
    %glb2 = arith.addi %glb1, %c8 : index
    %glb3 = arith.addi %glb2, %c8 : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c128 to %dim step %c128 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {

      // Global loads of lhs.
      %lhs_block = tensor.extract_slice %lhs [0, 0, %i] [1, 256, 128] [1, 1, 1] : !exp_in_ty_f8 to !exp_block_in_f8
      %lhs_thread_0 = tensor.extract_slice %lhs_block [0, %glb0, %gko] [1, 1, 16] [1, 1, 1] : !exp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs_block [0, %glb1, %gko] [1, 1, 16] [1, 1, 1] : !exp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %lhs_thread_2 = tensor.extract_slice %lhs_block [0, %glb2, %gko] [1, 1, 16] [1, 1, 1] : !exp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %lhs_thread_3 = tensor.extract_slice %lhs_block [0, %glb3, %gko] [1, 1, 16] [1, 1, 1] : !exp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>

      %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_gpu.multi_mma %lhs_vec_0, %rhs_vec_0, %iter {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_block = tensor.extract_slice %rhs [0, %i] [256, 128] [1, 1] : !in_ty_f8 to !block_in_f8
      %rhs_thread_0 = tensor.extract_slice %rhs_block [%glb0, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs_block [%glb1, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %rhs_thread_2 = tensor.extract_slice %rhs_block [%glb2, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %rhs_thread_3 = tensor.extract_slice %rhs_block [%glb3, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>

      %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot1 = iree_gpu.multi_mma %lhs_vec_1, %rhs_vec_1, %dot0 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>

      %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_gpu.multi_mma %lhs_vec_2, %rhs_vec_2, %dot1 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %lhs_vec_local_2, %lhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %lhs_vec_local_3, %lhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot3 = iree_gpu.multi_mma %lhs_vec_3, %rhs_vec_3, %dot2 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot3 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>
    %dot0 = iree_gpu.multi_mma %lhs_vec_0, %rhs_vec_0, %3 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>
    %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>
    %dot1 = iree_gpu.multi_mma %lhs_vec_1, %rhs_vec_1, %dot0 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>
    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>
    %dot2 = iree_gpu.multi_mma %lhs_vec_2, %rhs_vec_2, %dot1 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>
    %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x1x8xf8E4M3FNUZ>
    %dot3 = iree_gpu.multi_mma %lhs_vec_3, %rhs_vec_3, %dot2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %tp = vector.transpose %dot3, [0, 2, 1, 3] : vector<8x4x1x4xf32> to vector<8x1x4x4xf32>
    %empty = tensor.empty() : tensor<1x8x1x4x4xf32>
    %4 = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x1x4x4xf32>, tensor<1x8x1x4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[0, %m_outer_id, %ids#3, %n_outer_id, %inner_id_acc] [1, 8, 1, 4, 4] [1, 1, 1, 1, 1] : tensor<1x8x1x4x4xf32> into tensor<1x16x16x16x16xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %collapse = tensor.collapse_shape %1 [[0], [1, 2], [3, 4]] : tensor<1x16x16x16x16xf32> into tensor<1x256x256xf32>
  util.return %collapse : tensor<1x256x256xf32>
}

// ------------------------------------------------------------
// * Medium Pingpong f16 *
// ------------------------------------------------------------

util.func private @pingpong_medium_expanded(%lhs_base: !mexp_in_ty, %rhs_base: !in_ty, %unused_acc: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16
  %lhs_shared_base = memref.alloc() : !mflat_shared
  %rhs_shared_base = memref.alloc() : !flat_shared

  %dim = tensor.dim %rhs_base, %c1 : !in_ty
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim) : !mexp_in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim) : !in_ty

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !mflat_shared
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [128, 64] : !mflat_shared into !mshared
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared

  %lhs_init = tensor.extract_slice %lhs [0, 0, 0] [1, 128, 64] [1, 1, 1] : !mexp_in_ty to !mexp_block_in
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in

  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (128, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [0, %delin#0, %vec] [1, 1, 8] [1, 1, 1] : !mexp_block_in to tensor<1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !mshared
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [8, 16, 4, 16] : !mshared into !mshared_exp
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp

  %0 = tensor.empty() : tensor<1x8x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x8x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c4 : index
    %m_outer_id = arith.muli %ids#0, %c4 : index
    %n_outer_id = arith.muli %ids#1, %c4 : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 8 elements.
    %gko = arith.muli %wt#2, %c8 : index
    // RHS indexing. Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo : index
    %glb1 = arith.addi %glb0, %c8 : index
    %glb2 = arith.addi %glb1, %c8 : index
    %glb3 = arith.addi %glb2, %c8 : index
    // LHS indexing.
    %bpo_lhs = arith.muli %wt#0, %c16 : index
    %glb0_lhs = arith.addi %wt#1, %bpo_lhs : index
    %glb1_lhs = arith.addi %glb0_lhs, %c8 : index

    %2 = arith.constant dense<0.0> : vector<4x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c64 to %dim step %c64 iter_args(%iter = %2) -> vector<4x4x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp, vector<4x1x2x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x2x4xf16>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>

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

      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp, vector<4x1x2x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x2x4xf16>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_block = tensor.extract_slice %lhs [0, 0, %i] [1, 128, 64] [1, 1, 1] : !mexp_in_ty to !mexp_block_in
      %lhs_thread_0 = tensor.extract_slice %lhs_block [0, %glb0_lhs, %gko] [1, 1, 8] [1, 1, 1] : !mexp_block_in to tensor<1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs_block [0, %glb1_lhs, %gko] [1, 1, 8] [1, 1, 1] : !mexp_block_in to tensor<1x1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_gpu.multi_mma %lhs_vec_0_t, %rhs_vec_0_t, %iter {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
      } : vector<4x2x1x4xf16>, vector<4x2x1x4xf16> into vector<4x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0_lhs, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !mshared
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1_lhs, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !mshared

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_gpu.multi_mma %lhs_vec_2_t, %rhs_vec_2_t, %dot0 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
      } : vector<4x2x1x4xf16>, vector<4x2x1x4xf16> into vector<4x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<4x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp, vector<4x1x2x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x2x4xf16>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>

    %dot0 = iree_gpu.multi_mma %lhs_vec_0_t, %rhs_vec_0_t, %3 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<4x2x1x4xf16>, vector<4x2x1x4xf16> into vector<4x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp, vector<4x1x2x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x2x4xf16>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>

    %dot2 = iree_gpu.multi_mma %lhs_vec_2_t, %rhs_vec_2_t, %dot0 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>
    } : vector<4x2x1x4xf16>, vector<4x2x1x4xf16> into vector<4x4x1x4xf32>

    %tp = vector.transpose %dot2, [0, 2, 1, 3] : vector<4x4x1x4xf32> to vector<4x1x4x4xf32>
    %empty = tensor.empty() : tensor<1x4x1x4x4xf32>
    %4 = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<4x1x4x4xf32>, tensor<1x4x1x4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[0, %m_outer_id, %ids#3, %n_outer_id, %inner_id] [1, 4, 1, 4, 4] [1, 1, 1, 1, 1] : tensor<1x4x1x4x4xf32> into tensor<1x8x16x16x16xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %collapse = tensor.collapse_shape %1 [[0], [1, 2], [3, 4]] : tensor<1x8x16x16x16xf32> into tensor<1x128x256xf32>
  util.return %collapse : tensor<1x128x256xf32>
}

// ------------------------------------------------------------
// * Medium Pingpong f8 *
// ------------------------------------------------------------

util.func private @pingpong_medium_f8_expanded(%lhs_base: !mexp_in_ty_f8, %rhs_base: !in_ty_f8, %unused_acc: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ
  %lhs_shared_base = memref.alloc() : !mflat_shared_f8
  %rhs_shared_base = memref.alloc() : !flat_shared_f8

  %dim = tensor.dim %rhs_base, %c1 : !in_ty_f8
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim) : !mexp_in_ty_f8
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim) : !in_ty_f8

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<128, 8>] : !mflat_shared_f8
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<128, 8>] : !flat_shared_f8

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [128, 128] : !mflat_shared_f8 into !mshared_f8
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 128] : !flat_shared_f8 into !shared_f8

  %lhs_init = tensor.extract_slice %lhs [0, 0, 0] [1, 128, 128] [1, 1, 1] : !mexp_in_ty_f8 to !mexp_block_in_f8
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 128] [1, 1] : !in_ty_f8 to !block_in_f8

  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (128, 8) : index, index
    %vec = arith.muli %delin#1, %c16 : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [0, %delin#0, %vec] [1, 1, 16] [1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !mshared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c16 : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [8, 16, 4, 32] : !mshared_f8 into !mshared_exp_f8
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 32] : !shared_f8 into !shared_exp_f8

  %0 = tensor.empty() : tensor<1x8x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x8x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c8 : index
    %inner_id_acc = arith.muli %ids#2, %c4 : index
    %m_outer_id = arith.muli %ids#0, %c4 : index
    %n_outer_id = arith.muli %ids#1, %c4 : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 16 elements.
    %gko = arith.muli %wt#2, %c16 : index
    // RHS indexing. Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo : index
    %glb1 = arith.addi %glb0, %c8 : index
    %glb2 = arith.addi %glb1, %c8 : index
    %glb3 = arith.addi %glb2, %c8 : index
    // LHS indexing.
    %bpo_lhs = arith.muli %wt#0, %c16 : index
    %glb0_lhs = arith.addi %wt#1, %bpo_lhs : index
    %glb1_lhs = arith.addi %glb0_lhs, %c8 : index

    %2 = arith.constant dense<0.0> : vector<4x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c128 to %dim step %c128 iter_args(%iter = %2) -> vector<4x4x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_block = tensor.extract_slice %rhs [0, %i] [256, 128] [1, 1] : !in_ty_f8 to !block_in_f8
      %rhs_thread_0 = tensor.extract_slice %rhs_block [%glb0, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs_block [%glb1, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %rhs_thread_2 = tensor.extract_slice %rhs_block [%glb2, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %rhs_thread_3 = tensor.extract_slice %rhs_block [%glb3, %gko] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_block = tensor.extract_slice %lhs [0, 0, %i] [1, 128, 128] [1, 1, 1] : !mexp_in_ty_f8 to !mexp_block_in_f8
      %lhs_thread_0 = tensor.extract_slice %lhs_block [0, %glb0_lhs, %gko] [1, 1, 16] [1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs_block [0, %glb1_lhs, %gko] [1, 1, 16] [1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_gpu.multi_mma %lhs_vec_0_t, %rhs_vec_0_t, %iter {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
      } : vector<4x2x1x8xf8E4M3FNUZ>, vector<4x2x1x8xf8E4M3FNUZ> into vector<4x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0_lhs, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !mshared_f8
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1_lhs, %gko] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !mshared_f8

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_gpu.multi_mma %lhs_vec_2_t, %rhs_vec_2_t, %dot0 {
        indexing_maps = #contraction_accesses,
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
      } : vector<4x2x1x8xf8E4M3FNUZ>, vector<4x2x1x8xf8E4M3FNUZ> into vector<4x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<4x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>

    %dot0 = iree_gpu.multi_mma %lhs_vec_0_t, %rhs_vec_0_t, %3 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
    } : vector<4x2x1x8xf8E4M3FNUZ>, vector<4x2x1x8xf8E4M3FNUZ> into vector<4x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>

    %dot2 = iree_gpu.multi_mma %lhs_vec_2_t, %rhs_vec_2_t, %dot0 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
    } : vector<4x2x1x8xf8E4M3FNUZ>, vector<4x2x1x8xf8E4M3FNUZ> into vector<4x4x1x4xf32>

    %tp = vector.transpose %dot2, [0, 2, 1, 3] : vector<4x4x1x4xf32> to vector<4x1x4x4xf32>
    %empty = tensor.empty() : tensor<1x4x1x4x4xf32>
    %4 = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<4x1x4x4xf32>, tensor<1x4x1x4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[0, %m_outer_id, %ids#3, %n_outer_id, %inner_id_acc] [1, 4, 1, 4, 4] [1, 1, 1, 1, 1] : tensor<1x4x1x4x4xf32> into tensor<1x8x16x16x16xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %collapse = tensor.collapse_shape %1 [[0], [1, 2], [3, 4]] : tensor<1x8x16x16x16xf32> into tensor<1x128x256xf32>
  util.return %collapse : tensor<1x128x256xf32>
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
        transform.type_conversion.tensor.cast_shape_dynamic_dims ignore_dynamic_info
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
  transform.yield
}

transform.named_sequence @match_mmt_f16_f16_f32_impl(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %empty: tensor<?x?xf32>):
    %cst = arith.constant 0.0 : f32
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
@match_mmt_f16_f16_f32_large(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  transform.iree.match.dim_is_multiple_of  %lhs[0], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[1], 64 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 64 : !transform.any_value

  // M, N >= 1024, K >= 256
  transform.iree.match.dim_bounds %lhs[0], umin = 1024, none : !transform.any_value
  transform.iree.match.dim_bounds %rhs[0], umin = 1024, none : !transform.any_value
  transform.iree.match.dim_bounds %lhs[1], umin = 256, none : !transform.any_value

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
// * Expanded Pingpong Matmul For Dynamic M *
// ============================================================

transform.named_sequence @cast_and_call_expanded_pingpong_matmul(%mm: !transform.any_op {transform.readonly}) {
  %module = transform.util.get_nearest_symbol_table %mm : (!transform.any_op) -> !transform.any_op
  %func = transform.util.lookup_nearest_symbol_from_self @pingpong_large_expanded : !transform.any_op
  %ins = transform.get_operand %mm[all] : (!transform.any_op) -> !transform.any_value
  %out = transform.get_result %mm[all] : (!transform.any_op) -> !transform.any_value
  // Replace
  transform.util.cast_and_call inline_call %func(%ins) -> %out after %mm {
        transform.type_conversion.tensor.cast_shape_dynamic_dims ignore_dynamic_info
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
  transform.yield
}

transform.named_sequence @match_emmt_f16_f16_f32_impl(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?x?xf16>, %rhs: tensor<?x?xf16>, %empty: tensor<?x?x?xf32>):
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?x?xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %8 = arith.extf %in : f16 to f32
        %9 = arith.extf %in_0 : f16 to f32
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %acc, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_mmt_f16_f16_f32_large_expanded(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_emmt_f16_f16_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  transform.iree.match.dim_is_multiple_of  %lhs[1], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[2], 64 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 64 : !transform.any_value

  // M, N >= 1024, K >= 256

  // TODO: Kernel specialization is needed to apply this strategy selectively at
  // runtime. Additionally model exports don't specify lower bounds so it is
  // impossible to use this strategy with this check.
  // transform.iree.match.dim_bounds %lhs[0], umin = 4, none : !transform.any_value

  transform.iree.match.dim_bounds %rhs[0], umin = 1024, none : !transform.any_value
  transform.iree.match.dim_bounds %lhs[2], umin = 256, none : !transform.any_value

  // Lowering config for pingpong large. "cast_and_call_pingpong_matmul" refers
  // to the custom lowering strategy to use, which in this case replaces the
  // matmul with a call to the @pingpong_large implementation above.
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [1, 256, 256, 0],
      lowering_strategy = "cast_and_call_expanded_pingpong_matmul"}>,
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
transform.named_sequence @apply_expanded_pingpong_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  %module = transform.util.get_nearest_symbol_table %op : (!transform.any_op) -> !transform.any_op

  // Create and serialize a module with the needed symbols.
  %syms = transform.util.create_serialized_module {
    ^bb0(%m: !transform.any_op):
      transform.util.import_symbol @cast_and_call_expanded_pingpong_matmul into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.util.import_symbol @pingpong_large_expanded into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.annotate %m "transform.with_named_sequence" : !transform.any_op
  } -> !transform.any_param

  // Annotate the parent function with the serialized module.
  %func = transform.get_parent_op %op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  transform.annotate %func "iree_codegen_external_symbols" = %syms : !transform.any_op, !transform.any_param
  transform.yield
}

// ============================================================
// * Expanded F8 Pingpong Matmul For Dynamic M *
// ============================================================

transform.named_sequence @cast_and_call_expanded_f8_pingpong_matmul(%mm: !transform.any_op {transform.readonly}) {
  %module = transform.util.get_nearest_symbol_table %mm : (!transform.any_op) -> !transform.any_op
  %func = transform.util.lookup_nearest_symbol_from_self @pingpong_large_f8_expanded : !transform.any_op
  %ins = transform.get_operand %mm[all] : (!transform.any_op) -> !transform.any_value
  %out = transform.get_result %mm[all] : (!transform.any_op) -> !transform.any_value
  // Replace
  transform.util.cast_and_call inline_call %func(%ins) -> %out after %mm {
        transform.type_conversion.tensor.cast_shape_dynamic_dims ignore_dynamic_info
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
  transform.yield
}

transform.named_sequence @match_emmt_f8_f8_f32_impl(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?x?xf8E4M3FNUZ>, %rhs: tensor<?x?xf8E4M3FNUZ>, %empty: tensor<?x?x?xf32>):
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x?x?xf8E4M3FNUZ>, tensor<?x?xf8E4M3FNUZ>) outs(%out : tensor<?x?x?xf32>) {
      ^bb0(%in: f8E4M3FNUZ, %in_0: f8E4M3FNUZ, %acc: f32):
        %8 = arith.extf %in : f8E4M3FNUZ to f32
        %9 = arith.extf %in_0 : f8E4M3FNUZ to f32
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %acc, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_mmt_f8_f8_f32_large_expanded(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_emmt_f8_f8_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // M % 256 == 0, K % 128 == 0, N % 256 == 0
  transform.iree.match.dim_is_multiple_of  %lhs[1], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[2], 128 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 128 : !transform.any_value

  // M, N >= 1024, K >= 512

  // TODO: Kernel specialization is needed to apply this strategy selectively at
  // runtime. Additionally model exports don't specify lower bounds so it is
  // impossible to use this strategy with this check.
  // transform.iree.match.dim_bounds %lhs[0], umin = 4, none : !transform.any_value

  transform.iree.match.dim_bounds %rhs[0], umin = 1024, none : !transform.any_value
  transform.iree.match.dim_bounds %lhs[2], umin = 512, none : !transform.any_value

  // Lowering config for pingpong large. "cast_and_call_pingpong_matmul" refers
  // to the custom lowering strategy to use, which in this case replaces the
  // matmul with a call to the @pingpong_large implementation above.
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [1, 256, 256, 0],
      lowering_strategy = "cast_and_call_expanded_f8_pingpong_matmul"}>,
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
transform.named_sequence @apply_expanded_f8_pingpong_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  %module = transform.util.get_nearest_symbol_table %op : (!transform.any_op) -> !transform.any_op

  // Create and serialize a module with the needed symbols.
  %syms = transform.util.create_serialized_module {
    ^bb0(%m: !transform.any_op):
      transform.util.import_symbol @cast_and_call_expanded_f8_pingpong_matmul into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.util.import_symbol @pingpong_large_f8_expanded into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.annotate %m "transform.with_named_sequence" : !transform.any_op
  } -> !transform.any_param

  // Annotate the parent function with the serialized module.
  %func = transform.get_parent_op %op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  transform.annotate %func "iree_codegen_external_symbols" = %syms : !transform.any_op, !transform.any_param
  transform.yield
}

// ============================================================
// * Medium Pingpong *
// ============================================================

transform.named_sequence @cast_and_call_expanded_medium_pingpong_matmul(%mm: !transform.any_op {transform.readonly}) {
  %module = transform.util.get_nearest_symbol_table %mm : (!transform.any_op) -> !transform.any_op
  %func = transform.util.lookup_nearest_symbol_from_self @pingpong_medium_expanded : !transform.any_op
  %ins = transform.get_operand %mm[all] : (!transform.any_op) -> !transform.any_value
  %out = transform.get_result %mm[all] : (!transform.any_op) -> !transform.any_value
  // Replace
  transform.util.cast_and_call inline_call %func(%ins) -> %out after %mm {
        transform.type_conversion.tensor.cast_shape_dynamic_dims ignore_dynamic_info
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
  transform.yield
}

transform.named_sequence @match_medium_emmt_f16_f16_f32_impl(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?x?xf16>, %rhs: tensor<?x?xf16>, %empty: tensor<?x?x?xf32>):
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?x?xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %8 = arith.extf %in : f16 to f32
        %9 = arith.extf %in_0 : f16 to f32
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %acc, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_mmt_f16_f16_f32_medium_expanded(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_medium_emmt_f16_f16_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // M % 128 == 0, K % 64 == 0, N % 256 == 0
  transform.iree.match.dim_is_multiple_of  %lhs[1], 128 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[2], 64 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 64 : !transform.any_value

  // M >= 512, N >= 1024, K >= 256

  // TODO: Kernel specialization is needed to apply this strategy selectively at
  // runtime. Additionally model exports don't specify lower bounds so it is
  // impossible to use this strategy with this check.
  // transform.iree.match.dim_bounds %lhs[0], umin = 4, none : !transform.any_value

  transform.iree.match.dim_bounds %rhs[0], umin = 1024, none : !transform.any_value
  transform.iree.match.dim_bounds %lhs[2], umin = 256, none : !transform.any_value

  // Lowering config for pingpong large. "cast_and_call_pingpong_matmul" refers
  // to the custom lowering strategy to use, which in this case replaces the
  // matmul with a call to the @pingpong_large implementation above.
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [1, 128, 256, 0],
      lowering_strategy = "cast_and_call_expanded_medium_pingpong_matmul"}>,
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

/// Applies the op config for pingpong_medium_expanded. This requires importing
/// external symbols needed for the custom lowering (in this case inline +
/// replace).
transform.named_sequence @apply_expanded_medium_pingpong_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  %module = transform.util.get_nearest_symbol_table %op : (!transform.any_op) -> !transform.any_op

  // Create and serialize a module with the needed symbols.
  %syms = transform.util.create_serialized_module {
    ^bb0(%m: !transform.any_op):
      transform.util.import_symbol @cast_and_call_expanded_medium_pingpong_matmul into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.util.import_symbol @pingpong_medium_expanded into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.annotate %m "transform.with_named_sequence" : !transform.any_op
  } -> !transform.any_param

  // Annotate the parent function with the serialized module.
  %func = transform.get_parent_op %op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  transform.annotate %func "iree_codegen_external_symbols" = %syms : !transform.any_op, !transform.any_param
  transform.yield
}

// ============================================================
// * Medium F8 Pingpong *
// ============================================================

transform.named_sequence @cast_and_call_expanded_f8_medium_pingpong_matmul(%mm: !transform.any_op {transform.readonly}) {
  %module = transform.util.get_nearest_symbol_table %mm : (!transform.any_op) -> !transform.any_op
  %func = transform.util.lookup_nearest_symbol_from_self @pingpong_medium_f8_expanded : !transform.any_op
  %ins = transform.get_operand %mm[all] : (!transform.any_op) -> !transform.any_value
  %out = transform.get_result %mm[all] : (!transform.any_op) -> !transform.any_value
  // Replace
  transform.util.cast_and_call inline_call %func(%ins) -> %out after %mm {
        transform.type_conversion.tensor.cast_shape_dynamic_dims ignore_dynamic_info
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
  transform.yield
}

transform.named_sequence @match_medium_emmt_f8_f8_f32_impl(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?x?xf8E4M3FNUZ>, %rhs: tensor<?x?xf8E4M3FNUZ>, %empty: tensor<?x?x?xf32>):
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x?x?xf8E4M3FNUZ>, tensor<?x?xf8E4M3FNUZ>) outs(%out : tensor<?x?x?xf32>) {
      ^bb0(%in: f8E4M3FNUZ, %in_0: f8E4M3FNUZ, %acc: f32):
        %8 = arith.extf %in : f8E4M3FNUZ to f32
        %9 = arith.extf %in_0 : f8E4M3FNUZ to f32
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %acc, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_mmt_f8_f8_f32_medium_expanded(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_medium_emmt_f8_f8_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // M % 128 == 0, K % 128 == 0, N % 256 == 0
  transform.iree.match.dim_is_multiple_of  %lhs[1], 128 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[2], 128 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 256 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 128 : !transform.any_value

  // N >= 1024, K >= 512
  transform.iree.match.dim_bounds %rhs[0], umin = 1024, none : !transform.any_value
  transform.iree.match.dim_bounds %lhs[2], umin = 512, none : !transform.any_value

  // Lowering config for pingpong large. "cast_and_call_pingpong_matmul" refers
  // to the custom lowering strategy to use, which in this case replaces the
  // matmul with a call to the @pingpong_large implementation above.
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [1, 128, 256, 0],
      lowering_strategy = "cast_and_call_expanded_f8_medium_pingpong_matmul"}>,
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

/// Applies the op config for pingpong_medium_f8_expanded. This requires importing
/// external symbols needed for the custom lowering (in this case inline +
/// replace).
transform.named_sequence @apply_expanded_f8_medium_pingpong_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  %module = transform.util.get_nearest_symbol_table %op : (!transform.any_op) -> !transform.any_op

  // Create and serialize a module with the needed symbols.
  %syms = transform.util.create_serialized_module {
    ^bb0(%m: !transform.any_op):
      transform.util.import_symbol @cast_and_call_expanded_f8_medium_pingpong_matmul into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.util.import_symbol @pingpong_medium_f8_expanded into %m if undefined : (!transform.any_op) -> !transform.any_op
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

transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
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
  transform.iree.match.has_no_lowering_config %attention : !transform.any_op

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
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

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
    // Match pingpong variants.
    @match_mmt_f16_f16_f32_large_expanded -> @apply_expanded_pingpong_op_config,
    @match_mmt_f8_f8_f32_large_expanded -> @apply_expanded_f8_pingpong_op_config,
    @match_mmt_f16_f16_f32_large -> @apply_pingpong_op_config,

    // Medium pingpong variants are lower priority.
    @match_mmt_f16_f16_f32_medium_expanded -> @apply_expanded_medium_pingpong_op_config,
    @match_mmt_f8_f8_f32_medium_expanded -> @apply_expanded_f8_medium_pingpong_op_config,

    // Expected speedup: 1.22x.
    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config,
    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}

}
