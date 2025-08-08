//  RUN: iree-opt %s

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

util.func @pingpong_medium_f8_expanded(%lhs_base: !mexp_in_ty_f8, %rhs_base: !in_ty_f8, %unused_acc: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> {
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
    %vec = arith.muli %delin#1, %c16 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [0, %delin#0, %vec] [1, 1, 16] [1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !mshared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c16 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 16] [1, 1] : !block_in_f8 to tensor<1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x16xf8E4M3FNUZ>, vector<1x16xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x16xf8E4M3FNUZ>, !shared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [8, 16, 4, 32] : !mshared_f8 into !mshared_exp_f8
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 32] : !shared_f8 into !shared_exp_f8

  %0 = tensor.empty() : tensor<1x8x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x8x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c8 overflow<nsw, nuw> : index
    %inner_id_acc = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
    %m_outer_id = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
    %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 16 elements.
    %gko = arith.muli %wt#2, %c16 overflow<nsw, nuw> : index
    // RHS indexing. Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 overflow<nsw, nuw> : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo overflow<nsw, nuw> : index
    %glb1 = arith.addi %glb0, %c8 overflow<nsw, nuw> : index
    %glb2 = arith.addi %glb1, %c8 overflow<nsw, nuw> : index
    %glb3 = arith.addi %glb2, %c8 overflow<nsw, nuw> : index
    // LHS indexing.
    %bpo_lhs = arith.muli %wt#0, %c16 overflow<nsw, nuw> : index
    %glb0_lhs = arith.addi %wt#1, %bpo_lhs overflow<nsw, nuw> : index
    %glb1_lhs = arith.addi %glb0_lhs, %c8 overflow<nsw, nuw> : index

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

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
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

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
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

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
    } : vector<4x2x1x8xf8E4M3FNUZ>, vector<4x2x1x8xf8E4M3FNUZ> into vector<4x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp_f8, vector<4x1x2x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x8xf8E4M3FNUZ> to vector<4x2x1x8xf8E4M3FNUZ>

    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
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
