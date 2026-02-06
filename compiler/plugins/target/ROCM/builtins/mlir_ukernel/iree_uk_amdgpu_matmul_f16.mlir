//  RUN: iree-opt %s

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

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func private @pingpong_large_f16(%lhs_base: !in_ty, %rhs_base: !in_ty, %unused_acc: tensor<256x256xf32>) -> tensor<256x256xf32> {
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
  %dim_bytes = arith.muli %dim, %c2 overflow<nsw, nuw>: index
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim_bytes) : !in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim_bytes) : !in_ty

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared

  %lhs_init = tensor.extract_slice %lhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in

  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp

  %0 = tensor.empty() : tensor<16x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<16x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
    %m_outer_id = arith.muli %ids#0, %c8 overflow<nsw, nuw> : index
    %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 8 elements.
    %gko = arith.muli %wt#2, %c8 overflow<nsw, nuw> : index
    // Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 overflow<nsw, nuw> : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo overflow<nsw, nuw> : index
    %glb1 = arith.addi %glb0, %c8 overflow<nsw, nuw> : index
    %glb2 = arith.addi %glb1, %c8 overflow<nsw, nuw> : index
    %glb3 = arith.addi %glb2, %c8 overflow<nsw, nuw> : index

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

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
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

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1, %rhs_vec_1) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot1) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_2, %lhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_3, %lhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3, %rhs_vec_3) outs(%dot2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0

      scf.yield %dot3 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1, %rhs_vec_1) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot1) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3, %rhs_vec_3) outs(%dot2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
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

util.func private @pingpong_medium_f16_expanded(%lhs_base: !mexp_in_ty, %rhs_base: !in_ty, %unused_acc: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> {
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
  %dim_bytes = arith.muli %dim, %c2 overflow<nsw, nuw>: index
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim_bytes) : !mexp_in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim_bytes) : !in_ty

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !mflat_shared
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [128, 64] : !mflat_shared into !mshared
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared

  %lhs_init = tensor.extract_slice %lhs [0, 0, 0] [1, 128, 64] [1, 1, 1] : !mexp_in_ty to !mexp_block_in
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in

  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (128, 8) : index, index
    %vec = arith.muli %delin#1, %c8 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [0, %delin#0, %vec] [1, 1, 8] [1, 1, 1] : !mexp_block_in to tensor<1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !mshared
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [8, 16, 4, 16] : !mshared into !mshared_exp
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp

  %0 = tensor.empty() : tensor<1x8x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x8x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
    %m_outer_id = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
    %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 8 elements.
    %gko = arith.muli %wt#2, %c8 overflow<nsw, nuw> : index
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

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x2x1x4xf16>, vector<4x2x1x4xf16> into vector<4x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0_lhs, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !mshared
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1_lhs, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !mshared

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x2x1x4xf16>, vector<4x2x1x4xf16> into vector<4x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
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

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<4x2x1x4xf16>, vector<4x2x1x4xf16> into vector<4x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !mshared_exp, vector<4x1x2x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x2x4xf16>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [0, 2, 1, 3] : vector<4x1x2x4xf16> to vector<4x2x1x4xf16>

    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
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

util.func private @pingpong_large_f16_expanded(%lhs_base: !exp_in_ty, %rhs_base: !in_ty, %unused_acc: tensor<1x256x256xf32>) -> tensor<1x256x256xf32> {
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
  %dim_bytes = arith.muli %dim, %c2 overflow<nsw, nuw>: index
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim_bytes) : !exp_in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim_bytes) : !in_ty

  %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared
  %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<64, 4>] : !flat_shared

  %lhs_shared = memref.expand_shape %lhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared
  %rhs_shared = memref.expand_shape %rhs_shared_swizzle [[0, 1]] output_shape [256, 64] : !flat_shared into !shared

  %lhs_init = tensor.extract_slice %lhs [0, 0, 0] [1, 256, 64] [1, 1, 1] : !exp_in_ty to !exp_block_in
  %rhs_init = tensor.extract_slice %rhs [0, 0] [256, 64] [1, 1] : !in_ty to !block_in

  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs_init [0, %delin#0, %vec] [1, 1, 8] [1, 1, 1] : !exp_block_in to tensor<1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1x8xf16>, vector<1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (256, 8) : index, index
    %vec = arith.muli %delin#1, %c8 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %vec] [1, 8] [1, 1] : !block_in to tensor<1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x8xf16>, vector<1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %vec] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp
  %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 16] : !shared into !shared_exp

  %0 = tensor.empty() : tensor<1x16x16x16x16xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x16x16x16x16xf32> {
    %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %inner_id = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
    %m_outer_id = arith.muli %ids#0, %c8 overflow<nsw, nuw> : index
    %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index
    %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // Inner 64 loads 8 threads x 8 elements.
    %gko = arith.muli %wt#2, %c8 overflow<nsw, nuw> : index
    // Each subgroup loads 32 contiguous rows out of 256.
    %bpo = arith.muli %wt#0, %c32 overflow<nsw, nuw> : index
    // Base index is remaining outer 8 lanes + subgroup base.
    %glb0 = arith.addi %wt#1, %bpo overflow<nsw, nuw> : index
    %glb1 = arith.addi %glb0, %c8 overflow<nsw, nuw> : index
    %glb2 = arith.addi %glb1, %c8 overflow<nsw, nuw> : index
    %glb3 = arith.addi %glb2, %c8 overflow<nsw, nuw> : index

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

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
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

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1, %rhs_vec_1) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
      %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot1) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0

      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_2, %lhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %lhs_vec_local_3, %lhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb0, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb1, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb2, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb3, %gko] {in_bounds = [true, true]} : vector<1x8xf16>, !shared

      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3, %rhs_vec_3) outs(%dot2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      rocdl.sched.barrier 0

      scf.yield %dot3 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c0, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_1 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_1 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c1, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1, %rhs_vec_1) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_2 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c2, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot1) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>
    %lhs_vec_3 = vector.transfer_read %lhs_shared_expand[%m_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<8x1x1x4xf16>
    %rhs_vec_3 = vector.transfer_read %rhs_shared_expand[%n_outer_id, %ids#3, %c3, %inner_id], %cst {in_bounds = [true, true, true, true]} : !shared_exp, vector<4x1x1x4xf16>
    %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3, %rhs_vec_3) outs(%dot2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
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
