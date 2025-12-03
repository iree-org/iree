//  RUN: iree-opt %s

!acc_base_ty = tensor<1x1x2x4x8x4x4x16x4xf32>
!lhs_base_ty = tensor<1x?x2x8x4x16x8xf8E4M3FNUZ>
!lhs_expand_ty = tensor<1x?x4x2x8x4x4x2x2x8xf8E4M3FNUZ>
!rhs_base_ty = tensor<1x?x4x4x4x16x8xf8E4M3FNUZ>
!rhs_expand_ty = tensor<1x?x4x4x4x4x8x2x8xf8E4M3FNUZ>
!in_ty = tensor<?x4x16x32x16xf8E4M3FNUZ>
!shared_ty = memref<4x16x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

!m_acc_base_ty = tensor<1x1x8x8x2x4x16x4xf32>
!m_lhs_base_ty = tensor<1x?x8x4x16x2x8xf8E4M3FNUZ>
!m_lhs_expand_ty = tensor<1x?x2x8x4x4x4x2x8xf8E4M3FNUZ>
!m_rhs_base_ty = tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>
!m_rhs_expand_ty = tensor<1x?x2x8x2x4x16x2x8xf8E4M3FNUZ>
!m_lhs_ty = tensor<?x2x8x64x16xf8E4M3FNUZ>
!m_rhs_ty = tensor<?x2x16x64x16xf8E4M3FNUZ>
!m_lhs_shared_ty = memref<2x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!m_rhs_shared_ty = memref<2x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func @pingpong_dt_large_f8E4M3FNUZ(%lhs_base: !lhs_base_ty, %rhs_base: !rhs_base_ty, %unused_acc: !acc_base_ty) -> !acc_base_ty attributes {
  ukernel_info = #rocm.ukernel_info<
    match = {
      types = [f8E4M3FNUZ, f8E4M3FNUZ, f32],
      iteration_sizes_constraints = [
        #rocm.ukernel_interation_size_constraint<
          index = 0,
          size_min = 64
        >,
        #rocm.ukernel_interation_size_constraint<
          index = 1,
          size_min = 2048,
          size_max = 8192
        >
      ]
    },
    // Benefit larger than the default 0 means we prefer this "large" kernel when it matches.
    benefit = 1,
    mma = #iree_gpu.data_tiled_mma_layout<
      intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ,
      intrinsics_m = 8,
      subgroups_m = 2,
      intrinsics_n = 4,
      subgroups_n = 4,
      intrinsics_k = 1, operands_interleaving_intrinsics_k = [0, 1]
    >
  >
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : !rhs_base_ty
  %nDim = arith.divui %dim, %c4 : index

  %lhs_expand = tensor.expand_shape %lhs_base [[0], [1, 2], [3], [4], [5], [6, 7, 8], [9]] output_shape [1, %nDim, 4, 2, 8, 4, 4, 2, 2, 8] : !lhs_base_ty into !lhs_expand_ty
  %rhs_expand = tensor.expand_shape %rhs_base [[0], [1, 2], [3], [4], [5], [6, 7], [8]] output_shape [1, %nDim, 4, 4, 4, 4, 8, 2, 8] : !rhs_base_ty into !rhs_expand_ty

  %lhs = tensor.collapse_shape %lhs_expand [[0, 1], [2], [3, 4], [5, 6, 7], [8, 9]] : !lhs_expand_ty into !in_ty
  %rhs = tensor.collapse_shape %rhs_expand [[0, 1], [2], [3, 4], [5, 6], [7, 8]] : !rhs_expand_ty into !in_ty

  %lhs_shared = memref.alloc() : !shared_ty
  %rhs_shared = memref.alloc() : !shared_ty

  scf.forall (%id) in (2048) {
    %delin:3 = affine.delinearize_index %id into (4, 16, 32) : index, index, index
    %inner = arith.muli %delin#2, %c2 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1]  : !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    %lhs_vec_local_t = vector.shape_cast %lhs_vec_local : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local_t, %lhs_shared[%delin#0, %delin#1, %inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:3 = affine.delinearize_index %id into (4, 16, 32) : index, index, index
    %inner = arith.muli %delin#2, %c2 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    %rhs_vec_local_t = vector.shape_cast %rhs_vec_local : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local_t, %rhs_shared[%delin#0, %delin#1, %inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : !acc_base_ty
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> !acc_base_ty {
    %ids:3 = affine.delinearize_index %id into (2, 4, 64) : index, index, index
    %threads:2 = affine.delinearize_index %ids#2 into (4, 16) : index, index

    %m_outer = arith.muli %ids#0, %c8 overflow<nsw, nuw> : index
    %n_outer = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index

    %glb:2 = affine.delinearize_index %id into (16, 32) : index, index
    %glb_inner = arith.muli %glb#1, %c2 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }

    %3 = scf.for %i = %c1 to %nDim step %c1 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {
      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %c0, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] :  !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0_t = vector.shape_cast %lhs_vec_local_0 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %c1, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_1_t = vector.shape_cast %lhs_vec_local_1 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_2 = tensor.extract_slice %lhs [%i, %c2, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] :  !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_2_t = vector.shape_cast %lhs_vec_local_2 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_3 = tensor.extract_slice %lhs [%i, %c3, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_3_t = vector.shape_cast %lhs_vec_local_3 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>

      // Local loads.
      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %c0, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] :  !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_0_t = vector.shape_cast %rhs_vec_local_0 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %c1, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_1_t = vector.shape_cast %rhs_vec_local_1 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_2 = tensor.extract_slice %rhs [%i, %c2, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] :  !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_2_t = vector.shape_cast %rhs_vec_local_2 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_3 = tensor.extract_slice %rhs [%i, %c3, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_3_t = vector.shape_cast %rhs_vec_local_3 : vector<1x1x1x16xf8E4M3FNUZ> to vector<1x1x2x8xf8E4M3FNUZ>

      // Local loads.
      %lhs_vec_1 = vector.transfer_read %lhs_shared[%c1, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_1 = vector.transfer_read %rhs_shared[%c1, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Local loads.
      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c2, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c2, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      // Local loads.
      %lhs_vec_3 = vector.transfer_read %lhs_shared[%c3, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_3 = vector.transfer_read %rhs_shared[%c3, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_3_t = vector.shape_cast %lhs_vec_3 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_3_t = vector.shape_cast %rhs_vec_3 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot1) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Local stores of lhs and rhs.
      vector.transfer_write %rhs_vec_local_0_t, %rhs_shared [%c0, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
      vector.transfer_write %rhs_vec_local_1_t, %rhs_shared [%c1, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
      vector.transfer_write %rhs_vec_local_2_t, %rhs_shared [%c2, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
      vector.transfer_write %rhs_vec_local_3_t, %rhs_shared [%c3, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty

      vector.transfer_write %lhs_vec_local_0_t, %lhs_shared [%c0, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
      vector.transfer_write %lhs_vec_local_1_t, %lhs_shared [%c1, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
      vector.transfer_write %lhs_vec_local_2_t, %lhs_shared [%c2, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty
      vector.transfer_write %lhs_vec_local_3_t, %lhs_shared [%c3, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x8xf8E4M3FNUZ>, !shared_ty

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3_t, %rhs_vec_3_t) outs(%dot2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
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
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %lhs_vec_1 = vector.transfer_read %lhs_shared[%c1, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_1 = vector.transfer_read %rhs_shared[%c1, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c2, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c2, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot1) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %lhs_vec_3 = vector.transfer_read %lhs_shared[%c3, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_3 = vector.transfer_read %rhs_shared[%c3, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_3_t = vector.shape_cast %lhs_vec_3 : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_3_t = vector.shape_cast %rhs_vec_3 : vector<1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3_t, %rhs_vec_3_t) outs(%dot2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x1x8x4x1x1x4xf32>
    %cast = vector.shape_cast %dot3 : vector<8x4x1x4xf32> to vector<1x1x1x1x8x4x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true, true]} : vector<1x1x1x1x8x4x1x1x4xf32>, tensor<1x1x1x1x8x4x1x1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 1, 8, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x4x1x1x4xf32> into !acc_base_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : !acc_base_ty
}

util.func private @pingpong_dt_medium_f8E4M3FNUZ(%lhs_base: !m_lhs_base_ty, %rhs_base: !m_rhs_base_ty, %unused_acc: !m_acc_base_ty) -> !m_acc_base_ty attributes {
  ukernel_info = #rocm.ukernel_info<
    match = {
      types = [f8E4M3FNUZ, f8E4M3FNUZ, f32],
      iteration_sizes_constraints = [
        #rocm.ukernel_interation_size_constraint<
          index = 0,
          size_min = 32
        >
      ]
    },
    mma = #iree_gpu.data_tiled_mma_layout<
      intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ,
      intrinsics_m = 8,
      subgroups_m = 1,
      intrinsics_n = 2,
      subgroups_n = 8,
      intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]
    >
  >
} {
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
  %c4096 = arith.constant 4096 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : !m_rhs_base_ty
  %nDim = arith.divui %dim, %c2 : index

  %lhs_expand = tensor.expand_shape %lhs_base [[0], [1, 2], [3], [4], [5, 6], [7], [8]] output_shape [1, %nDim, 2, 8, 4, 4, 4, 2, 8] : !m_lhs_base_ty into !m_lhs_expand_ty
  %rhs_expand = tensor.expand_shape %rhs_base [[0], [1, 2], [3], [4], [5], [6], [7], [8]] output_shape [1, %nDim, 2, 8, 2, 4, 16, 2, 8] : !m_rhs_base_ty into !m_rhs_expand_ty

  %lhs = tensor.collapse_shape %lhs_expand [[0, 1], [2], [3], [4, 5, 6], [7, 8]] : !m_lhs_expand_ty into !m_lhs_ty
  %rhs = tensor.collapse_shape %rhs_expand [[0, 1], [2], [3, 4], [5, 6], [7, 8]] : !m_rhs_expand_ty into !m_rhs_ty

  %lhs_shared = memref.alloc() : !m_lhs_shared_ty
  %rhs_shared = memref.alloc() : !m_rhs_shared_ty

  scf.forall (%id) in (1024) {
    %delin:3 = affine.delinearize_index %id into (2, 8, 64) : index, index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1]  : !m_lhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %delin#1, %delin#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_lhs_shared_ty
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:3 = affine.delinearize_index %id into (2, 16, 64) : index, index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !m_rhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %delin#1, %delin#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_rhs_shared_ty
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : !m_acc_base_ty
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> !m_acc_base_ty {
    %ids:3 = affine.delinearize_index %id into (1, 8, 64) : index, index, index
    %threads:2 = affine.delinearize_index %ids#2 into (4, 16) : index, index

    %glb0_rhs = arith.muli %ids#1, %c2 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }

    %3 = scf.for %i = %c1 to %nDim step %c1 iter_args(%iter = %2) -> vector<8x2x1x4xf32> {
      // Local loads of lhs.
      %lhs_vec = vector.transfer_read %lhs_shared[%c0, %ids#0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !m_lhs_shared_ty, vector<2x8x1x16xf8E4M3FNUZ>
      %lhs_vec_0 = vector.extract_strided_slice %lhs_vec {offsets = [0, 0, 0, 0], sizes = [1, 8, 1, 16], strides = [1, 1, 1, 1]} : vector<2x8x1x16xf8E4M3FNUZ> to vector<1x8x1x16xf8E4M3FNUZ>
      %lhs_vec_2 = vector.extract_strided_slice %lhs_vec {offsets = [1, 0, 0, 0], sizes = [1, 8, 1, 16], strides = [1, 1, 1, 1]} : vector<2x8x1x16xf8E4M3FNUZ> to vector<1x8x1x16xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x8x1x16xf8E4M3FNUZ> to vector<8x2x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x8x1x16xf8E4M3FNUZ> to vector<8x2x1x8xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %c0, %ids#1, %ids#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !m_lhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %c1, %ids#1, %ids#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !m_lhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Local loads of rhs.
      %rhs_vec = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !m_rhs_shared_ty, vector<2x2x1x16xf8E4M3FNUZ>
      %rhs_vec_0 = vector.extract_strided_slice %rhs_vec {offsets = [0, 0, 0, 0], sizes = [1, 2, 1, 16], strides = [1, 1, 1, 1]} : vector<2x2x1x16xf8E4M3FNUZ> to vector<1x2x1x16xf8E4M3FNUZ>
      %rhs_vec_2 = vector.extract_strided_slice %rhs_vec {offsets = [1, 0, 0, 0], sizes = [1, 2, 1, 16], strides = [1, 1, 1, 1]} : vector<2x2x1x16xf8E4M3FNUZ> to vector<1x2x1x16xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x2x1x16xf8E4M3FNUZ> to vector<2x2x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x2x1x16xf8E4M3FNUZ> to vector<2x2x1x8xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %c0, %glb0_rhs, %ids#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !m_rhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %c0, %glb1_rhs, %ids#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !m_rhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_thread_2 = tensor.extract_slice %rhs [%i, %c1, %glb0_rhs, %ids#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !m_rhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_thread_3 = tensor.extract_slice %rhs [%i, %c1, %glb1_rhs, %ids#2, %c0] [1, 1, 1, 1, 16] [1, 1, 1, 1, 1] : !m_rhs_ty to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x2x1x8xf8E4M3FNUZ>, vector<2x2x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Local stores of lhs and rhs.
      vector.transfer_write %rhs_vec_local_0, %rhs_shared[%c0, %glb0_rhs, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_rhs_shared_ty
      vector.transfer_write %rhs_vec_local_1, %rhs_shared[%c0, %glb1_rhs, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_rhs_shared_ty
      vector.transfer_write %rhs_vec_local_2, %rhs_shared[%c1, %glb0_rhs, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_rhs_shared_ty
      vector.transfer_write %rhs_vec_local_3, %rhs_shared[%c1, %glb1_rhs, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_rhs_shared_ty

      vector.transfer_write %lhs_vec_local_0, %lhs_shared[%c0, %ids#1, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_lhs_shared_ty
      vector.transfer_write %lhs_vec_local_1, %lhs_shared[%c1, %ids#1, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, !m_lhs_shared_ty

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x2x1x8xf8E4M3FNUZ>, vector<2x2x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec = vector.transfer_read %lhs_shared[%c0, %ids#0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !m_lhs_shared_ty, vector<2x8x1x16xf8E4M3FNUZ>
    %lhs_vec_0 = vector.extract_strided_slice %lhs_vec {offsets = [0, 0, 0, 0], sizes = [1, 8, 1, 16], strides = [1, 1, 1, 1]} : vector<2x8x1x16xf8E4M3FNUZ> to vector<1x8x1x16xf8E4M3FNUZ>
    %lhs_vec_2 = vector.extract_strided_slice %lhs_vec {offsets = [1, 0, 0, 0], sizes = [1, 8, 1, 16], strides = [1, 1, 1, 1]} : vector<2x8x1x16xf8E4M3FNUZ> to vector<1x8x1x16xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x8x1x16xf8E4M3FNUZ> to vector<8x2x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x8x1x16xf8E4M3FNUZ> to vector<8x2x1x8xf8E4M3FNUZ>

    %rhs_vec = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !m_rhs_shared_ty, vector<2x2x1x16xf8E4M3FNUZ>
    %rhs_vec_0 = vector.extract_strided_slice %rhs_vec {offsets = [0, 0, 0, 0], sizes = [1, 2, 1, 16], strides = [1, 1, 1, 1]} : vector<2x2x1x16xf8E4M3FNUZ> to vector<1x2x1x16xf8E4M3FNUZ>
    %rhs_vec_2 = vector.extract_strided_slice %rhs_vec {offsets = [1, 0, 0, 0], sizes = [1, 2, 1, 16], strides = [1, 1, 1, 1]} : vector<2x2x1x16xf8E4M3FNUZ> to vector<1x2x1x16xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x2x1x16xf8E4M3FNUZ> to vector<2x2x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x2x1x16xf8E4M3FNUZ> to vector<2x2x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x2x1x8xf8E4M3FNUZ>, vector<2x2x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<8x2x1x8xf8E4M3FNUZ>, vector<2x2x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into !m_acc_base_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : !m_acc_base_ty
}
