//  RUN: iree-opt %s

!acc_base_ty = tensor<1x1x2x4x8x4x4x16x4xf32>
!lhs_base_ty = tensor<1x?x2x8x4x16x4xf16>
!lhs_expand_ty = tensor<1x?x4x2x8x4x4x2x2x4xf16>
!rhs_base_ty = tensor<1x?x4x4x4x16x4xf16>
!rhs_expand_ty = tensor<1x?x4x4x4x4x8x2x4xf16>
!in_ty = tensor<?x4x16x32x8xf16>
!shared_ty = memref<4x16x64x4xf16, #gpu.address_space<workgroup>>

!out_sref = !pcf.sref<1x1x2x4x8x4x4x16x4xf32, #iree_gpu.subgroup_scope>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func @pingpong_dt_large_f16(%lhs_base: !lhs_base_ty, %rhs_base: !rhs_base_ty, %unused_acc: !acc_base_ty) -> !acc_base_ty attributes {
  ukernel_info = #rocm.ukernel_info<
    match = {
      archs = ["gfx942"],
      types = [f16, f16, f32],
      iteration_sizes_constraints = [
        #rocm.ukernel_interation_size_constraint<
          index = 0,
          size_min = 512,
          size_div = 64
        >,
        #rocm.ukernel_interation_size_constraint<
          index = 1,
          size_min = 32832,
          size_div = 64
        >,
        #rocm.ukernel_interation_size_constraint<
          index = 2,
          size_min = 512,
          size_div = 64
        >
      ]
    },
    mma = #iree_gpu.data_tiled_mma_layout<
      intrinsic = MFMA_F32_16x16x16_F16,
      intrinsics_m = 8,
      subgroups_m = 2,
      intrinsics_n = 4,
      subgroups_n = 4
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
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : !rhs_base_ty
  %nDim =  arith.divui %dim, %c4 : index

  %lhs_expand = tensor.expand_shape %lhs_base [[0], [1, 2], [3], [4], [5], [6, 7, 8], [9]] output_shape [1, %nDim, 4, 2, 8, 4, 4, 2, 2, 4] : !lhs_base_ty into !lhs_expand_ty
  %rhs_expand = tensor.expand_shape %rhs_base [[0], [1, 2], [3], [4], [5], [6, 7], [8]] output_shape [1, %nDim, 4, 4, 4, 4, 8, 2, 4] : !rhs_base_ty into !rhs_expand_ty

  %lhs = tensor.collapse_shape %lhs_expand [[0, 1], [2], [3, 4], [5, 6, 7], [8, 9]] : !lhs_expand_ty into !in_ty
  %rhs = tensor.collapse_shape %rhs_expand [[0, 1], [2], [3, 4], [5, 6], [7, 8]] : !rhs_expand_ty into !in_ty

  %lhs_shared = memref.alloc() : !shared_ty
  %rhs_shared = memref.alloc() : !shared_ty

  scf.forall (%id) in (2048) {
    %delin:3 = affine.delinearize_index %id into (4, 16, 32) : index, index, index
    %inner = arith.muli %delin#2, %c2 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    %lhs_vec_local_t = vector.shape_cast %lhs_vec_local : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
    vector.transfer_write %lhs_vec_local_t, %lhs_shared[%delin#0, %delin#1, %inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:3 = affine.delinearize_index %id into (4, 16, 32) : index, index, index
    %inner = arith.muli %delin#2, %c2 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    %rhs_vec_local_t = vector.shape_cast %rhs_vec_local : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
    vector.transfer_write %rhs_vec_local_t, %rhs_shared[%delin#0, %delin#1, %inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier memfence [#gpu.address_space<workgroup>]

  %result = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%out_ref = %unused_acc)[%sg_id: index, %num_sg: index]
         : (!out_sref) -> (!acc_base_ty) {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %sg_size: index] {
      %id = affine.linearize_index disjoint [%sg_id, %lane_id] by (%num_sg, %sg_size) : index
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
        %lhs_thread_0 = tensor.extract_slice %lhs [%i, %c0, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %lhs_vec_local_0_t = vector.shape_cast %lhs_vec_local_0 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
        %lhs_thread_1 = tensor.extract_slice %lhs [%i, %c1, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %lhs_vec_local_1_t = vector.shape_cast %lhs_vec_local_1 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
        %lhs_thread_2 = tensor.extract_slice %lhs [%i, %c2, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %lhs_vec_local_2_t = vector.shape_cast %lhs_vec_local_2 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
        %lhs_thread_3 = tensor.extract_slice %lhs [%i, %c3, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %lhs_vec_local_3_t = vector.shape_cast %lhs_vec_local_3 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>

        // Local loads of lhs and rhs.
        %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
        %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
        %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
        %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

        gpu.barrier memfence [#gpu.address_space<workgroup>]
        rocdl.sched.barrier 0
        rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

        %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

        rocdl.s.setprio 0
        gpu.barrier memfence [#gpu.address_space<workgroup>]
        rocdl.sched.barrier 0

        // Global loads of rhs.
        %rhs_thread_0 = tensor.extract_slice %rhs [%i, %c0, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %rhs_vec_local_0_t = vector.shape_cast %rhs_vec_local_0 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
        %rhs_thread_1 = tensor.extract_slice %rhs [%i, %c1, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %rhs_vec_local_1_t = vector.shape_cast %rhs_vec_local_1 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
        %rhs_thread_2 = tensor.extract_slice %rhs [%i, %c2, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %rhs_vec_local_2_t = vector.shape_cast %rhs_vec_local_2 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>
        %rhs_thread_3 = tensor.extract_slice %rhs [%i, %c3, %glb#0, %glb#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : !in_ty to tensor<1x1x1x8xf16>
        %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
        %rhs_vec_local_3_t = vector.shape_cast %rhs_vec_local_3 : vector<1x1x1x8xf16> to vector<1x1x2x4xf16>

        // Local loads of lhs and rhs.
        %lhs_vec_1 = vector.transfer_read %lhs_shared[%c1, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
        %rhs_vec_1 = vector.transfer_read %rhs_shared[%c1, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
        %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
        %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

        gpu.barrier memfence [#gpu.address_space<workgroup>]
        rocdl.sched.barrier 0
        rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

        %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

        rocdl.s.setprio 0
        gpu.barrier memfence [#gpu.address_space<workgroup>]
        rocdl.sched.barrier 0

        // Local loads of lhs and rhs.
        %lhs_vec_2 = vector.transfer_read %lhs_shared[%c2, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
        %rhs_vec_2 = vector.transfer_read %rhs_shared[%c2, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
        %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
        %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

        %lhs_vec_3 = vector.transfer_read %lhs_shared[%c3, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
        %rhs_vec_3 = vector.transfer_read %rhs_shared[%c3, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
        %lhs_vec_3_t = vector.shape_cast %lhs_vec_3 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
        %rhs_vec_3_t = vector.shape_cast %rhs_vec_3 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

        gpu.barrier memfence [#gpu.address_space<workgroup>]
        rocdl.sched.barrier 0
        rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

        %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot1) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

        rocdl.s.setprio 0
        gpu.barrier memfence [#gpu.address_space<workgroup>]
        rocdl.sched.barrier 0

        // Local stores of lhs and rhs.
        vector.transfer_write %rhs_vec_local_0_t, %rhs_shared [%c0, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
        vector.transfer_write %rhs_vec_local_1_t, %rhs_shared [%c1, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
        vector.transfer_write %rhs_vec_local_2_t, %rhs_shared [%c2, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
        vector.transfer_write %rhs_vec_local_3_t, %rhs_shared [%c3, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty

        vector.transfer_write %lhs_vec_local_0_t, %lhs_shared [%c0, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
        vector.transfer_write %lhs_vec_local_1_t, %lhs_shared [%c1, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
        vector.transfer_write %lhs_vec_local_2_t, %lhs_shared [%c2, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty
        vector.transfer_write %lhs_vec_local_3_t, %lhs_shared [%c3, %glb#0, %glb_inner, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf16>, !shared_ty

        gpu.barrier memfence [#gpu.address_space<workgroup>]
        rocdl.sched.barrier 0
        rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

        %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3_t, %rhs_vec_3_t) outs(%dot2) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
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
      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      %lhs_vec_1 = vector.transfer_read %lhs_shared[%c1, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
      %rhs_vec_1 = vector.transfer_read %rhs_shared[%c1, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
      %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c2, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c2, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot1) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      %lhs_vec_3 = vector.transfer_read %lhs_shared[%c3, %m_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x8x1x4xf16>
      %rhs_vec_3 = vector.transfer_read %rhs_shared[%c3, %n_outer, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : !shared_ty, vector<1x4x1x4xf16>
      %lhs_vec_3_t = vector.shape_cast %lhs_vec_3 : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_3_t = vector.shape_cast %rhs_vec_3 : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3_t, %rhs_vec_3_t) outs(%dot2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      %empty = tensor.empty() : tensor<1x1x1x1x8x4x1x1x4xf32>
      %cast = vector.shape_cast %dot3 : vector<8x4x1x4xf32> to vector<1x1x1x1x8x4x1x1x4xf32>
      %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true, true]} : vector<1x1x1x1x8x4x1x1x4xf32>, tensor<1x1x1x1x8x4x1x1x4xf32>

      pcf.write_slice %4 into %out_ref[%c0, %c0, %ids#0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 1, 8, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x4x1x1x4xf32> into !out_sref
      pcf.return
    }
    pcf.return
  }
  util.return %result : !acc_base_ty
}
