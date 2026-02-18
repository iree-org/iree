// RUN: iree-opt %s

// Input LHS and RHS layouts.
!lhs_ty = tensor<1x?x1x2x4x2x4x16x32xf4E2M1FN>
!rhs_ty = tensor<1x?x1x2x8x2x4x16x32xf4E2M1FN>

!lhs_byte_ty = tensor<1x?x1x2x4x2x4x16x16xi8>
!rhs_byte_ty = tensor<1x?x1x2x8x2x4x16x16xi8>

!lhs_buffer_ty = memref<1x?x1x2x4x2x4x16x16xi8, strided<[?, 16384, 16384, 8192, 2048, 1024, 256, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_ty = memref<1x?x1x2x8x2x4x16x16xi8, strided<[?, 32768, 32768, 16384, 2048, 1024, 256, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_buffer_collapse_ty = memref<?x16384xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_collapse_ty = memref<?x32768xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_shared_ty = memref<32x1024xi8, #gpu.address_space<workgroup>>
!rhs_shared_ty = memref<64x1024xi8, #gpu.address_space<workgroup>>

!lhs_copy_vec_ty = vector<16xi8>
!rhs_copy_vec_ty = vector<16xi8>

!lhs_byte_vec_ty = vector<4x1x2x16xi8>
!lhs_vec_ty = vector<4x1x2x32xf4E2M1FN>

!rhs_byte_vec_ty = vector<8x1x2x16xi8>
!rhs_vec_ty = vector<8x1x2x32xf4E2M1FN>

// Input scale layouts.
!lhs_scale_ty = tensor<1x?x2x4x16x4x2xf8E8M0FNU>
!rhs_scale_ty = tensor<1x?x2x4x16x8x2xf8E8M0FNU>

!lhs_scale_byte_ty = tensor<1x?x2x4x16x4x2xi8>
!rhs_scale_byte_ty = tensor<1x?x2x4x16x8x2xi8>

!lhs_scale_buffer_ty = memref<1x?x2x4x16x4x2xi8, strided<[?, 1024, 512, 128, 8, 2, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_ty = memref<1x?x2x4x16x8x2xi8, strided<[?, 2048, 1024, 256, 16, 2, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_buffer_collapse_ty = memref<?x1024xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_collapse_ty = memref<?x2048xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_copy_vec_ty = vector<16xi8>
!rhs_scale_copy_vec_ty = vector<16xi8>

!lhs_scale_shared_ty = memref<4x512xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_ty = memref<4x1024xi8, #gpu.address_space<workgroup>>

!lhs_scale_byte_vec_ty = vector<4x1x2x1xi8>
!lhs_scale_vec_ty = vector<4x1x2x1xf8E8M0FNU>

!rhs_scale_byte_vec_ty = vector<8x1x2x1xi8>
!rhs_scale_vec_ty = vector<8x1x2x1xf8E8M0FNU>

!return_ty = tensor<1x1x2x2x4x8x4x16x4xf32>

!acc_ty = vector<4x8x4x1xf32>

!store_ty = vector<1x1x1x1x4x8x1x1x4xf32>
!tensor_store_ty = tensor<1x1x1x1x4x8x1x1x4xf32>

!out_sref = !pcf.sref<1x1x2x2x4x8x4x16x4xf32, #iree_gpu.subgroup_scope>

#contraction_accesses = [
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (i, j)>
]

#iterator_types = [
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<reduction>
]

#mfma_type = #iree_gpu.scaled_mma_layout<
  intrinsic = MFMA_SCALE_F32_16x16x128_B32,
  lhs_elem_type = f4E2M1FN,
  rhs_elem_type = f4E2M1FN,
  acc_elem_type = f32>

#result_reassoc = [[0, 1], [2, 3]]

util.func @pingpong_dt_medium_f4E2M1FN(
    %lhs_base: !lhs_ty,
    %rhs_base: !rhs_ty,
    %lhs_scale_base: !lhs_scale_ty,
    %rhs_scale_base: !rhs_scale_ty,
    %unused_acc: !return_ty) -> !return_ty attributes {
  ukernel_info = #rocm.ukernel_info<
    match = {
      archs = ["gfx950"],
      types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32]
    },
    benefit = 1,
    // Tile size calculation:
    // M dimension: 16 (intrinsic) × 4 (intrinsics_m) × 2 (subgroups_m) = 128.
    // N dimension: 16 (intrinsic) × 8 (intrinsics_n) × 2 (subgroups_n) = 256.
    // KxKB dimension: 128 (intrinsic) × 2 (intrinsics_k) = 256.
    mma = #iree_gpu.data_tiled_scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f4E2M1FN,
      acc_elem_type = f32,
      intrinsics_m = 4, operands_interleaving_intrinsics_m = [2],
      subgroups_m = 2,
      intrinsics_n = 8, operands_interleaving_intrinsics_n = [3],
      subgroups_n = 2,
      intrinsics_k = 2, operands_interleaving_intrinsics_k = [2, 3]
    >
  >
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %c4096 = arith.constant 4096 : index

  %cst_lhs = arith.constant 0 : i8
  %cst_rhs = arith.constant 0 : i8
  %cst_scale = arith.constant 0 : i8
  %lhs_shared_base = memref.alloc() : !lhs_shared_ty
  %rhs_shared_base = memref.alloc() : !rhs_shared_ty
  %lhs_scale_shared = memref.alloc() : !lhs_scale_shared_ty
  %rhs_scale_shared = memref.alloc() : !rhs_scale_shared_ty

  %k = tensor.dim %lhs_base, %c1 : !lhs_ty

  // Bitcast inputs to i8 type. This is required to use gather_to_lds instructions.
  %lhs_byte = iree_tensor_ext.bitcast %lhs_base : !lhs_ty{%k} -> !lhs_byte_ty{%k}
  %rhs_byte = iree_tensor_ext.bitcast %rhs_base : !rhs_ty{%k} -> !rhs_byte_ty{%k}

  %lhs_scale_byte = iree_tensor_ext.bitcast %lhs_scale_base : !lhs_scale_ty{%k} -> !lhs_scale_byte_ty{%k}
  %rhs_scale_byte = iree_tensor_ext.bitcast %rhs_scale_base : !rhs_scale_ty{%k} -> !rhs_scale_byte_ty{%k}

  %lhs = bufferization.to_buffer %lhs_byte {read_only} : !lhs_byte_ty to !lhs_buffer_ty
  %rhs = bufferization.to_buffer %rhs_byte {read_only} : !rhs_byte_ty to !rhs_buffer_ty

  %lhs_scale = bufferization.to_buffer %lhs_scale_byte {read_only} : !lhs_scale_byte_ty to !lhs_scale_buffer_ty
  %rhs_scale = bufferization.to_buffer %rhs_scale_byte {read_only} : !rhs_scale_byte_ty to !rhs_scale_buffer_ty

  // Collapse shapes to reduce memory indexing overhead.
  %lhs_collapse = memref.collapse_shape %lhs [[0, 1], [2, 3, 4, 5, 6, 7, 8]] : !lhs_buffer_ty into !lhs_buffer_collapse_ty
  %rhs_collapse = memref.collapse_shape %rhs [[0, 1], [2, 3, 4, 5, 6, 7, 8]] : !rhs_buffer_ty into !rhs_buffer_collapse_ty
  %lhs_scale_collapse = memref.collapse_shape %lhs_scale [[0, 1], [2, 3, 4, 5, 6]] : !lhs_scale_buffer_ty into !lhs_scale_buffer_collapse_ty
  %rhs_scale_collapse = memref.collapse_shape %rhs_scale [[0, 1], [2, 3, 4, 5, 6]] : !rhs_scale_buffer_ty into !rhs_scale_buffer_collapse_ty

  scf.forall (%base_id) in (512) {
    // Make the upper 4 waves start copying data.
    %cmp = arith.cmpi sge, %base_id, %c256 : index
    scf.if %cmp {
      %id = arith.subi %base_id, %c256 : index

      %ids:2 = affine.delinearize_index %id into (4, 64) : index, index
      %subgroups:2 = affine.delinearize_index %ids#0 into (2, 2) : index, index
      %threads:2 = affine.delinearize_index %ids#1 into (2, 32) : index, index

      scf.for %i = %c0 to %k step %c1 {
        // Double buffering.
        %buffer_num = arith.andi %i, %c1 : index

        // Copy RHS from global memory to LDS.
        scf.for %j = %c0 to %c8 step %c1 {
          %buffer_unroll = arith.muli %j, %c4096 : index
          %buffer_thread = arith.muli %id, %c16 : index
          %buffer_inner = arith.addi %buffer_unroll, %buffer_thread : index

          %shared_num = arith.muli %buffer_num, %c32 : index
          %shared_unroll = arith.muli %j, %c4 : index
          %shared_subgroup = arith.addi %shared_unroll, %ids#0 : index
          %shared_outer = arith.addi %shared_num, %shared_subgroup : index
          %shared_inner = arith.muli %ids#1, %c16 : index

          amdgpu.gather_to_lds %rhs_collapse[%i, %buffer_inner], %rhs_shared_base[%shared_outer, %shared_inner]
            : !rhs_copy_vec_ty, !rhs_buffer_collapse_ty, !rhs_shared_ty
        }

        // Copy RHS scale from global memory to LDS.
        %rhs_scale_buffer_subgroup = arith.muli %subgroups#1, %c1024 : index
        %rhs_scale_buffer_thread = arith.muli %ids#1, %c16 : index
        %rhs_scale_buffer_inner = arith.addi %rhs_scale_buffer_subgroup, %rhs_scale_buffer_thread : index

        %rhs_scale_shared_num = arith.muli %buffer_num, %c2 : index
        %rhs_scale_shared_outer = arith.addi %rhs_scale_shared_num, %subgroups#1 : index
        %rhs_scale_shared_inner = arith.muli %ids#1, %c16 : index

        amdgpu.gather_to_lds %rhs_scale_collapse[%i, %rhs_scale_buffer_inner], %rhs_scale_shared[%rhs_scale_shared_outer, %rhs_scale_shared_inner]
          : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_collapse_ty, !rhs_scale_shared_ty

        // There are (8 + 1) gather_to_lds instructions above.
        amdgpu.memory_counter_wait load(9)
        rocdl.s.barrier

        // Copy LHS from global memory to LDS.
        scf.for %j = %c0 to %c4 step %c1 {
          %buffer_unroll = arith.muli %j, %c4096 : index
          %buffer_thread = arith.muli %id, %c16 : index
          %buffer_inner = arith.addi %buffer_unroll, %buffer_thread : index

          %shared_num = arith.muli %buffer_num, %c16 : index
          %shared_unroll = arith.muli %j, %c4 : index
          %shared_subgroup = arith.addi %shared_unroll, %ids#0 : index
          %shared_outer = arith.addi %shared_num, %shared_subgroup : index
          %shared_inner = arith.muli %ids#1, %c16 : index

          amdgpu.gather_to_lds %lhs_collapse[%i, %buffer_inner], %lhs_shared_base[%shared_outer, %shared_inner]
            : !lhs_copy_vec_ty, !lhs_buffer_collapse_ty, !lhs_shared_ty
        }

        // Copy LHS scale from global memory to LDS.
        %lhs_scale_buffer_inner = arith.muli %ids#1, %c16 : index

        %lhs_scale_shared_num = arith.muli %buffer_num, %c2 : index
        %lhs_scale_shared_outer = arith.addi %lhs_scale_shared_num, %threads#0 : index
        %lhs_scale_shared_inner = arith.muli %threads#1, %c16 : index

        amdgpu.gather_to_lds %lhs_scale_collapse[%i, %lhs_scale_buffer_inner], %lhs_scale_shared[%lhs_scale_shared_outer, %lhs_scale_shared_inner]
          : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_collapse_ty, !lhs_scale_shared_ty

        // There are (4 + 1) gather_to_lds instructions above.
        amdgpu.memory_counter_wait load(5)
        rocdl.s.barrier


        scf.yield
      }

      // Realign subgroups and wait on the last group.
      rocdl.s.waitcnt 0
      rocdl.s.barrier
    }

  } {mapping = [#gpu.thread<linear_dim_0>]}


  %result = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%out_ref = %unused_acc)[%sg_id: index, %num_sg: index]
         : (!out_sref) -> (!return_ty) {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %sg_size: index] {
      %id = affine.linearize_index disjoint [%sg_id, %lane_id] by (%num_sg, %sg_size) : index
      %init = arith.constant dense<0.0> : !acc_ty
      %ids:2 = affine.delinearize_index %id into (4, 64) : index, index
      %subgroups:2 = affine.delinearize_index %ids#0 into (2, 2) : index, index
      %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

      // Misalign by one group.
      rocdl.s.barrier

      %loop = scf.for %i = %c0 to %k step %c1 iter_args(%iter = %init) -> !acc_ty {
        // wait till available.
        rocdl.s.barrier

        %buffer_num = arith.andi %i, %c1 : index

        %lhs_num = arith.muli %buffer_num, %c16 : index
        %lhs_subgroup = arith.muli %subgroups#0, %c8 : index
        %lhs_outer = arith.addi %lhs_num, %lhs_subgroup : index
        %lhs_inner = arith.muli %ids#1, %c16 : index

        %lhs_scale_num = arith.muli %buffer_num, %c2 : index
        %lhs_scale_outer = arith.addi %lhs_scale_num, %subgroups#0 : index
        %lhs_scale_inner = arith.muli %ids#1, %c8 : index

        %rhs_num = arith.muli %buffer_num, %c32 : index
        %rhs_subgroup = arith.muli %subgroups#1, %c16 : index
        %rhs_outer = arith.addi %rhs_num, %rhs_subgroup : index
        %rhs_inner = arith.muli %ids#1, %c16 : index

        %rhs_scale_num = arith.muli %buffer_num, %c2 : index
        %rhs_scale_outer = arith.addi %rhs_scale_num, %subgroups#1 : index
        %rhs_scale_inner = arith.muli %ids#1, %c16 : index

        // Load inputs/scales from LDS.
        %lhs_byte_vec = vector.transfer_read %lhs_shared_base[%lhs_outer, %lhs_inner],
          %cst_lhs {in_bounds = [true, true]} : !lhs_shared_ty, vector<8x16xi8>
        %lhs_byte_vec_t = vector.shape_cast %lhs_byte_vec : vector<8x16xi8> to !lhs_byte_vec_ty
        %lhs_vec = vector.bitcast %lhs_byte_vec_t : !lhs_byte_vec_ty to !lhs_vec_ty

        %lhs_scale_byte_vec = vector.transfer_read %lhs_scale_shared[%lhs_scale_outer, %lhs_scale_inner],
          %cst_scale {in_bounds = [true, true]} : !lhs_scale_shared_ty, vector<1x8xi8>
        %lhs_scale_byte_vec_t = vector.shape_cast %lhs_scale_byte_vec : vector<1x8xi8> to !lhs_scale_byte_vec_ty
        %lhs_scale_vec = vector.bitcast %lhs_scale_byte_vec_t : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty

        rocdl.sched.barrier 0

        %rhs_byte_vec = vector.transfer_read %rhs_shared_base[%rhs_outer, %rhs_inner],
          %cst_rhs {in_bounds = [true, true]} : !rhs_shared_ty, vector<16x16xi8>
        %rhs_byte_vec_t = vector.shape_cast %rhs_byte_vec : vector<16x16xi8> to !rhs_byte_vec_ty
        %rhs_vec = vector.bitcast %rhs_byte_vec_t : !rhs_byte_vec_ty to !rhs_vec_ty

        %rhs_scale_byte_vec = vector.transfer_read %rhs_scale_shared[%rhs_scale_outer, %rhs_scale_inner],
          %cst_scale {in_bounds = [true, true]} : !rhs_scale_shared_ty, vector<1x16xi8>
        %rhs_scale_byte_vec_t = vector.shape_cast %rhs_scale_byte_vec : vector<1x16xi8> to !rhs_scale_byte_vec_ty
        %rhs_scale_vec = vector.bitcast %rhs_scale_byte_vec_t : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

        amdgpu.lds_barrier
        rocdl.sched.barrier 0

        %dot = iree_codegen.inner_tiled ins(%lhs_vec, %rhs_vec, %lhs_scale_vec, %rhs_scale_vec) outs(%iter) {
          indexing_maps = #contraction_accesses,
          iterator_types = #iterator_types,
          kind = #mfma_type,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : !lhs_vec_ty, !rhs_vec_ty, !lhs_scale_vec_ty, !rhs_scale_vec_ty into !acc_ty

        scf.yield %dot : !acc_ty
      }

      %t = vector.shape_cast %loop : !acc_ty to !store_ty

      %empty = tensor.empty() : !tensor_store_ty
      %to_tensor = vector.transfer_write %t, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0]
        {in_bounds = [true, true, true, true, true, true, true, true, true]} : !store_ty, !tensor_store_ty

      pcf.write_slice %to_tensor into %out_ref[%c0, %c0, %subgroups#0, %subgroups#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 1, 4, 8, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1, 1] : !tensor_store_ty into !out_sref
      pcf.return
    }
    pcf.return
  }
  util.return %result : !return_ty
}
