// RUN: iree-opt %s

// Element types

!lhs = f4E2M1FN
!rhs = f4E2M1FN

!scale_ty = f8E8M0FNU

// Inputs

!lhs_ty = tensor<1x?x1x4x4x4x16x32x!lhs>
!rhs_ty = tensor<1x?x1x4x2x4x4x16x32x!rhs>

!lhs_byte_ty = tensor<1x?x1x4x4x4x16x16xi8>
!rhs_byte_ty = tensor<1x?x1x4x2x4x4x16x16xi8>

!lhs_buffer_ty = memref<1x?x1x4x4x4x16x16xi8, strided<[?, 16384, 16384, 4096, 1024, 256, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_ty = memref<1x?x1x4x2x4x4x16x16xi8, strided<[?, 32768, 32768, 8192, 4096, 1024, 256, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_buffer_collapse_ty = memref<?x16384xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_collapse_ty = memref<?x32768xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

// 2 = double buffer
// 64 = m tile
// 64 = n tile
// 8 = k split factor
// 64 = 128 / 2 = k tile / 2
!lhs_shared_ty = memref<32x1024xi8, #gpu.address_space<workgroup>>
!rhs_shared_ty = memref<64x1024xi8, #gpu.address_space<workgroup>>

!lhs_copy_vec_ty = vector<16xi8>
!rhs_copy_vec_ty = vector<16xi8>

!lhs_byte_vec_ty = vector<4x1x4x16xi8>
!lhs_vec_ty = vector<4x1x4x32xf4E2M1FN>

!rhs_byte_vec_ty = vector<2x1x4x16xi8>
!rhs_vec_ty = vector<2x1x4x32xf4E2M1FN>

// Scales

!lhs_scale_ty = tensor<1x?x4x4x16x4x!scale_ty>
!rhs_scale_ty = tensor<1x?x4x2x4x16x4x!scale_ty>

!lhs_scale_byte_ty = tensor<1x?x4x4x16x4xi8>
!rhs_scale_byte_ty = tensor<1x?x4x2x4x16x4xi8>

!lhs_scale_buffer_ty = memref<1x?x4x4x16x4xi8, strided<[?, 1024, 256, 64, 4, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_ty = memref<1x?x4x2x4x16x4xi8, strided<[?, 2048, 512, 256, 64, 4, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_buffer_collapse_ty = memref<?x1024xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_collapse_ty = memref<?x2048xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_copy_vec_ty = vector<16xi8>
!rhs_scale_copy_vec_ty = vector<16xi8>

// 2 = double buffer
// 64 = m tile
// 64 = n tile
// 8 = k split factor
// 4 = k tile / k block
!lhs_scale_shared_ty = memref<8x256xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_ty = memref<16x256xi8, #gpu.address_space<workgroup>>

!lhs_scale_byte_vec_ty = vector<4x1x4x1xi8>
!lhs_scale_vec_ty = vector<4x1x4x1xf8E8M0FNU>

!rhs_scale_byte_vec_ty = vector<2x1x4x1xi8>
!rhs_scale_vec_ty = vector<2x1x4x1xf8E8M0FNU>

!res_ty = tensor<4x16x8x16xf32>
!return_ty = tensor<1x1x4x4x2x4x16x4xf32>

!acc_ty = vector<4x2x4x1xf32>

!store_ty = vector<1x1x1x4x2x1x1x4xf32>
!tensor_store_ty = tensor<1x1x1x4x2x1x1x4xf32>

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

util.func @dt_scaled_matmul_f4f4f32_m64_n128_k512(
    %lhs_base: !lhs_ty,
    %rhs_base: !rhs_ty,
    %lhs_scale_base: !lhs_scale_ty,
    %rhs_scale_base: !rhs_scale_ty,
    %unused_acc: !return_ty) -> !return_ty attributes {
  ukernel_info = #rocm.ukernel_info<
    match = {
      types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32]
    },
    benefit = 2,
    mma = #iree_gpu.data_tiled_scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f4E2M1FN,
      acc_elem_type = f32,
      intrinsics_m = 4,
      intrinsics_n = 2,
      subgroups_n = 4,
      intrinsics_k = 4, operands_interleaving_intrinsics_k = [2, 3]
    >
  >
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
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

  %lhs_byte = iree_tensor_ext.bitcast %lhs_base : !lhs_ty{%k} -> !lhs_byte_ty{%k}
  %rhs_byte = iree_tensor_ext.bitcast %rhs_base : !rhs_ty{%k} -> !rhs_byte_ty{%k}

  %lhs_scale_byte = iree_tensor_ext.bitcast %lhs_scale_base : !lhs_scale_ty{%k} -> !lhs_scale_byte_ty{%k}
  %rhs_scale_byte = iree_tensor_ext.bitcast %rhs_scale_base : !rhs_scale_ty{%k} -> !rhs_scale_byte_ty{%k}

  %lhs = bufferization.to_buffer %lhs_byte {read_only} : !lhs_byte_ty to !lhs_buffer_ty
  %rhs = bufferization.to_buffer %rhs_byte {read_only} : !rhs_byte_ty to !rhs_buffer_ty

  %lhs_scale = bufferization.to_buffer %lhs_scale_byte {read_only} : !lhs_scale_byte_ty to !lhs_scale_buffer_ty
  %rhs_scale = bufferization.to_buffer %rhs_scale_byte {read_only} : !rhs_scale_byte_ty to !rhs_scale_buffer_ty

  %lhs_collapse = memref.collapse_shape %lhs [[0, 1], [2, 3, 4, 5, 6, 7]] : !lhs_buffer_ty into !lhs_buffer_collapse_ty
  %rhs_collapse = memref.collapse_shape %rhs [[0, 1], [2, 3, 4, 5, 6, 7, 8]] : !rhs_buffer_ty into !rhs_buffer_collapse_ty
  %lhs_scale_collapse = memref.collapse_shape %lhs_scale [[0, 1], [2, 3, 4, 5]] : !lhs_scale_buffer_ty into !lhs_scale_buffer_collapse_ty
  %rhs_scale_collapse = memref.collapse_shape %rhs_scale [[0, 1], [2, 3, 4, 5, 6]] : !rhs_scale_buffer_ty into !rhs_scale_buffer_collapse_ty

  scf.forall (%base_id) in (512) {
    // Make the upper 4 waves start copying data.
    %cmp = arith.cmpi sge, %base_id, %c256 : index
    scf.if %cmp {
      %id = arith.subi %base_id, %c256 : index

      %ids:2 = affine.delinearize_index %id into (4, 64) : index, index
      %subgroups:2 = affine.delinearize_index %ids#0 into (2, 2) : index, index
      %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

      scf.for %i = %c0 to %k step %c1 {
        %buffer_num = arith.andi %i, %c1 : index

        // Copy inputs.
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

        scf.for %j = %c0 to %c1 step %c1 {
          %buffer_inner = arith.muli %ids#1, %c16 : index

          %shared_num = arith.muli %buffer_num, %c4 : index
          %shared_outer = arith.addi %shared_num, %threads#0 : index
          %shared_inner = arith.muli %threads#1, %c16 : index

          amdgpu.gather_to_lds %lhs_scale_collapse[%i, %buffer_inner], %lhs_scale_shared[%shared_outer, %shared_inner]
            : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_collapse_ty, !lhs_scale_shared_ty
        }

        // Wait on previous group.
        rocdl.s.waitcnt 5
        rocdl.s.barrier

        // Copy scales.
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

        scf.for %j = %c0 to %c1 step %c1 {
          %buffer_subgroup = arith.muli %subgroups#1, %c1024 : index
          %buffer_thread = arith.muli %ids#1, %c16 : index
          %buffer_inner = arith.addi %buffer_subgroup, %buffer_thread : index

          %shared_num = arith.muli %buffer_num, %c8 : index
          %shared_subgroup = arith.muli %subgroups#1, %c4 : index
          %shared_thread = arith.addi %shared_subgroup, %threads#0 : index
          %shared_outer = arith.addi %shared_num, %shared_thread : index
          %shared_inner = arith.muli %threads#1, %c16 : index
          amdgpu.gather_to_lds %rhs_scale_collapse[%i, %buffer_inner], %rhs_scale_shared[%shared_outer, %shared_inner]
            : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_collapse_ty, !rhs_scale_shared_ty
        }

        // Wait on previous group.
        rocdl.s.waitcnt 9
        rocdl.s.barrier

        scf.yield
      }

      // Realign subgroups and wait on the last group.
      rocdl.s.waitcnt 0
      rocdl.s.barrier
    }

  } {mapping = [#gpu.thread<linear_dim_0>]}


  %0 = tensor.empty() : !return_ty
  %1 = scf.forall (%id) in (256) shared_outs(%out = %0) -> !return_ty {
    %init = arith.constant dense<0.0> : !acc_ty
    %ids:2 = affine.delinearize_index %id into (4, 64) : index, index
    %subgroups:2 = affine.delinearize_index %ids#0 into (1, 4) : index, index
    %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

    // Misalign by one group.
    rocdl.s.barrier

    %loop = scf.for %i = %c0 to %k step %c1 iter_args(%iter = %init) -> !acc_ty {
      // wait till available.
      rocdl.s.barrier

      %buffer_num = arith.andi %i, %c1 : index

      %lhs_outer = arith.muli %buffer_num, %c16 : index
      %lhs_inner = arith.muli %ids#1, %c16 : index
      %lhs_scale_outer = arith.muli %buffer_num, %c4 : index
      %lhs_scale_inner = arith.muli %ids#1, %c4 : index

      %rhs_num = arith.muli %buffer_num, %c32 : index
      %rhs_subgroup = arith.muli %subgroups#1, %c8 : index
      %rhs_outer = arith.addi %rhs_num, %rhs_subgroup : index
      %rhs_inner = arith.muli %ids#1, %c16 : index
      %rhs_scale_num = arith.muli %buffer_num, %c8 : index
      %rhs_scale_subgroup = arith.muli %subgroups#1, %c2 : index
      %rhs_scale_outer = arith.addi %rhs_scale_num, %rhs_scale_subgroup : index
      %rhs_scale_inner = arith.muli %ids#1, %c4 : index

      // Load inputs/scales from LDS.
      %lhs_byte_vec = vector.transfer_read %lhs_shared_base[%lhs_outer, %lhs_inner],
        %cst_lhs {in_bounds = [true, true]} : !lhs_shared_ty, vector<16x16xi8>
      %lhs_byte_vec_t = vector.shape_cast %lhs_byte_vec : vector<16x16xi8> to !lhs_byte_vec_ty
      %lhs_vec = vector.bitcast %lhs_byte_vec_t : !lhs_byte_vec_ty to !lhs_vec_ty

      %lhs_scale_byte_vec = vector.transfer_read %lhs_scale_shared[%lhs_scale_outer, %lhs_scale_inner],
        %cst_scale {in_bounds = [true, true]} : !lhs_scale_shared_ty, vector<4x4xi8>
      %lhs_scale_byte_vec_t = vector.shape_cast %lhs_scale_byte_vec : vector<4x4xi8> to !lhs_scale_byte_vec_ty
      %lhs_scale_vec = vector.bitcast %lhs_scale_byte_vec_t : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty

      rocdl.sched.barrier 0

      %rhs_byte_vec = vector.transfer_read %rhs_shared_base[%rhs_outer, %rhs_inner],
        %cst_rhs {in_bounds = [true, true]} : !rhs_shared_ty, vector<8x16xi8>
      %rhs_byte_vec_t = vector.shape_cast %rhs_byte_vec : vector<8x16xi8> to !rhs_byte_vec_ty
      %rhs_vec = vector.bitcast %rhs_byte_vec_t : !rhs_byte_vec_ty to !rhs_vec_ty

      %rhs_scale_byte_vec = vector.transfer_read %rhs_scale_shared[%rhs_scale_outer, %rhs_scale_inner],
        %cst_scale {in_bounds = [true, true]} : !rhs_scale_shared_ty, vector<2x4xi8>
      %rhs_scale_byte_vec_t = vector.shape_cast %rhs_scale_byte_vec : vector<2x4xi8> to !rhs_scale_byte_vec_ty
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
    %to_tensor = vector.transfer_write %t, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0]
      {in_bounds = [true, true, true, true, true, true, true, true]} : !store_ty, !tensor_store_ty

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %to_tensor into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 4, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1]
        : !tensor_store_ty into !return_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : !return_ty
}
