// RUN: iree-opt %s

// Element types

!lhs = f4E2M1FN
!rhs = f4E2M1FN

!scale_ty = f8E8M0FNU

// Inputs

!lhs_ty = tensor<1x?x1x2x2x4x4x16x32x!lhs>
!rhs_ty = tensor<1x?x1x4x4x4x16x32x!rhs>

!lhs_byte_ty = tensor<1x?x1x2x2x4x4x16x16xi8>
!rhs_byte_ty = tensor<1x?x1x4x4x4x16x16xi8>

!lhs_buffer_ty = memref<1x?x1x2x2x4x4x16x16xi8, strided<[?, 16384, 16384, 8192, 4096, 1024, 256, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_ty = memref<1x?x1x4x4x4x16x16xi8, strided<[?, 16384, 16384, 4096, 1024, 256, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

// 2 = double buffer
// 64 = m tile
// 64 = n tile
// 8 = k split factor
// 64 = 128 / 2 = k tile / 2
!lhs_shared_ty = memref<2x2x2x4x4x16x16xi8, #gpu.address_space<workgroup>>
!rhs_shared_ty = memref<2x4x4x4x16x16xi8, #gpu.address_space<workgroup>>

!lhs_copy_vec_ty = vector<16xi8>
!rhs_copy_vec_ty = vector<16xi8>

!lhs_byte_vec_ty = vector<2x1x4x16xi8>
!lhs_vec_ty = vector<2x1x4x32xf4E2M1FN>

!rhs_byte_vec_ty = vector<1x1x4x16xi8>
!rhs_vec_ty = vector<1x1x4x32xf4E2M1FN>

// Scales

!lhs_scale_ty = tensor<1x?x2x2x4x16x4x!scale_ty>
!rhs_scale_ty = tensor<1x?x4x4x16x4x!scale_ty>

!lhs_scale_byte_ty = tensor<1x?x2x2x4x16x4xi8>
!rhs_scale_byte_ty = tensor<1x?x4x4x16x4xi8>

!lhs_scale_buffer_ty = memref<1x?x2x2x4x16x4xi8, strided<[?, 1024, 512, 256, 64, 4, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_ty = memref<1x?x4x4x16x4xi8, strided<[?, 1024, 256, 64, 4, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_copy_vec_ty = vector<4xi8>
!rhs_scale_copy_vec_ty = vector<4xi8>

// 2 = double buffer
// 64 = m tile
// 64 = n tile
// 8 = k split factor
// 4 = k tile / k block
!lhs_scale_shared_ty = memref<2x2x2x4x16x4xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_ty = memref<2x4x4x16x4xi8, #gpu.address_space<workgroup>>

!lhs_scale_byte_vec_ty = vector<2x1x4x1xi8>
!lhs_scale_vec_ty = vector<2x1x4x1xf8E8M0FNU>

!rhs_scale_byte_vec_ty = vector<1x1x4x1xi8>
!rhs_scale_vec_ty = vector<1x1x4x1xf8E8M0FNU>


!res_ty = tensor<4x16x4x16xf32>
!return_ty = tensor<1x1x2x4x2x4x16x4xf32>

// subgroup_m = 2 waves subgroup_n = 4 waves
!acc_ty = vector<2x1x4x4x1xf32>

!reduce_ty = vector<2x1x4x1xf32>
!store_ty = vector<1x1x1x1x2x1x1x4xf32>
!tensor_store_ty = tensor<1x1x1x1x2x1x1x4xf32>

#contraction_accesses = [
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (i, j, k)>
]

#iterator_types = [
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>
]

#mfma_type = #iree_gpu.scaled_mma_layout<
  intrinsic = MFMA_SCALE_F32_16x16x128_B32,
  lhs_elem_type = f4E2M1FN,
  rhs_elem_type = f4E2M1FN,
  acc_elem_type = f32>

#result_reassoc = [[0, 1], [2, 3]]

util.func @dt_scaled_matmul_f4f4f32_m64_n64_k512(
    %lhs_base: !lhs_ty,
    %rhs_base: !rhs_ty,
    %lhs_scale_base: !lhs_scale_ty,
    %rhs_scale_base: !rhs_scale_ty,
    %unused_acc: !return_ty) -> !return_ty attributes {
  ukernel_info = #rocm.ukernel_info<
    match = {
      types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32]
    },
    benefit = 1,
    mma = #iree_gpu.data_tiled_scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f4E2M1FN,
      acc_elem_type = f32,
      intrinsics_m = 2,
      subgroups_m = 2,
      intrinsics_n = 1,
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

  scf.forall (%base_id) in (512) {
    // Copy inputs.
    %ids:4 = affine.delinearize_index %base_id into (2, 4, 4, 16) : index, index, index, index
    scf.for %i = %c0 to %c2 step %c1 {
      %rhs_outer_base = arith.muli %i, %c2 : index
      %rhs_outer = arith.addi %rhs_outer_base, %ids#0 : index
      amdgpu.gather_to_lds %lhs[%c0, %c0, %c0, %i, %ids#0, %ids#1, %ids#2, %ids#3, %c0], %lhs_shared_base[%c0, %i, %ids#0, %ids#1, %ids#2, %ids#3, %c0]
        : !lhs_copy_vec_ty, !lhs_buffer_ty, !lhs_shared_ty
      amdgpu.gather_to_lds %rhs[%c0, %c0, %c0, %rhs_outer, %ids#1, %ids#2, %ids#3, %c0], %rhs_shared_base[%c0, %rhs_outer, %ids#1, %ids#2, %ids#3, %c0]
        : !rhs_copy_vec_ty, !rhs_buffer_ty, !rhs_shared_ty
    }

    // Copy scales.
    %scale_id_outers:2 = affine.delinearize_index %ids#1 into (2, 2) : index, index
    %cmp0 = arith.cmpi slt, %base_id, %c256 : index
    %cmp1 = arith.cmpi sge, %base_id, %c256 : index
    scf.if %cmp0 {
      amdgpu.gather_to_lds %lhs_scale[%c0, %c0, %scale_id_outers#0, %scale_id_outers#1, %ids#2, %ids#3, %c0], %lhs_scale_shared[%c0, %scale_id_outers#0, %scale_id_outers#1, %ids#2, %ids#3, %c0]
        : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_ty, !lhs_scale_shared_ty
    }
    scf.if %cmp1 {
      amdgpu.gather_to_lds %rhs_scale[%c0, %c0, %ids#1, %ids#2, %ids#3, %c0], %rhs_scale_shared[%c0, %ids#1, %ids#2, %ids#3, %c0]
        : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_ty, !rhs_scale_shared_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : !return_ty
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> !return_ty {
    %init = arith.constant dense<0.0> : !acc_ty

    %ids:3 = affine.delinearize_index %id into (8, 4, 16) : index, index, index
    %subgroup_ids:2 = affine.delinearize_index %ids#0 into (2, 4) : index, index

    %scale_id_outers:2 = affine.delinearize_index %subgroup_ids#1 into (2, 2) : index, index

    // cond barrier.
    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %loop = scf.for %i = %c1 to %k step %c1 iter_args(%iter = %init) -> !acc_ty {
      %buffer_num = arith.andi %i, %c1 : index
      %curr_buffer_num = arith.xori %buffer_num, %c1 : index

      rocdl.s.waitcnt 0
      rocdl.s.barrier

      // Copy inputs / scales.
      scf.for %j = %c0 to %c2 step %c1 {
        %rhs_outer_base = arith.muli %j, %c2 : index
        %rhs_outer = arith.addi %rhs_outer_base, %subgroup_ids#0 : index
        amdgpu.gather_to_lds %lhs[%c0, %i, %c0, %j, %subgroup_ids#0, %subgroup_ids#1, %ids#1, %ids#2, %c0], %lhs_shared_base[%buffer_num, %j, %subgroup_ids#0, %subgroup_ids#1, %ids#1, %ids#2, %c0]
          : !lhs_copy_vec_ty, !lhs_buffer_ty, !lhs_shared_ty
        amdgpu.gather_to_lds %rhs[%c0, %i, %c0, %rhs_outer, %subgroup_ids#1, %ids#1, %ids#2, %c0], %rhs_shared_base[%buffer_num, %rhs_outer, %subgroup_ids#1, %ids#1, %ids#2, %c0]
          : !rhs_copy_vec_ty, !rhs_buffer_ty, !rhs_shared_ty
      }

      scf.if %cmp0 {
        amdgpu.gather_to_lds %lhs_scale[%c0, %i, %scale_id_outers#0, %scale_id_outers#1, %ids#1, %ids#2, %c0], %lhs_scale_shared[%buffer_num, %scale_id_outers#0, %scale_id_outers#1, %ids#1, %ids#2, %c0]
          : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_ty, !lhs_scale_shared_ty
      }
      scf.if %cmp1 {
        amdgpu.gather_to_lds %rhs_scale[%c0, %i, %subgroup_ids#1, %ids#1, %ids#2, %c0], %rhs_scale_shared[%buffer_num, %subgroup_ids#1, %ids#1, %ids#2, %c0]
          : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_ty, !rhs_scale_shared_ty
      }

      rocdl.s.barrier
      amdgpu.lds_barrier
      rocdl.s.waitcnt 0
      rocdl.sched.barrier 0
      rocdl.s.setprio 1

      // Load inputs/scales from LDS.
      %lhs_byte_vec = vector.transfer_read %lhs_shared_base[%curr_buffer_num, %subgroup_ids#0, %c0, %c0, %ids#1, %ids#2, %c0],
        %cst_lhs {in_bounds = [true, true, true, true, true, true, true]} : !lhs_shared_ty, vector<1x1x2x4x1x1x16xi8>
      %lhs_byte_vec_t = vector.shape_cast %lhs_byte_vec : vector<1x1x2x4x1x1x16xi8> to !lhs_byte_vec_ty
      %lhs_vec = vector.bitcast %lhs_byte_vec_t : !lhs_byte_vec_ty to !lhs_vec_ty
      %rhs_byte_vec = vector.transfer_read %rhs_shared_base[%curr_buffer_num, %subgroup_ids#1, %c0, %ids#1, %ids#2, %c0],
        %cst_rhs {in_bounds = [true, true, true, true, true, true]} : !rhs_shared_ty, vector<1x1x4x1x1x16xi8>
      %rhs_byte_vec_t = vector.shape_cast %rhs_byte_vec : vector<1x1x4x1x1x16xi8> to !rhs_byte_vec_ty
      %rhs_vec = vector.bitcast %rhs_byte_vec_t : !rhs_byte_vec_ty to !rhs_vec_ty

      %lhs_scale_byte_vec = vector.transfer_read %lhs_scale_shared[%curr_buffer_num, %subgroup_ids#0, %c0, %ids#1, %ids#2, %c0],
        %cst_scale {in_bounds = [true, true, true, true, true, true]} : !lhs_scale_shared_ty, vector<1x1x2x1x1x4xi8>
      %lhs_scale_byte_vec_t = vector.shape_cast %lhs_scale_byte_vec : vector<1x1x2x1x1x4xi8> to !lhs_scale_byte_vec_ty
      %lhs_scale_vec = vector.bitcast %lhs_scale_byte_vec_t : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty
      %rhs_scale_byte_vec = vector.transfer_read %rhs_scale_shared[%curr_buffer_num, %subgroup_ids#1, %ids#1, %ids#2, %c0],
        %cst_scale {in_bounds = [true, true, true, true, true]} : !rhs_scale_shared_ty, vector<1x1x1x1x4xi8>
      %rhs_scale_byte_vec_t = vector.shape_cast %rhs_scale_byte_vec : vector<1x1x1x1x4xi8> to !rhs_scale_byte_vec_ty
      %rhs_scale_vec = vector.bitcast %rhs_scale_byte_vec_t : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

      %dot = iree_codegen.inner_tiled ins(%lhs_vec, %rhs_vec, %lhs_scale_vec, %rhs_scale_vec) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = #iterator_types,
        kind = #mfma_type,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : !lhs_vec_ty, !rhs_vec_ty, !lhs_scale_vec_ty, !rhs_scale_vec_ty into !acc_ty

      rocdl.s.setprio 0
      amdgpu.lds_barrier
      rocdl.sched.barrier 0
      rocdl.s.waitcnt 0

      scf.yield %dot : !acc_ty
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    amdgpu.lds_barrier
    rocdl.s.waitcnt 0
    rocdl.s.barrier

    %prev_buffer_num = arith.andi %k, %c1 : index
    %final_buffer_num = arith.xori %prev_buffer_num, %c1 : index

    // Load inputs/scales from LDS.
    %lhs_byte_vec = vector.transfer_read %lhs_shared_base[%final_buffer_num, %subgroup_ids#0, %c0, %c0, %ids#1, %ids#2, %c0],
      %cst_lhs {in_bounds = [true, true, true, true, true, true, true]} : !lhs_shared_ty, vector<1x1x2x4x1x1x16xi8>
    %lhs_byte_vec_t = vector.shape_cast %lhs_byte_vec : vector<1x1x2x4x1x1x16xi8> to !lhs_byte_vec_ty
    %lhs_vec = vector.bitcast %lhs_byte_vec_t : !lhs_byte_vec_ty to !lhs_vec_ty
    %rhs_byte_vec = vector.transfer_read %rhs_shared_base[%final_buffer_num, %subgroup_ids#1, %c0, %ids#1, %ids#2, %c0],
      %cst_rhs {in_bounds = [true, true, true, true, true, true]} : !rhs_shared_ty, vector<1x1x4x1x1x16xi8>
    %rhs_byte_vec_t = vector.shape_cast %rhs_byte_vec : vector<1x1x4x1x1x16xi8> to !rhs_byte_vec_ty
    %rhs_vec = vector.bitcast %rhs_byte_vec_t : !rhs_byte_vec_ty to !rhs_vec_ty

    %lhs_scale_byte_vec = vector.transfer_read %lhs_scale_shared[%final_buffer_num, %subgroup_ids#0, %c0, %ids#1, %ids#2, %c0],
      %cst_scale {in_bounds = [true, true, true, true, true, true]} : !lhs_scale_shared_ty, vector<1x1x2x1x1x4xi8>
    %lhs_scale_byte_vec_t = vector.shape_cast %lhs_scale_byte_vec : vector<1x1x2x1x1x4xi8> to !lhs_scale_byte_vec_ty
    %lhs_scale_vec = vector.bitcast %lhs_scale_byte_vec_t : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty
    %rhs_scale_byte_vec = vector.transfer_read %rhs_scale_shared[%final_buffer_num, %subgroup_ids#1, %ids#1, %ids#2, %c0],
        %cst_scale {in_bounds = [true, true, true, true, true]} : !rhs_scale_shared_ty, vector<1x1x1x1x4xi8>
    %rhs_scale_byte_vec_t = vector.shape_cast %rhs_scale_byte_vec : vector<1x1x1x1x4xi8> to !rhs_scale_byte_vec_ty
    %rhs_scale_vec = vector.bitcast %rhs_scale_byte_vec_t : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

    %epilogue_dot = iree_codegen.inner_tiled ins(%lhs_vec, %rhs_vec, %lhs_scale_vec, %rhs_scale_vec) outs(%loop) {
      indexing_maps = #contraction_accesses,
      iterator_types = #iterator_types,
      kind = #mfma_type,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : !lhs_vec_ty, !rhs_vec_ty, !lhs_scale_vec_ty, !rhs_scale_vec_ty into !acc_ty

    %reduce_init = arith.constant dense<0.0> : !reduce_ty
    %reduce = vector.multi_reduction <add>, %epilogue_dot, %reduce_init [2] :
      !acc_ty to !reduce_ty

    %t = vector.shape_cast %reduce : !reduce_ty to !store_ty

    %empty = tensor.empty() : !tensor_store_ty
    %to_tensor = vector.transfer_write %t, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0]
      {in_bounds = [true, true, true, true, true, true, true, true]} : !store_ty, !tensor_store_ty

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %to_tensor into %out[%c0, %c0, %subgroup_ids#0, %subgroup_ids#1, %c0, %ids#1, %ids#2, %c0] [1, 1, 1, 1, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1]
        : !tensor_store_ty into !return_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : !return_ty
}
