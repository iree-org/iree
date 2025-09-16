// RUN: iree-opt %s

// Element types

!lhs = f4E2M1FN
!rhs = f4E2M1FN

!scale_ty = f8E8M0FNU

// Inputs

// m = 8
// n = 64
// k = 2048

!lhs_ty = tensor<8x?x32x!lhs>
!rhs_ty = tensor<64x?x32x!rhs>

!lhs_byte_ty = tensor<8x?x16xi8>
!rhs_byte_ty = tensor<64x?x16xi8>

!lhs_global_ty = memref<8x?x16xi8, strided<[?, 16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
!rhs_global_ty = memref<64x?x16xi8, strided<[?, 16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>

!lhs_buffer_ty = memref<8x?x16xi8, strided<[?, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_ty = memref<64x?x16xi8, strided<[?, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

// 2 = double buffer
// 16 = padded m tile
// 64 = n tile
// 16 = outer k tile
// 64 = 128 / 2 = inner k tile / 2
!lhs_shared_ty = memref<2x8x16x64xi8, strided<[8320, 1040, 64, 1]>, #gpu.address_space<workgroup>>
!rhs_shared_ty = memref<2x64x16x64xi8, strided<[66560, 1040, 64, 1]>, #gpu.address_space<workgroup>>

// Padded by 4 dwords. Misaligns by 4 banks, but keep minimum alignment
// needed for b128 loads.
!lhs_padded_shared_ty = memref<2x8x1040xi8, #gpu.address_space<workgroup>>
!rhs_padded_shared_ty = memref<2x64x1040xi8, #gpu.address_space<workgroup>>

!lhs_flat_shared_ty = memref<2x8x1024xi8, strided<[8320, 1040, 1]>, #gpu.address_space<workgroup>>
!rhs_flat_shared_ty = memref<2x64x1024xi8, strided<[66560, 1040, 1]>, #gpu.address_space<workgroup>>

!lhs_shared_expand_ty = memref<2x1x8x16x64xi8, strided<[8320, 8320, 1040, 64, 1]>, #gpu.address_space<workgroup>>
!rhs_shared_expand_ty = memref<2x4x16x16x64xi8, strided<[66560, 16640, 1040, 64, 1]>, #gpu.address_space<workgroup>>

!flattened_lhs_shared_ty = memref<16640xi8, #gpu.address_space<workgroup>>

!lhs_copy_vec_ty = vector<16xi8>
!rhs_copy_vec_ty = vector<16xi8>

!lhs_byte_vec_ty = vector<1x1x8x16xi8>
!lhs_vec_ty = vector<1x1x8x32xf4E2M1FN>

!rhs_byte_vec_ty = vector<1x1x8x16xi8>
!rhs_vec_ty = vector<1x1x8x32xf4E2M1FN>

// Scales

!lhs_scale_ty = tensor<8x?x!scale_ty>
!rhs_scale_ty = tensor<64x?x!scale_ty>

!lhs_scale_byte_ty = tensor<8x?xi8>
!rhs_scale_byte_ty = tensor<64x?xi8>

!lhs_scale_global_ty = memref<8x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
!rhs_scale_global_ty = memref<64x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>

!lhs_scale_buffer_ty = memref<8x?xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_ty = memref<64x?xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_copy_vec_ty = vector<2xi8>
!rhs_scale_copy_vec_ty = vector<16xi8>

// 2 = double buffer
// 8 = m tile. Just read out of bounds nbd :)
// 128 = n tile
// 8 = outer k tile
// 4 = inner k tile / k block
//
// lhs gets an extra factor of 2 because data is padded to dwords on load.
!lhs_scale_shared_ty = memref<2x8x16x2x4xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_ty = memref<2x64x16x4xi8, #gpu.address_space<workgroup>>

!lhs_scale_shared_expand_ty = memref<2x1x8x16x2x4xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_expand_ty = memref<2x4x16x16x4xi8, #gpu.address_space<workgroup>>

!lhs_scale_byte_vec_ty = vector<1x1x8x1x1xi8>
!lhs_scale_vec_ty = vector<1x1x8x1x1xf8E8M0FNU>

!rhs_scale_byte_vec_ty = vector<1x1x8x1xi8>
!rhs_scale_vec_ty = vector<1x1x8x1xf8E8M0FNU>


!return_ty = tensor<8x64xf32>

// subgroup_m = 1 waves subgroup_n = 4 waves.
// 8 total waves, half loading data half computing.
!acc_ty = vector<1x1x1x4xf32>
!bc_acc_ty = vector<1x1x1x16xi8>
!reduce_ty = vector<1x2xf32>
!bc_reduce_ty = vector<1x8xi8>

// split factor = 2
// inner m tile = 8
// outer n tile = 2x2
// inner n tile = 4x4
// 4xi8 -> 1xf32 for bitcasting
!flattened_sg_reduce_ty = memref<4096xi8, #gpu.address_space<workgroup>>
!flat_sg_reduce_ty = memref<2x8x256xi8, #gpu.address_space<workgroup>>
!sg_reduce_ty = memref<2x8x2x2x4x16xi8, #gpu.address_space<workgroup>>

// TODO: Swap to col major smfma to vectorize along N and allow use of DPP to
// broadcast out padded stuff.
!tensor_store_ty = tensor<1x2xf32>

#contraction_accesses = [
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
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
  acc_elem_type = f32,
  col_major = true>

util.func @mmt_8x64_f4f4f32(
    %lhs_base: !lhs_ty,
    %rhs_base: !rhs_ty,
    %lhs_scale_base: !lhs_scale_ty,
    %rhs_scale_base: !rhs_scale_ty,
    %unused_acc: !return_ty) -> !return_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
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
  %cst_acc = arith.constant 0 : i8
  %lhs_shared_padded_base = memref.alloc() : !lhs_padded_shared_ty
  %rhs_shared_padded_base = memref.alloc() : !rhs_padded_shared_ty
  %lhs_scale_shared = memref.alloc() : !lhs_scale_shared_ty
  %rhs_scale_shared = memref.alloc() : !rhs_scale_shared_ty

  %lhs_shared_flat_base = memref.subview %lhs_shared_padded_base [0, 0, 0] [2, 8, 1024] [1, 1, 1]
    : !lhs_padded_shared_ty to !lhs_flat_shared_ty
  %rhs_shared_flat_base = memref.subview %rhs_shared_padded_base [0, 0, 0] [2, 64, 1024] [1, 1, 1]
    : !rhs_padded_shared_ty to !rhs_flat_shared_ty

  %lhs_shared_base = memref.expand_shape %lhs_shared_flat_base [[0], [1], [2, 3]]
    output_shape [2, 8, 16, 64] : !lhs_flat_shared_ty into !lhs_shared_ty
  %rhs_shared_base = memref.expand_shape %rhs_shared_flat_base [[0], [1], [2, 3]]
    output_shape [2, 64, 16, 64] : !rhs_flat_shared_ty into !rhs_shared_ty

  %lhs_shared_expand = memref.expand_shape %lhs_shared_base [[0], [1, 2], [3], [4]]
    output_shape [2, 1, 8, 16, 64] : !lhs_shared_ty into !lhs_shared_expand_ty
  %rhs_shared_expand = memref.expand_shape %rhs_shared_base [[0], [1, 2], [3], [4]]
    output_shape [2, 4, 16, 16, 64] : !rhs_shared_ty into !rhs_shared_expand_ty

  %lhs_scale_shared_expand = memref.expand_shape %lhs_scale_shared [[0], [1, 2], [3], [4], [5]]
    output_shape [2, 1, 8, 16, 2, 4] : !lhs_scale_shared_ty into !lhs_scale_shared_expand_ty
  %rhs_scale_shared_expand = memref.expand_shape %rhs_scale_shared [[0], [1, 2], [3], [4]]
    output_shape [2, 4, 16, 16, 4] : !rhs_scale_shared_ty into !rhs_scale_shared_expand_ty

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

    // Make the upper 4 waves start copying data.
    %cmp = arith.cmpi sge, %base_id, %c256 : index
    scf.if %cmp {
      %id = arith.subi %base_id, %c256 : index
      %sg = arith.divui %id, %c64 : index

      // Input load offsets.
      %ids:2 = affine.delinearize_index %id into (4, 64) : index, index
      %inner = arith.muli %ids#1, %c1 : index
      %outer_base = arith.muli %ids#0, %c1 : index

      %shared_inner_base = arith.muli %sg, %c1 : index

      // Scale load offsets.
      %rhs_scale_ids:2 = affine.delinearize_index %id into (64, 4) : index, index
      %rhs_scale_inner_base = arith.muli %rhs_scale_ids#1, %c16 : index
      %rhs_scale_outer = arith.muli %rhs_scale_ids#0, %c1 : index
      %rhs_scale_shared_inner = arith.muli %sg, %c16 : index

      %lhs_scale_ids:2 = affine.delinearize_index %id into (8, 32) : index, index
      %lhs_scale_inner_base = arith.muli %lhs_scale_ids#1, %c2 : index
      %lhs_scale_outer = arith.muli %lhs_scale_ids#0, %c1 : index
      %lhs_scale_shared_outer = arith.muli %sg, %c2 : index

      scf.for %i = %c0 to %k step %c64 {
        %shift = arith.shrui %i, %c6 : index
        %buffer_num = arith.andi %shift, %c1 : index
        %loop_inner = arith.addi %inner, %i : index

        // Copy lhs.
        // 2 x b128
        scf.for %j = %c0 to %c8 step %c4 {
          %shared_inner = arith.addi %shared_inner_base, %j : index
          %outer = arith.addi %outer_base, %j : index
          amdgpu.gather_to_lds %lhs[%outer, %loop_inner, %c0], %lhs_shared_base[%buffer_num, %shared_inner, %c0, %c0]
            : !lhs_copy_vec_ty, !lhs_buffer_ty, !lhs_shared_ty
        }

        // Copy scales.
        %rhs_loop_scale_inner = arith.addi %rhs_scale_inner_base, %i : index
        // 1 x b128
        amdgpu.gather_to_lds %rhs_scale[%rhs_scale_outer, %rhs_loop_scale_inner], %rhs_scale_shared[%buffer_num, %rhs_scale_shared_inner, %c0, %c0] {aux = 3 : i32}
          : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_ty, !rhs_scale_shared_ty

        %lhs_loop_scale_inner = arith.addi %lhs_scale_inner_base, %i : index
        // 1 x b16
        amdgpu.gather_to_lds
          %lhs_scale[%lhs_scale_outer, %lhs_loop_scale_inner],
          %lhs_scale_shared[%buffer_num, %lhs_scale_shared_outer, %c0, %c0, %c0]
          : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_ty, !lhs_scale_shared_ty

        // Copy half of rhs.
        // 8 x b128
        scf.for %j = %c0 to %c32 step %c4 {
          %shared_inner = arith.addi %shared_inner_base, %j : index
          %outer = arith.addi %outer_base, %j : index
          amdgpu.gather_to_lds %rhs[%outer, %loop_inner, %c0], %rhs_shared_base[%buffer_num, %shared_inner, %c0, %c0] {aux = 3 : i32}
            : !rhs_copy_vec_ty, !rhs_buffer_ty, !rhs_shared_ty
        }

        // Wait on previous iteration's group.
        rocdl.s.waitcnt 12
        rocdl.s.barrier

        // Copy other half of rhs.
        // 8 x b128
        scf.for %j = %c32 to %c64 step %c4 {
          %shared_inner = arith.addi %shared_inner_base, %j : index
          %outer = arith.addi %outer_base, %j : index
          amdgpu.gather_to_lds %rhs[%outer, %loop_inner, %c0], %rhs_shared_base[%buffer_num, %shared_inner, %c0, %c0] {aux = 3 : i32}
            : !rhs_copy_vec_ty, !rhs_buffer_ty, !rhs_shared_ty
        }

        // Wait on previous group.
        rocdl.s.waitcnt 8
        rocdl.s.barrier

        scf.yield
      }

      // Realign subgroups and wait on the last iteration.
      rocdl.s.waitcnt 0
      rocdl.s.barrier

      // One extra for the local reduce. Whether we can skip this barrier
      // depends on whether logical and physical workgroups map 1-1.
      rocdl.s.barrier
    }

  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : !return_ty
  %1 = scf.forall (%id) in (256) shared_outs(%out = %0) -> !return_ty {
    %init0 = arith.constant dense<0.0> : !acc_ty
    %init1 = arith.constant dense<0.0> : !acc_ty

    %mfma_ids:5 = affine.delinearize_index %id into (1, 2, 2, 4, 16) : index, index, index, index, index
    %m_id = arith.muli %mfma_ids#0, %c1 : index
    %n_id = arith.muli %mfma_ids#1, %c1 : index
    %n_id_plus2 = arith.addi %n_id, %c2 : index
    %k_id = arith.muli %mfma_ids#2, %c8 : index
    %inner_lane_offset = arith.muli %mfma_ids#4, %c1 : index
    %outer_lane_offset = arith.muli %mfma_ids#3, %c16 : index

    %lhs_inner_lane_offset = arith.remui %inner_lane_offset, %c8 : index

    %oob = arith.cmpi sge, %mfma_ids#4, %c8 : index
    %keep = arith.constant dense<255> : !lhs_byte_vec_ty
    %discard = arith.constant dense<0> : !lhs_byte_vec_ty
    %mask = arith.select %oob, %discard, %keep : !lhs_byte_vec_ty

    %scale_keep = arith.constant dense<255> : !lhs_scale_byte_vec_ty
    %scale_discard = arith.constant dense<0> : !lhs_scale_byte_vec_ty
    %scale_mask = arith.select %oob, %scale_discard, %scale_keep : !lhs_scale_byte_vec_ty

    // Misalign by one group.
    rocdl.s.barrier

    %loop:2 = scf.for %i = %c0 to %k step %c64
      iter_args(%iter0 = %init0, %iter1 = %init1) -> (!acc_ty, !acc_ty) {
      %shift = arith.shrui %i, %c6 : index
      %buffer_num = arith.andi %shift, %c1 : index

      // wait till first half is available.
      rocdl.s.barrier

      // Load inputs/scales from LDS.
      %lhs_byte_vec = vector.transfer_read %lhs_shared_expand[%buffer_num, %m_id, %lhs_inner_lane_offset, %k_id, %outer_lane_offset],
        %cst_lhs {in_bounds = [true, true, true, true]} : !lhs_shared_expand_ty, !lhs_byte_vec_ty
      %lhs_mask_vec = arith.andi %lhs_byte_vec, %mask : !lhs_byte_vec_ty
      %lhs_vec = vector.bitcast %lhs_mask_vec : !lhs_byte_vec_ty to !lhs_vec_ty

      %lhs_scale_split:2 = affine.delinearize_index %mfma_ids#3 into (2, 2) : index, index
      %lhs_scale_byte_vec = vector.transfer_read
        %lhs_scale_shared_expand[%buffer_num, %m_id, %lhs_inner_lane_offset, %k_id, %lhs_scale_split#0, %lhs_scale_split#1],
        %cst_scale {in_bounds = [true, true, true, true, true]} : !lhs_scale_shared_expand_ty, !lhs_scale_byte_vec_ty
      %lhs_scale_mask_vec = arith.andi %lhs_scale_byte_vec, %scale_mask : !lhs_scale_byte_vec_ty
      %lhs_scale_vec = vector.bitcast %lhs_scale_mask_vec : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty

      %rhs_byte_vec_0 = vector.transfer_read %rhs_shared_expand[%buffer_num, %n_id, %inner_lane_offset, %k_id, %outer_lane_offset],
        %cst_rhs {in_bounds = [true, true, true, true]} : !rhs_shared_expand_ty, !rhs_byte_vec_ty
      %rhs_vec_0 = vector.bitcast %rhs_byte_vec_0 : !rhs_byte_vec_ty to !rhs_vec_ty

      %rhs_scale_byte_vec_0 = vector.transfer_read %rhs_scale_shared_expand[%buffer_num, %n_id, %inner_lane_offset, %k_id, %mfma_ids#3],
        %cst_scale {in_bounds = [true, true, true, true]} : !rhs_scale_shared_expand_ty, !rhs_scale_byte_vec_ty
      %rhs_scale_vec_0 = vector.bitcast %rhs_scale_byte_vec_0 : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

      %rhs_scale_byte_vec_1 = vector.transfer_read %rhs_scale_shared_expand[%buffer_num, %n_id_plus2, %inner_lane_offset, %k_id, %mfma_ids#3],
        %cst_scale {in_bounds = [true, true, true, true]} : !rhs_scale_shared_expand_ty, !rhs_scale_byte_vec_ty
      %rhs_scale_vec_1 = vector.bitcast %rhs_scale_byte_vec_1 : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec, %lhs_scale_vec, %rhs_vec_0, %rhs_scale_vec_0) outs(%iter0) {
        indexing_maps = #contraction_accesses,
        iterator_types = #iterator_types,
        kind = #mfma_type
      } : !lhs_vec_ty, !lhs_scale_vec_ty, !rhs_vec_ty, !rhs_scale_vec_ty into !acc_ty

      // Wait till second half is available.
      amdgpu.lds_barrier

      %rhs_byte_vec_1 = vector.transfer_read %rhs_shared_expand[%buffer_num, %n_id_plus2, %inner_lane_offset, %k_id, %outer_lane_offset],
        %cst_rhs {in_bounds = [true, true, true, true]} : !rhs_shared_expand_ty, !rhs_byte_vec_ty
      %rhs_vec_1 = vector.bitcast %rhs_byte_vec_1 : !rhs_byte_vec_ty to !rhs_vec_ty

      %dot1 = iree_codegen.inner_tiled ins(%lhs_vec, %lhs_scale_vec, %rhs_vec_1, %rhs_scale_vec_1) outs(%iter1) {
        indexing_maps = #contraction_accesses,
        iterator_types = #iterator_types,
        kind = #mfma_type
      } : !lhs_vec_ty, !lhs_scale_vec_ty, !rhs_vec_ty, !rhs_scale_vec_ty into !acc_ty

      scf.yield %dot0, %dot1 : !acc_ty, !acc_ty
    }

    %o0 = vector.extract %loop#0[0, 0, 0, 0] : f32 from !acc_ty
    %o1 = vector.extract %loop#0[0, 0, 0, 1] : f32 from !acc_ty
    %o2 = vector.extract %loop#0[0, 0, 0, 2] : f32 from !acc_ty
    %o3 = vector.extract %loop#0[0, 0, 0, 3] : f32 from !acc_ty

    %s0 = vector.extract %loop#1[0, 0, 0, 0] : f32 from !acc_ty
    %s1 = vector.extract %loop#1[0, 0, 0, 1] : f32 from !acc_ty
    %s2 = vector.extract %loop#1[0, 0, 0, 2] : f32 from !acc_ty
    %s3 = vector.extract %loop#1[0, 0, 0, 3] : f32 from !acc_ty

    %dpp0 = amdgpu.dpp %o0 %s0 row_shr ( 0x8 : i32 ) { bank_mask = 12 : i32 } : f32
    %dpp1 = amdgpu.dpp %o1 %s1 row_shr ( 0x8 : i32 ) { bank_mask = 12 : i32 } : f32
    %dpp2 = amdgpu.dpp %o2 %s2 row_shr ( 0x8 : i32 ) { bank_mask = 12 : i32 } : f32
    %dpp3 = amdgpu.dpp %o3 %s3 row_shr ( 0x8 : i32 ) { bank_mask = 12 : i32 } : f32

    %shuffle_init = arith.constant dense<0.0> : !acc_ty
    %i0 = vector.insert %dpp0, %shuffle_init[0, 0, 0, 0] : f32 into !acc_ty
    %i1 = vector.insert %dpp1, %i0[0, 0, 0, 1] : f32 into !acc_ty
    %i2 = vector.insert %dpp2, %i1[0, 0, 0, 2] : f32 into !acc_ty
    %i3 = vector.insert %dpp3, %i2[0, 0, 0, 3] : f32 into !acc_ty
    %i3_bc = vector.bitcast %i3 : !acc_ty to !bc_acc_ty

    // Reuse the lhs because we're done with it at this point.
    %lhs_collapse_shape = memref.collapse_shape %lhs_shared_padded_base[[0, 1, 2]] : !lhs_padded_shared_ty into !flattened_lhs_shared_ty
    %flat_subgroup_reduce = memref.subview %lhs_collapse_shape [0] [4096] [1]
      : !flattened_lhs_shared_ty to !flattened_sg_reduce_ty

    %subgroup_reduce = memref.expand_shape %flat_subgroup_reduce [[0, 1, 2]]
    output_shape [2, 8, 256] : !flattened_sg_reduce_ty into !flat_sg_reduce_ty
    %subgroup_reduce_expand = memref.expand_shape %subgroup_reduce [[0], [1], [2, 3, 4, 5]]
    output_shape [2, 8, 2, 2, 4, 16] : !flat_sg_reduce_ty into !sg_reduce_ty
    %reduce_ids:5 = affine.delinearize_index %id into (2, 2, 4, 2, 8) : index, index, index, index, index
    vector.transfer_write %i3_bc, %subgroup_reduce_expand[%reduce_ids#1, %reduce_ids#4, %reduce_ids#3, %reduce_ids#0, %reduce_ids#2, %c0]
      {in_bounds = [true, true, true, true]} : !bc_acc_ty, !sg_reduce_ty
    amdgpu.lds_barrier

    %store_ids:2 = affine.delinearize_index %id into (8, 32) : index, index
    %bc_inner_id = arith.muli %store_ids#1, %c8 : index
    %outer_id = arith.muli %store_ids#0, %c1 : index
    %left_i8 = vector.transfer_read %subgroup_reduce[%c0, %outer_id, %bc_inner_id],
      %cst_acc {in_bounds = [true, true]} : !flat_sg_reduce_ty, !bc_reduce_ty
    %right_i8 = vector.transfer_read %subgroup_reduce[%c1, %outer_id, %bc_inner_id],
      %cst_acc {in_bounds = [true, true]} : !flat_sg_reduce_ty, !bc_reduce_ty
    %left = vector.bitcast %left_i8 : !bc_reduce_ty to !reduce_ty
    %right = vector.bitcast %right_i8 : !bc_reduce_ty to !reduce_ty
    %reduce = arith.addf %left, %right : !reduce_ty

    %empty = tensor.empty() : !tensor_store_ty
    %to_tensor = vector.transfer_write %reduce, %empty[%c0, %c0]
      {in_bounds = [true, true]} : !reduce_ty, !tensor_store_ty

    %inner_id = arith.muli %store_ids#1, %c2 : index
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %to_tensor into
        %out[%outer_id, %inner_id] [1, 2] [1, 1]
        : !tensor_store_ty into !return_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : !return_ty
}
