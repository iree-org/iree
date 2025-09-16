// RUN: iree-opt %s

// Element types

!lhs = f4E2M1FN
!rhs = f4E2M1FN

!scale_ty = f8E8M0FNU

// Inputs

// m = 8
// n = 128
// k = 1024

!lhs_ty = tensor<8x?x32x!lhs>
!rhs_ty = tensor<128x?x32x!rhs>

!lhs_byte_ty = tensor<8x?x16xi8>
!rhs_byte_ty = tensor<128x?x16xi8>

!lhs_global_ty = memref<8x?x16xi8, strided<[?, 16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
!rhs_global_ty = memref<128x?x16xi8, strided<[?, 16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>

!lhs_buffer_ty = memref<8x?x16xi8, strided<[?, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_ty = memref<128x?x16xi8, strided<[?, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

// 2 = double buffer
// 16 = padded m tile
// 128 = n tile
// 8 = outer k tile
// 64 = 128 / 2 = inner k tile / 2
!lhs_shared_ty = memref<2x16x8x64xi8, #gpu.address_space<workgroup>>
!rhs_shared_ty = memref<2x128x8x64xi8, #gpu.address_space<workgroup>>

!lhs_shared_expand_ty = memref<2x1x16x8x64xi8, #gpu.address_space<workgroup>>
!rhs_shared_expand_ty = memref<2x8x16x8x64xi8, #gpu.address_space<workgroup>>

!lhs_copy_vec_ty = vector<16xi8>
!rhs_copy_vec_ty = vector<16xi8>

!lhs_byte_vec_ty = vector<1x1x8x16xi8>
!lhs_vec_ty = vector<1x1x8x32xf4E2M1FN>

!rhs_byte_vec_ty = vector<1x1x8x16xi8>
!rhs_vec_ty = vector<1x1x8x32xf4E2M1FN>

// Scales

!lhs_scale_ty = tensor<8x?x!scale_ty>
!rhs_scale_ty = tensor<128x?x!scale_ty>

!lhs_scale_byte_ty = tensor<8x?xi8>
!rhs_scale_byte_ty = tensor<128x?xi8>

!lhs_scale_global_ty = memref<8x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
!rhs_scale_global_ty = memref<128x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>

!lhs_scale_buffer_ty = memref<8x?xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_ty = memref<128x?xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_copy_vec_ty = vector<1xi8>
!rhs_scale_copy_vec_ty = vector<16xi8>

// 2 = double buffer
// 16 = padded m tile
// 128 = n tile
// 8 = outer k tile
// 4 = inner k tile / k block
//
// lhs gets an extra factor of 4 because data is padded to dwords on load.
!lhs_scale_shared_ty = memref<2x16x8x4x4xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_ty = memref<2x128x8x4xi8, #gpu.address_space<workgroup>>

!lhs_scale_shared_expand_ty = memref<2x1x16x8x4x4xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_expand_ty = memref<2x8x16x8x4xi8, #gpu.address_space<workgroup>>

!lhs_scale_byte_vec_ty = vector<1x1x8x1x1xi8>
!lhs_scale_vec_ty = vector<1x1x8x1x1xf8E8M0FNU>

!rhs_scale_byte_vec_ty = vector<1x1x8x1xi8>
!rhs_scale_vec_ty = vector<1x1x8x1xf8E8M0FNU>


!res_ty = tensor<1x8x8x16xf32>
!return_ty = tensor<8x128xf32>

// subgroup_m = 1 waves subgroup_n = 4 waves.
// 8 total waves, half loading data half computing.
// Unrolled by a factor of 2 for shuffling.
!acc_ty = vector<1x1x1x4xf32>
!shuffle_ty = vector<2xi64>

// TODO: Swap to col major smfma to vectorize along N and allow use of DPP to
// broadcast out padded stuff.
!tensor_store_ty = tensor<1x1x1x4xf32>

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

#result_reassoc = [[0, 1], [2, 3]]

util.func @mmt_8x128_f4f4f32(
    %lhs_base: !lhs_ty,
    %rhs_base: !rhs_ty,
    %lhs_scale_base: !lhs_scale_ty,
    %rhs_scale_base: !rhs_scale_ty,
    %unused_acc: !return_ty) -> !return_ty {
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

  %lhs_shared_expand = memref.expand_shape %lhs_shared_base [[0], [1, 2], [3], [4]]
    output_shape [2, 1, 16, 8, 64] : !lhs_shared_ty into !lhs_shared_expand_ty
  %rhs_shared_expand = memref.expand_shape %rhs_shared_base [[0], [1, 2], [3], [4]]
    output_shape [2, 8, 16, 8, 64] : !rhs_shared_ty into !rhs_shared_expand_ty

  %lhs_scale_shared_expand = memref.expand_shape %lhs_scale_shared [[0], [1, 2], [3], [4], [5]]
    output_shape [2, 1, 16, 8, 4, 4] : !lhs_scale_shared_ty into !lhs_scale_shared_expand_ty
  %rhs_scale_shared_expand = memref.expand_shape %rhs_scale_shared [[0], [1, 2], [3], [4]]
    output_shape [2, 8, 16, 8, 4] : !rhs_scale_shared_ty into !rhs_scale_shared_expand_ty

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

    // Zerofill lhs inputs and scales.
    %zids:3 = affine.delinearize_index %base_id into (16, 8, 4) : index, index, index
    %zinner = arith.muli %zids#2, %c16 : index
    %v0_i8 = arith.constant dense<0> : vector<2x1x1x16xi8>
    vector.transfer_write %v0_i8, %lhs_shared_base[%c0, %zids#0, %zids#1, %zinner] {in_bounds = [true, true, true, true]} : vector<2x1x1x16xi8>, !lhs_shared_ty
    %v0_i8_scales = arith.constant dense<127> : vector<2x1x1x1x4xi8>
    vector.transfer_write %v0_i8_scales, %lhs_scale_shared[%c0, %zids#0, %zids#1, %zids#2, %c0] {in_bounds = [true, true, true, true, true]} : vector<2x1x1x1x4xi8>, !lhs_scale_shared_ty
    amdgpu.lds_barrier

    // Make the upper 4 waves start copying data.
    %cmp = arith.cmpi sge, %base_id, %c256 : index
    scf.if %cmp {
      %id = arith.subi %base_id, %c256 : index
      %sg = arith.divui %id, %c64 : index

      // Input load offsets.
      %ids:2 = affine.delinearize_index %id into (8, 32) : index, index
      %inner = arith.muli %ids#1, %c1 : index
      %outer_base = arith.muli %ids#0, %c1 : index

      %shared_inner_base = arith.muli %sg, %c2 : index

      // Scale load offsets.
      %rhs_scale_ids:2 = affine.delinearize_index %id into (128, 2) : index, index
      %rhs_scale_inner_base = arith.muli %rhs_scale_ids#1, %c16 : index
      %rhs_scale_outer = arith.muli %rhs_scale_ids#0, %c1 : index
      %rhs_scale_shared_inner = arith.muli %sg, %c32 : index

      %lhs_scale_ids:2 = affine.delinearize_index %id into (8, 32) : index, index
      %lhs_scale_inner_base = arith.muli %lhs_scale_ids#1, %c1 : index
      %lhs_scale_outer = arith.muli %lhs_scale_ids#0, %c1 : index
      %lhs_scale_shared_inner = arith.muli %sg, %c2 : index

      scf.for %i = %c0 to %k step %c32 {
        %shift = arith.shrui %i, %c5 : index
        %buffer_num = arith.andi %shift, %c1 : index
        %loop_inner = arith.addi %inner, %i : index

        // Copy lhs.
        // 1 x b128
        amdgpu.gather_to_lds %lhs[%outer_base, %loop_inner, %c0], %lhs_shared_base[%buffer_num, %shared_inner_base, %c0, %c0]
          : !lhs_copy_vec_ty, !lhs_buffer_ty, !lhs_shared_ty

        // Copy scales.
        %rhs_loop_scale_inner = arith.addi %rhs_scale_inner_base, %i : index
        // 1 x b128
        amdgpu.gather_to_lds %rhs_scale[%rhs_scale_outer, %rhs_loop_scale_inner], %rhs_scale_shared[%buffer_num, %rhs_scale_shared_inner, %c0, %c0] {aux = 3 : i32}
          : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_ty, !rhs_scale_shared_ty
        %lhs_loop_scale_inner = arith.addi %lhs_scale_inner_base, %i : index
        // 1 x b8
        amdgpu.gather_to_lds %lhs_scale[%lhs_scale_outer, %lhs_loop_scale_inner], %lhs_scale_shared[%buffer_num, %lhs_scale_shared_inner, %c0, %c0, %c0]
          : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_ty, !lhs_scale_shared_ty

        // Copy half of rhs.
        // 8 x b128
        scf.for %j = %c0 to %c64 step %c8 {
          %shared_inner = arith.addi %shared_inner_base, %j : index
          %outer = arith.addi %outer_base, %j : index
          amdgpu.gather_to_lds %rhs[%outer, %loop_inner, %c0], %rhs_shared_base[%buffer_num, %shared_inner, %c0, %c0] {aux = 3 : i32}
            : !rhs_copy_vec_ty, !rhs_buffer_ty, !rhs_shared_ty
        }

        // Wait on previous iteration's group.
        rocdl.s.waitcnt 11
        rocdl.s.barrier

        // Copy other half of rhs.
        // 8 x b128
        scf.for %j = %c64 to %c128 step %c8 {
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

      // Realign subgroups and wait on the last group.
      rocdl.s.waitcnt 0
      rocdl.s.barrier
    }

  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : !res_ty
  %1 = scf.forall (%id) in (256) shared_outs(%out = %0) -> !res_ty {
    %init0 = arith.constant dense<0.0> : !acc_ty
    %init1 = arith.constant dense<0.0> : !acc_ty

    %mfma_ids:4 = affine.delinearize_index %id into (1, 4, 4, 16) : index, index, index, index
    %m_id = arith.muli %mfma_ids#0, %c2 : index
    %n_id = arith.muli %mfma_ids#1, %c1 : index
    %n_id_plus4 = arith.addi %n_id, %c4 : index
    %inner_lane_offset = arith.muli %mfma_ids#3, %c1 : index
    %outer_lane_offset = arith.muli %mfma_ids#2, %c16 : index

    // Misalign by one group.
    rocdl.s.barrier

    %loop:2 = scf.for %i = %c0 to %k step %c32
      iter_args(%iter0 = %init0, %iter1 = %init1) -> (!acc_ty, !acc_ty) {
      %shift = arith.shrui %i, %c5 : index
      %buffer_num = arith.andi %shift, %c1 : index

      // wait till first half is available.
      rocdl.s.barrier

      // Load inputs/scales from LDS.
      %lhs_byte_vec = vector.transfer_read %lhs_shared_expand[%buffer_num, %m_id, %inner_lane_offset, %c0, %outer_lane_offset],
        %cst_lhs {in_bounds = [true, true, true, true]} : !lhs_shared_expand_ty, !lhs_byte_vec_ty
      %lhs_vec = vector.bitcast %lhs_byte_vec : !lhs_byte_vec_ty to !lhs_vec_ty
      %lhs_scale_byte_vec = vector.transfer_read %lhs_scale_shared_expand[%buffer_num, %m_id, %inner_lane_offset, %c0, %mfma_ids#2, %c0],
        %cst_scale {in_bounds = [true, true, true, true, true]} : !lhs_scale_shared_expand_ty, !lhs_scale_byte_vec_ty
      %lhs_scale_vec = vector.bitcast %lhs_scale_byte_vec : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty

      %rhs_byte_vec_0 = vector.transfer_read %rhs_shared_expand[%buffer_num, %n_id, %inner_lane_offset, %c0, %outer_lane_offset],
        %cst_rhs {in_bounds = [true, true, true, true]} : !rhs_shared_expand_ty, !rhs_byte_vec_ty
      %rhs_vec_0 = vector.bitcast %rhs_byte_vec_0 : !rhs_byte_vec_ty to !rhs_vec_ty

      %rhs_scale_byte_vec_0 = vector.transfer_read %rhs_scale_shared_expand[%buffer_num, %n_id, %inner_lane_offset, %c0, %mfma_ids#2],
        %cst_scale {in_bounds = [true, true, true, true]} : !rhs_scale_shared_expand_ty, !rhs_scale_byte_vec_ty
      %rhs_scale_vec_0 = vector.bitcast %rhs_scale_byte_vec_0 : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

      %rhs_scale_byte_vec_1 = vector.transfer_read %rhs_scale_shared_expand[%buffer_num, %n_id_plus4, %inner_lane_offset, %c0, %mfma_ids#2],
        %cst_scale {in_bounds = [true, true, true, true]} : !rhs_scale_shared_expand_ty, !rhs_scale_byte_vec_ty
      %rhs_scale_vec_1 = vector.bitcast %rhs_scale_byte_vec_1 : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec, %lhs_scale_vec, %rhs_vec_0, %rhs_scale_vec_0) outs(%iter0) {
        indexing_maps = #contraction_accesses,
        iterator_types = #iterator_types,
        kind = #mfma_type
      } : !lhs_vec_ty, !lhs_scale_vec_ty, !rhs_vec_ty, !rhs_scale_vec_ty into !acc_ty

      // Wait till second half is available.
      amdgpu.lds_barrier

      %rhs_byte_vec_1 = vector.transfer_read %rhs_shared_expand[%buffer_num, %n_id_plus4, %inner_lane_offset, %c0, %outer_lane_offset],
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

    %final = arith.constant dense<0.0> : !acc_ty
    %i0 = vector.insert %dpp0, %final[0, 0, 0, 0] : f32 into !acc_ty
    %i1 = vector.insert %dpp1, %i0[0, 0, 0, 1] : f32 into !acc_ty
    %i2 = vector.insert %dpp2, %i1[0, 0, 0, 2] : f32 into !acc_ty
    %i3 = vector.insert %dpp3, %i2[0, 0, 0, 3] : f32 into !acc_ty

    %empty = tensor.empty() : !tensor_store_ty
    %to_tensor = vector.transfer_write %i3, %empty[%c0, %c0, %c0, %c0]
      {in_bounds = [true, true, true, true]} : !acc_ty, !tensor_store_ty

    %store_mfma_ids:4 = affine.delinearize_index %id into (4, 4, 2, 8) : index, index, index, index
    %store_lane_outer = arith.muli %store_mfma_ids#2, %c4 : index
    %store_outer_n_offset = arith.addi %store_mfma_ids#0, %store_lane_outer : index
    %store_inner_n_offset = arith.muli %store_mfma_ids#1, %c4 : index
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %to_tensor into
        %out[%c0, %store_mfma_ids#3, %store_outer_n_offset, %store_inner_n_offset] [1, 1, 1, 4] [1, 1, 1, 1]
        : !tensor_store_ty into !res_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %collapse = "tensor.collapse_shape"(%1) {reassociation = #result_reassoc} : (!res_ty) -> (!return_ty)
  util.return %collapse : !return_ty
}
