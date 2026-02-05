// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds))" \
// RUN:   --verify-diagnostics %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_64 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: 4x128 memref with 32 lanes.
//   - Elements per lane = 128 / 32 = 4 (each lane reads 4 contiguous f32s)
//   - Source offset = divergent (includes lane_id * 4)
//   - Dest offset = uniform (excludes lane offset, subgroup-uniform for gather_to_lds)
//   - Loop iterations = 4 (one gather_to_lds per row)
//
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_multiple
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<4x128xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<4x128xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_gather_dma_multiple(
    %source: memref<4x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<4x128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_64} {
  // CHECK: scf.forall (%[[LANE_ID:.+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Each lane reads 4 elements, so lane offset = lane_id * 4.
    // laneOffset is precomputed once and reused across all rows.
    // CHECK: %[[C4:.+]] = arith.constant 4 : index
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: linearOffset = 0, src gets + lane_offset, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (4, 128)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[SRC_LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN128:.+]]:2 = affine.delinearize_index %[[SRC_LIN128]] into (4, 128)
    // CHECK: %[[DST_DELIN128:.+]]:2 = affine.delinearize_index %[[C128]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN128]]#0, %[[SRC_DELIN128]]#1], %[[DST]][%[[DST_DELIN128]]#0, %[[DST_DELIN128]]#1] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256
    // CHECK: %[[C256:.+]] = arith.constant 256 : index
    // CHECK: %[[SRC_LIN256:.+]] = arith.addi %[[C256]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN256:.+]]:2 = affine.delinearize_index %[[SRC_LIN256]] into (4, 128)
    // CHECK: %[[DST_DELIN256:.+]]:2 = affine.delinearize_index %[[C256]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN256]]#0, %[[SRC_DELIN256]]#1], %[[DST]][%[[DST_DELIN256]]#0, %[[DST_DELIN256]]#1] : vector<4xf32>
    //
    // Transfer 4: linearOffset = 384
    // CHECK: %[[C384:.+]] = arith.constant 384 : index
    // CHECK: %[[SRC_LIN384:.+]] = arith.addi %[[C384]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN384:.+]]:2 = affine.delinearize_index %[[SRC_LIN384]] into (4, 128)
    // CHECK: %[[DST_DELIN384:.+]]:2 = affine.delinearize_index %[[C384]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN384]]#0, %[[SRC_DELIN384]]#1], %[[DST]][%[[DST_DELIN384]]#0, %[[DST_DELIN384]]#1] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
      memref<4x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<4x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: 2x64 memref with 32 lanes.
//   - Elements per lane = 64 / 32 = 2 (each lane reads 2 contiguous f16s)
//   - Source offset = divergent (lane_id * 2)
//   - Dest offset = uniform (subgroup-uniform for gather_to_lds)
//   - Loop iterations = 2 (one gather_to_lds per row)
//
// CHECK-LABEL: func.func @lower_coalesced_copy_dma_basic
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x64xf16, #gpu.address_space<workgroup>>
func.func @lower_coalesced_copy_dma_basic(
    %source: memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x64xf16, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:.+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Each lane reads 2 elements, so lane offset = lane_id * 2.
    // laneOffset is precomputed once and reused across all rows.
    // CHECK: %[[C2:.+]] = arith.constant 2 : index
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C2]]
    //
    // Transfer 1: linearOffset = 0, src gets +lane_offset, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (2, 64)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (2, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<2xf16>
    //
    // Transfer 2: linearOffset = 64
    // CHECK: %[[C64:.+]] = arith.constant 64 : index
    // CHECK: %[[SRC_LIN64:.+]] = arith.addi %[[C64]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN64:.+]]:2 = affine.delinearize_index %[[SRC_LIN64]] into (2, 64)
    // CHECK: %[[DST_DELIN64:.+]]:2 = affine.delinearize_index %[[C64]] into (2, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN64]]#0, %[[SRC_DELIN64]]#1], %[[DST]][%[[DST_DELIN64]]#0, %[[DST_DELIN64]]#1] : vector<2xf16>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<2x64xf16, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: 1D memref with 128 elements and 32 lanes.
//   * Elements per lane = 128 / 32 = 4 (each lane reads 4 contiguous f32s)
//   * Only one gather_to_lds op
//
// CHECK-LABEL: func.func @lower_coalesced_copy_dma_1d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<128xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<128xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_copy_dma_1d(
    %source: memref<128xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // CHECK: %[[C0:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_LIN:[a-zA-Z0-9_]+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[SRC_LIN]] into (128)
    // CHECK: %[[DST_DELIN:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[C0]] into (128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN]]], %[[DST]][%[[DST_DELIN]]] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: Single-row 2D memref (1x128) with 32 lanes.
// This verifies that a 2D memref with only 1 row produces a single transfer,
// equivalent to the 1D case. The delinearization should still work correctly.
//   * Elements per lane = 128 / 32 = 4 (each lane reads 4 contiguous f32s)
//   * Only one gather_to_lds op (single row = single transfer)
//   * Source indices: divergent, Dest indices: uniform
//
// CHECK-LABEL: func.func @lower_coalesced_copy_dma_single_row_2d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<1x128xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<1x128xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_copy_dma_single_row_2d(
    %source: memref<1x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<1x128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Single transfer: src gets +lane_offset, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN:.+]]:2 = affine.delinearize_index %[[SRC_LIN]] into (1, 128)
    // CHECK: %[[DST_DELIN:.+]]:2 = affine.delinearize_index %[[C0]] into (1, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN]]#0, %[[SRC_DELIN]]#1], %[[DST]][%[[DST_DELIN]]#0, %[[DST_DELIN]]#1] : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<1x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<1x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: 3D memref with shape 2x2x128 and 32 lanes.
//   * Elements per lane = 128 / 32 = 4
//   * Tile sizes = [1, 1, 128], iterate over 2*2 = 4 tiles
//   * 4 gather_to_lds ops total
//
// CHECK-LABEL: func.func @lower_coalesced_copy_dma_3d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x2x128xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x2x128xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_copy_dma_3d(
    %source: memref<2x2x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x2x128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // laneOffset is precomputed once and reused across all tiles.
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: linearOffset = 0, src gets +lane_offset, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:3 = affine.delinearize_index %[[SRC_LIN0]] into (2, 2, 128)
    // CHECK: %[[DST_DELIN0:.+]]:3 = affine.delinearize_index %[[C0]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1, %[[SRC_DELIN0]]#2], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1, %[[DST_DELIN0]]#2] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[SRC_LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN128:.+]]:3 = affine.delinearize_index %[[SRC_LIN128]] into (2, 2, 128)
    // CHECK: %[[DST_DELIN128:.+]]:3 = affine.delinearize_index %[[C128]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN128]]#0, %[[SRC_DELIN128]]#1, %[[SRC_DELIN128]]#2], %[[DST]][%[[DST_DELIN128]]#0, %[[DST_DELIN128]]#1, %[[DST_DELIN128]]#2] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256
    // CHECK: %[[C256:.+]] = arith.constant 256 : index
    // CHECK: %[[SRC_LIN256:.+]] = arith.addi %[[C256]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN256:.+]]:3 = affine.delinearize_index %[[SRC_LIN256]] into (2, 2, 128)
    // CHECK: %[[DST_DELIN256:.+]]:3 = affine.delinearize_index %[[C256]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN256]]#0, %[[SRC_DELIN256]]#1, %[[SRC_DELIN256]]#2], %[[DST]][%[[DST_DELIN256]]#0, %[[DST_DELIN256]]#1, %[[DST_DELIN256]]#2] : vector<4xf32>
    //
    // Transfer 4: linearOffset = 384
    // CHECK: %[[C384:.+]] = arith.constant 384 : index
    // CHECK: %[[SRC_LIN384:.+]] = arith.addi %[[C384]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN384:.+]]:3 = affine.delinearize_index %[[SRC_LIN384]] into (2, 2, 128)
    // CHECK: %[[DST_DELIN384:.+]]:3 = affine.delinearize_index %[[C384]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN384]]#0, %[[SRC_DELIN384]]#1, %[[SRC_DELIN384]]#2], %[[DST]][%[[DST_DELIN384]]#0, %[[DST_DELIN384]]#1, %[[DST_DELIN384]]#2] : vector<4xf32>
    //
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x2x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x2x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb_wide = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [256, 256],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_256_wide = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 256>

// Test: Wide forall with 256 lanes, 2D memref 2x1024.
//   * subgroup_size = 256
//   * elementsPerTransfer = 1024 / 256 = 4 (128 bits per lane)
//   * tileSizes = [1, 1024]
//   * 2 rows * 1 tile per row = 2 gather_to_lds ops
//   * Source indices: divergent, Dest indices: uniform
//
// CHECK-LABEL: func.func @lower_coalesced_copy_dma_wide_forall_2d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x1024xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x1024xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_copy_dma_wide_forall_2d(
    %source: memref<2x1024xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x1024xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb_wide,
    translation_info = #translation_256_wide} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (256)
  scf.forall (%arg6) in (256) {
    // elementsPerTransfer = 256 * 4 = 1024 = 1 row
    // laneOffset is precomputed once and reused across all rows.
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: linearOffset = 0, src gets +lane_offset, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (2, 1024)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (2, 1024)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 1024
    // CHECK: %[[C1024:.+]] = arith.constant 1024 : index
    // CHECK: %[[SRC_LIN1024:.+]] = arith.addi %[[C1024]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN1024:.+]]:2 = affine.delinearize_index %[[SRC_LIN1024]] into (2, 1024)
    // CHECK: %[[DST_DELIN1024:.+]]:2 = affine.delinearize_index %[[C1024]] into (2, 1024)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN1024]]#0, %[[SRC_DELIN1024]]#1], %[[DST]][%[[DST_DELIN1024]]#0, %[[DST_DELIN1024]]#1] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x1024xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x1024xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test gather 2D with indices.
// Source indices use srcDimOffset (with lane offset) for index lookup.
// Destination indices use dstDimOffset (uniform, no lane offset).
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_with_indices
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<1024x128xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[IDX:[a-zA-Z0-9]+]]: memref<2xindex>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x128xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_gather_dma_with_indices(
    %source: memref<1024x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %row_indices: memref<2xindex>,
    %dest: memref<2x128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // laneOffset is precomputed once and reused across all rows.
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: load row_indices using srcDimOffset (with lane offset)
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (2, 128)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (2, 128)
    // CHECK: %[[LOADED_ROW0:.+]] = memref.load %[[IDX]][%[[SRC_DELIN0]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW0]], %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[SRC_LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN128:.+]]:2 = affine.delinearize_index %[[SRC_LIN128]] into (2, 128)
    // CHECK: %[[DST_DELIN128:.+]]:2 = affine.delinearize_index %[[C128]] into (2, 128)
    // CHECK: %[[LOADED_ROW1:.+]] = memref.load %[[IDX]][%[[SRC_DELIN128]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW1]], %[[SRC_DELIN128]]#1], %[[DST]][%[[DST_DELIN128]]#0, %[[DST_DELIN128]]#1] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source[%row_indices] into %dest lane(%arg6) : memref<1024x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2xindex>, memref<2x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: Verify iteration uses destShape, not sourceShape.
// If code iterated over sourceShape, would generate 256 ops.
// Should lower to exactly 3. Source indices: divergent, Dest indices: uniform.
//
// CHECK-LABEL: func.func @gather_iterates_over_dest_shape_not_source
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<256x128xf32
// CHECK-SAME:    %[[IDX:[a-zA-Z0-9]+]]: memref<3xindex>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<3x128xf32
func.func @gather_iterates_over_dest_shape_not_source(
    %source: memref<256x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %row_indices: memref<3xindex>,
    %dest: memref<3x128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // laneOffset is precomputed once and reused across all rows.
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: linearOffset = 0, src has lane offset, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (3, 128)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (3, 128)
    // CHECK: %[[LOADED_ROW0:.+]] = memref.load %[[IDX]][%[[SRC_DELIN0]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW0]], %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[SRC_LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN128:.+]]:2 = affine.delinearize_index %[[SRC_LIN128]] into (3, 128)
    // CHECK: %[[DST_DELIN128:.+]]:2 = affine.delinearize_index %[[C128]] into (3, 128)
    // CHECK: %[[LOADED_ROW1:.+]] = memref.load %[[IDX]][%[[SRC_DELIN128]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW1]], %[[SRC_DELIN128]]#1], %[[DST]][%[[DST_DELIN128]]#0, %[[DST_DELIN128]]#1] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256
    // CHECK: %[[C256:.+]] = arith.constant 256 : index
    // CHECK: %[[SRC_LIN256:.+]] = arith.addi %[[C256]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN256:.+]]:2 = affine.delinearize_index %[[SRC_LIN256]] into (3, 128)
    // CHECK: %[[DST_DELIN256:.+]]:2 = affine.delinearize_index %[[C256]] into (3, 128)
    // CHECK: %[[LOADED_ROW2:.+]] = memref.load %[[IDX]][%[[SRC_DELIN256]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW2]], %[[SRC_DELIN256]]#1], %[[DST]][%[[DST_DELIN256]]#0, %[[DST_DELIN256]]#1] : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source[%row_indices] into %dest lane(%arg6) : memref<256x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<3xindex>, memref<3x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: N-transfer mode for wide innermost dimension.
// When innermostDimSize > subgroupSize * elementsPerLane, multiple
// GatherToLDS ops are generated to cover the entire dimension.
//   - innermostDimSize = 256, subgroupSize = 32, dma_sizes = [128]
//   - elementsPerLane = 128 bits / 32 bits = 4 f32s per lane
//   - totalElementsPerTransfer = 32 * 4 = 128 elements
//   - numTransfers = 256 / 128 = 2 transfers
//
// CHECK-LABEL: func.func @lower_coalesced_dma_multiple_transfers_1d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<256xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_multiple_transfers_1d(
    %source: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<256xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Each lane reads 4 elements per transfer.
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: elements [0, 128), src has lane offset, dst is uniform
    // CHECK: %[[SRC_LIN0:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[SRC_LIN0]] into (256)
    // CHECK: %[[DST_DELIN0:[a-zA-Z0-9_]+]] = affine.delinearize_index %{{.+}} into (256)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]], %[[DST]][%[[DST_DELIN0]]] : vector<4xf32>
    //
    // Transfer 2: elements [128, 256), tile offset = 128
    // CHECK: %[[SRC_LIN1:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN1:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[SRC_LIN1]] into (256)
    // CHECK: %[[DST_DELIN1:[a-zA-Z0-9_]+]] = affine.delinearize_index %{{.+}} into (256)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN1]]], %[[DST]][%[[DST_DELIN1]]] : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<256xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: N-transfer mode for 2D memref with wide innermost dimension.
//   - Shape: 2x256, innermostDimSize = 256
//   - subgroupSize = 32, dma_sizes = [128]
//   - elementsPerLane = 4, totalElementsPerTransfer = 128
//   - numTransfers per row = 256 / 128 = 2
//   - Total gather_to_lds ops = 2 rows * 2 transfers = 4
//
// CHECK-LABEL: func.func @lower_coalesced_dma_multiple_transfers_2d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x256xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x256xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_multiple_transfers_2d(
    %source: memref<2x256xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x256xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0, Transfer 1: [0, 0:128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 0, Transfer 2: [0, 128:256)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 1, Transfer 1: [1, 0:128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 1, Transfer 2: [1, 128:256)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x256xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x256xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: Mixed DMA sizes within the same row.
// When innermost dimension isn't evenly divisible by the largest DMA size,
// use a combination of sizes to cover the entire dimension.
//   - Shape: 160 f32s (640 bytes)
//   - subgroupSize = 32, dma_sizes = [32, 128]
//   - 128-bit (4 f32s/lane): 32*4=128 elements/transfer, 160/128=1 transfer
//   - Remaining: 160-128=32 elements
//   - 32-bit (1 f32/lane): 32*1=32 elements/transfer, 32/32=1 transfer
//   - Total: 2 transfers (1×128-bit + 1×32-bit)
//
// CHECK-LABEL: func.func @lower_coalesced_dma_mixed_sizes_1d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<160xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<160xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_mixed_sizes_1d(
    %source: memref<160xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<160xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Transfer 1: 128-bit DMA, elements [0, 128)
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET_4:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer 2: 32-bit DMA, elements [128, 160)
    // CHECK: %[[C1:[a-zA-Z0-9_]+]] = arith.constant 1
    // CHECK: %[[LANE_OFFSET_1:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C1]]
    //
    // Transfer 1: 128-bit DMA, src has lane offset, dst is uniform
    // CHECK: %[[SRC_LIN0:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET_4]]
    // CHECK: %[[SRC_DELIN0:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[SRC_LIN0]] into (160)
    // CHECK: %[[DST_DELIN0:[a-zA-Z0-9_]+]] = affine.delinearize_index %{{.+}} into (160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]], %[[DST]][%[[DST_DELIN0]]] : vector<4xf32>
    //
    // Transfer 2: 32-bit DMA using LANE_OFFSET_1
    // CHECK: %[[SRC_LIN1:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET_1]]
    // CHECK: %[[SRC_DELIN1:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[SRC_LIN1]] into (160)
    // CHECK: %[[DST_DELIN1:[a-zA-Z0-9_]+]] = affine.delinearize_index %{{.+}} into (160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN1]]], %[[DST]][%[[DST_DELIN1]]] : vector<1xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<160xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<160xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: Mixed DMA sizes with 2D contiguous memref (linearized path).
// Since the destination is fully contiguous, the linearized transfer path is
// used, which issues all large DMA transfers first, then smaller ones.
//   - Shape: 2x160 f32s = 320 total elements
//   - subgroupSize = 32, dma_sizes = [32, 128]
//   - 128-bit DMA (4 f32s/lane): 32*4=128 elements/transfer, 320/128=2, rem=64
//   - 32-bit DMA (1 f32/lane): 32*1=32 elements/transfer, 64/32=2
//   - Total: 4 transfers (2×vector<4xf32> + 2×vector<1xf32>)
//
// CHECK-LABEL: func.func @lower_coalesced_dma_mixed_sizes_2d
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x160xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x160xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_mixed_sizes_2d(
    %source: memref<2x160xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x160xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Linearized path: all 128-bit transfers first, then 32-bit transfers.
    // Transfer 1: src has lane_offset_4, dst is uniform
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[SRC_DELIN128:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: %[[DST_DELIN128:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN128]]#0, %[[SRC_DELIN128]]#1], %[[DST]][%[[DST_DELIN128]]#0, %[[DST_DELIN128]]#1] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[SRC_DELIN256:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: %[[DST_DELIN256:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN256]]#0, %[[SRC_DELIN256]]#1], %[[DST]][%[[DST_DELIN256]]#0, %[[DST_DELIN256]]#1] : vector<1xf32>
    //
    // Transfer 4: linearOffset = 288
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[SRC_DELIN288:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: %[[DST_DELIN288:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN288]]#0, %[[SRC_DELIN288]]#1], %[[DST]][%[[DST_DELIN288]]#0, %[[DST_DELIN288]]#1] : vector<1xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x160xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x160xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: Linearized transfer spanning 2 rows per transfer.
// When destination is fully contiguous, the entire memref is treated as a 1D
// array and transfers can span multiple rows.
//   - Shape: 4x64 f32s = 256 total elements
//   - subgroupSize = 32, dma_sizes = [128]
//   - elementsPerLane = 128 bits / 32 bits = 4 f32s per lane
//   - elementsPerTransfer = 32 * 4 = 128 elements = 2 rows (64 elements/row)
//   - numTransfers = 256 / 128 = 2 transfers
//   - Transfer 0 covers rows 0-1 (linear elements 0-127)
//   - Transfer 1 covers rows 2-3 (linear elements 128-255)
//
// This tests the linearized transfer optimization that allows larger DMA
// transfers to span multiple rows when the destination is contiguous.
//
// CHECK-LABEL: func.func @lower_coalesced_dma_linearized_2_rows_per_transfer
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<4x64xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<4x64xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_linearized_2_rows_per_transfer(
    %source: memref<4x64xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<4x64xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Each lane transfers 4 elements.
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: linearOffset = 0, src has lane offset, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (4, 64)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[SRC_LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN128:.+]]:2 = affine.delinearize_index %[[SRC_LIN128]] into (4, 64)
    // CHECK: %[[DST_DELIN128:.+]]:2 = affine.delinearize_index %[[C128]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN128]]#0, %[[SRC_DELIN128]]#1], %[[DST]][%[[DST_DELIN128]]#0, %[[DST_DELIN128]]#1] : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<4x64xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<4x64xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb_3d = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [128, 32]>>}>

#translation_3d = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test with 3D shape where only innermost 2 dims are contiguous.
// Dest shape: <2x4x32xf32> with strided<[256, 32, 1]>
//   - dim 0 stride = 256, but product of dims 1-2 = 4*32 = 128, so NOT contiguous
//   - dim 1 stride = 32 = 32*1 (dim 2 size * stride), contiguous
//   - dim 2 stride = 1, contiguous
// numContiguousTrailingDims = 2 (dims 1-2), numLinearDims = 2
// linearSize = 4 * 32 = 128 elements
//
// With subgroupSize=32, dma_sizes=[128, 32]:
//   - 128-bit DMA: 4 elem/lane * 32 lanes = 128 = linearSize, 1 transfer
// Total: 2 outer iterations (dim 0) × 1 transfer = 2 GatherToLDS ops
//
// This tests the hybrid linearization where only trailing contiguous dims
// are linearized, while outer non-contiguous dims are iterated separately.
//
// CHECK-LABEL: func.func @lower_3d_partial_contiguous
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x4x32xf32, strided<[256, 32, 1]>, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x4x32xf32, strided<[256, 32, 1]>, #gpu.address_space<workgroup>>
func.func @lower_3d_partial_contiguous(
    %source: memref<2x4x32xf32, strided<[256, 32, 1]>, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x4x32xf32, strided<[256, 32, 1]>, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb_3d,
    translation_info = #translation_3d} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // 128-bit DMA: 4 elements per lane
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Outer iteration 0 (dim 0 = 0): src has lane offset, dst is uniform
    // CHECK: %[[OUTER0:[a-zA-Z0-9_]+]] = arith.constant 0 : index
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %{{.+}} into (4, 32)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %{{.+}} into (4, 32)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[OUTER0]], %[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[OUTER0]], %[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Outer iteration 1 (dim 0 = 1):
    // CHECK: %[[OUTER1:[a-zA-Z0-9_]+]] = arith.constant 1 : index
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN1:.+]]:2 = affine.delinearize_index %{{.+}} into (4, 32)
    // CHECK: %[[DST_DELIN1:.+]]:2 = affine.delinearize_index %{{.+}} into (4, 32)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[OUTER1]], %[[SRC_DELIN1]]#0, %[[SRC_DELIN1]]#1], %[[DST]][%[[OUTER1]], %[[DST_DELIN1]]#0, %[[DST_DELIN1]]#1] : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
      memref<2x4x32xf32, strided<[256, 32, 1]>, #amdgpu.address_space<fat_raw_buffer>>,
      memref<2x4x32xf32, strided<[256, 32, 1]>, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb_3d_mixed = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [128, 32]>>}>

#translation_3d_mixed = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test with 3D shape where innermost 2 dims require mixed DMA sizes.
// Dest shape: <2x5x32xf32> with strided<[256, 32, 1]>
//   - dim 0 stride = 256, but product of dims 1-2 = 5*32 = 160, so NOT contiguous
//   - dim 1 stride = 32 = 32*1 (dim 2 size * stride), contiguous
//   - dim 2 stride = 1, contiguous
// numContiguousTrailingDims = 2 (dims 1-2), numLinearDims = 2
// linearSize = 5 * 32 = 160 elements
//
// With subgroupSize=32, dma_sizes=[128, 32]:
//   - 128-bit DMA: 4 elem/lane * 32 lanes = 128, 1 transfer (covers 128 elements)
//   - 32-bit DMA: 1 elem/lane * 32 lanes = 32, 1 transfer (covers remaining 32)
// Per outer iteration: 2 transfers (one 128-bit at offset 0, one 32-bit at offset 128)
// Total: 2 outer iterations × 2 transfers = 4 GatherToLDS ops
//
// CHECK-LABEL: func.func @lower_3d_partial_contiguous_mixed_dma
func.func @lower_3d_partial_contiguous_mixed_dma(
    %source: memref<2x5x32xf32, strided<[256, 32, 1]>, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x5x32xf32, strided<[256, 32, 1]>, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb_3d_mixed,
    translation_info = #translation_3d_mixed} {
  // CHECK: scf.forall
  scf.forall (%arg6) in (32) {
    // Outer iteration 0: 128-bit DMA (vector<4xf32>)
    // CHECK: affine.delinearize_index %{{.+}} into (5, 32)
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<4xf32>
    // Outer iteration 0: 32-bit DMA (vector<1xf32>)
    // CHECK: affine.delinearize_index %{{.+}} into (5, 32)
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<1xf32>
    // Outer iteration 1: 128-bit DMA (vector<4xf32>)
    // CHECK: affine.delinearize_index %{{.+}} into (5, 32)
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<4xf32>
    // Outer iteration 1: 32-bit DMA (vector<1xf32>)
    // CHECK: affine.delinearize_index %{{.+}} into (5, 32)
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<1xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
      memref<2x5x32xf32, strided<[256, 32, 1]>, #amdgpu.address_space<fat_raw_buffer>>,
      memref<2x5x32xf32, strided<[256, 32, 1]>, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test gather with indices on the innermost (only) dimension.
// This tests true per-element gather where each lane loads its own index.
// Source indices: divergent (with lane offset for index lookup)
// Dest indices: uniform (subgroup-uniform for gather_to_lds)
//
// For a 1D gather from source[1024] to dest[256] with 32 lanes:
//   - 32 lanes * 4 elements/lane = 128 elements per transfer
//   - 256 / 128 = 2 transfers needed
//   - Each lane loads indices[srcDimOffset] (with lane offset) for source lookup
//   - Dest uses uniform offset (dstDimOffset, no lane offset)
//
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_innermost_1d
func.func @lower_coalesced_gather_dma_innermost_1d(
    %source: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %col_indices: memref<256xindex>,
    %dest: memref<256xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: src has lane offset for index lookup, dst is uniform
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]] = affine.delinearize_index %[[SRC_LIN0]] into (256)
    // CHECK: %[[DST_DELIN0:.+]] = affine.delinearize_index %[[C0]] into (256)
    // CHECK: %[[LOADED0:.+]] = memref.load %{{.+}}[%[[SRC_DELIN0]]]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[LOADED0]]], %{{.+}}[%[[DST_DELIN0]]]
    //
    // Transfer 2: linearOffset = 128
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[SRC_LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN128:.+]] = affine.delinearize_index %[[SRC_LIN128]] into (256)
    // CHECK: %[[DST_DELIN128:.+]] = affine.delinearize_index %[[C128]] into (256)
    // CHECK: %[[LOADED128:.+]] = memref.load %{{.+}}[%[[SRC_DELIN128]]]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[LOADED128]]], %{{.+}}[%[[DST_DELIN128]]]
    //
    // CHECK-NOT: amdgpu.gather_to_lds
    iree_gpu.coalesced_gather_dma %source[%col_indices] into %dest lane(%arg6) :
      memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<256xindex>,
      memref<256xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

// Test: Critical regression test for lane offset incorporation.
// This tests the fix for the bug where lane offset was incorrectly added to
// the innermost dimension after delinearization, causing out-of-bounds access.
//
// Shape: 16x64 f32s, 32 lanes, 4 elements per lane = 128 elements per transfer
// Without the fix (buggy):
//   - delinearize(0) = (0, 0)
//   - lane 16 src_col = 0 + 16*4 = 64 → OUT OF BOUNDS (innermost dim = 64)
//
// With the fix (correct):
//   - lane 16: linearOffset = 0 + 16*4 = 64
//   - delinearize(64) = (1, 0) → correctly spreads source access across rows
//   - dest uses uniform offset (no lane offset) for subgroup-uniform LDS access
//
// This is critical for matmul operations where innermost (K) dim < 64 lanes × 4 elements.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// CHECK-LABEL: func.func @lower_coalesced_dma_lane_offset_regression
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<16x64xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<16x64xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_lane_offset_regression(
    %source: memref<16x64xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<16x64xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // 128 elements per transfer = 2 rows of 64
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: src has lane_offset (divergent), dst is uniform
    // For lane 16: srcLinear = 0 + 64 = 64 → delinearize → (1, 0)
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (16, 64)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (16, 64)
    // Source uses divergent indices, dest uses uniform indices
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    //
    // Transfer 2-8: similar pattern for remaining 896 elements
    // CHECK-COUNT-7: amdgpu.gather_to_lds {{.+}} : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
      memref<16x64xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<16x64xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

// Test: coalesced_gather_dma with in_bounds attribute (tensor.pad fusion case).
// When in_bounds = [false, true], the source dim 0 can differ from dest dim 0.
// This happens when tensor.pad is fused - source is the pre-padded tensor,
// dest is the padded shape. Hardware OOB returns 0 for reads beyond source bounds.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// CHECK-LABEL: func.func @lower_coalesced_dma_with_in_bounds
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x128xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<4x128xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_with_in_bounds(
    %source: memref<2x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<4x128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // Source is 2x128 (pre-padded), dest is 4x128 (padded).
  // in_bounds = [false, true]: dim 0 may OOB (padding), dim 1 is in-bounds.
  // Lowering uses dest shape (4x128) to compute transfer pattern.
  // Reads beyond source row 2 will return 0 via hardware OOB.
  //
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // 4 rows * 128 cols = 512 elements total, 4 elements per lane = 4 transfers
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: row 0
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<4xf32>
    // Transfer 2: row 1
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<4xf32>
    // Transfer 3: row 2 (OOB in source, hardware returns 0)
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<4xf32>
    // Transfer 4: row 3 (OOB in source, hardware returns 0)
    // CHECK: amdgpu.gather_to_lds {{.+}} : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) in_bounds [false, true] :
      memref<2x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<4x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

// Test: coalesced_gather_dma with in_bounds for unaligned matmul tensor.pad fusion.
// This tests the exact pattern from unaligned matmul (65x64x121):
//   - RHS slice shape: 4x64 (K-tile x N-dim)
//   - 64 lanes (one subgroup)
//   - in_bounds = [false, true]: K-dim may OOB (last tile 121 % 4 = 1), N-dim is aligned
//
// With 64 lanes and 4x64 dest shape:
//   - Elements per lane = 64 / 64 = 1 (each lane reads 1 f32)
//   - Delinearization basis = (4, 64)
//   - 4 transfers per lane (one per row)
//
// This verifies correct row access pattern: all 4 rows (0-3) are accessed,
// not just row 0 repeated 4 times (which was the bug before the fix).

#executable_target_rocm_hsaco_fb_unaligned = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = shuffle, dot = none, mma = [],
    subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32]>>}>

#translation_64_unaligned = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @lower_coalesced_dma_4x64_tensor_pad_fusion
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<?x64xf32, strided<[64, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<4x64xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_4x64_tensor_pad_fusion(
    %source: memref<?x64xf32, strided<[64, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<4x64xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb_unaligned,
    translation_info = #translation_64_unaligned} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (64)
  scf.forall (%arg6) in (64) {
    // Each lane reads 1 element (64 elements / 64 lanes = 1).
    // CHECK: %[[C1:[a-zA-Z0-9_]+]] = arith.constant 1 : index
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C1]]
    //
    // 4 transfers with delinearization basis (4, 64):
    // Transfer 1: linearOffset = 0, accesses row 0
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (4, 64)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN0]]#0, %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<1xf32>
    //
    // Transfer 2: linearOffset = 64, accesses row 1
    // CHECK: %[[C64:.+]] = arith.constant 64 : index
    // CHECK: %[[SRC_LIN64:.+]] = arith.addi %[[C64]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN64:.+]]:2 = affine.delinearize_index %[[SRC_LIN64]] into (4, 64)
    // CHECK: %[[DST_DELIN64:.+]]:2 = affine.delinearize_index %[[C64]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN64]]#0, %[[SRC_DELIN64]]#1], %[[DST]][%[[DST_DELIN64]]#0, %[[DST_DELIN64]]#1] : vector<1xf32>
    //
    // Transfer 3: linearOffset = 128, accesses row 2
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[SRC_LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN128:.+]]:2 = affine.delinearize_index %[[SRC_LIN128]] into (4, 64)
    // CHECK: %[[DST_DELIN128:.+]]:2 = affine.delinearize_index %[[C128]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN128]]#0, %[[SRC_DELIN128]]#1], %[[DST]][%[[DST_DELIN128]]#0, %[[DST_DELIN128]]#1] : vector<1xf32>
    //
    // Transfer 4: linearOffset = 192, accesses row 3
    // CHECK: %[[C192:.+]] = arith.constant 192 : index
    // CHECK: %[[SRC_LIN192:.+]] = arith.addi %[[C192]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN192:.+]]:2 = affine.delinearize_index %[[SRC_LIN192]] into (4, 64)
    // CHECK: %[[DST_DELIN192:.+]]:2 = affine.delinearize_index %[[C192]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_DELIN192]]#0, %[[SRC_DELIN192]]#1], %[[DST]][%[[DST_DELIN192]]#0, %[[DST_DELIN192]]#1] : vector<1xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) in_bounds [false, true] :
      memref<?x64xf32, strided<[64, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>,
      memref<4x64xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

// Test: Non-outermost dimension padding with in_bounds = [false, false].
// Source: 4x6, dest: 4x8. Dim 1 has padding (6 → 8).
// Raw buffer OOB is linear/1D, so for non-outermost dim OOB, we must
// replace the outermost index with sourceShape[0] to force hardware OOB.
//
// Without the fix: reading at [0, 6] computes a byte offset within the
// buffer and wraps to [1, 0] instead of returning 0.
// With the fix: when srcIndices[1] >= 6, srcIndices[0] is replaced with 4
// (source dim 0 size), guaranteeing linear offset >= buffer size → returns 0.

#executable_target_rocm_hsaco_fb_pad = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32]>>}>

#translation_32_pad = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// CHECK-LABEL: func.func @gather_dma_non_outermost_oob_check
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<4x6xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<4x8xf32, #gpu.address_space<workgroup>>
func.func @gather_dma_non_outermost_oob_check(
    %source: memref<4x6xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<4x8xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb_pad,
    translation_info = #translation_32_pad} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[C1:[a-zA-Z0-9_]+]] = arith.constant 1
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C1]]
    //
    // Transfer 1: linearOffset = 0
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (4, 8)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (4, 8)
    //
    // Bounds check: compare srcIndices[1] >= 6 (source dim 1 size)
    // CHECK: %[[C6:.+]] = arith.constant 6 : index
    // CHECK: %[[OOB:.+]] = arith.cmpi uge, %[[SRC_DELIN0]]#1, %[[C6]] : index
    // Replace outermost index with 4 (source dim 0 size) to force hardware OOB
    // CHECK: %[[C4_OOB:.+]] = arith.constant 4 : index
    // CHECK: %[[FIXED_IDX:.+]] = arith.select %[[OOB]], %[[C4_OOB]], %[[SRC_DELIN0]]#0 : index
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[FIXED_IDX]], %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<1xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) in_bounds [false, false] :
      memref<4x6xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<4x8xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

// Test: Inner-dim padding OOB check with <64x62xf32> source padded to <64x64xf32>.
// Only inner dim (dim 1) has padding: 62 → 64. in_bounds = [true, false].
// Raw buffer OOB is 1D (linear): reading <4 x f32> at [0, 60] would compute a
// linear offset within the buffer and wrap to [1, 0], [1, 1] instead of returning 0.
// Fix: when srcIndices[1] >= 62, replace srcIndices[0] with 64 (past buffer end)
// so the linearized offset exceeds buffer size → hardware returns 0.

#executable_target_rocm_hsaco_fb_inner_pad = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_64_inner_pad = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @gather_dma_inner_dim_oob_64x62
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<64x62xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<64x64xf32, #gpu.address_space<workgroup>>
func.func @gather_dma_inner_dim_oob_64x62(
    %source: memref<64x62xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<64x64xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb_inner_pad,
    translation_info = #translation_64_inner_pad} {
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (64)
  scf.forall (%arg6) in (64) {
    // Each lane transfers vector<4xf32> (dma_sizes [128] = 128 bits = 4 x f32).
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: linearOffset = 0
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[SRC_DELIN0:.+]]:2 = affine.delinearize_index %[[SRC_LIN0]] into (64, 64)
    // CHECK: %[[DST_DELIN0:.+]]:2 = affine.delinearize_index %[[C0]] into (64, 64)
    //
    // Bounds check: compare srcIndices[1] >= 62 (source inner dim size).
    // CHECK: %[[C62:.+]] = arith.constant 62 : index
    // CHECK: %[[OOB:.+]] = arith.cmpi uge, %[[SRC_DELIN0]]#1, %[[C62]] : index
    // Replace outermost index with 64 (source dim 0 size) to force hardware OOB.
    // CHECK: %[[C64_OOB:.+]] = arith.constant 64 : index
    // CHECK: %[[FIXED_IDX:.+]] = arith.select %[[OOB]], %[[C64_OOB]], %[[SRC_DELIN0]]#0 : index
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[FIXED_IDX]], %[[SRC_DELIN0]]#1], %[[DST]][%[[DST_DELIN0]]#0, %[[DST_DELIN0]]#1] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) in_bounds [true, false] :
      memref<64x62xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<64x64xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}
