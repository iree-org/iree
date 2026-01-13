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
//   - Source offset = lane_id * 4 (strided access across lanes)
//   - Dest offset = 0 (LDS write offset is implicit in gather_to_lds)
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
    // Transfer 1: linearOffset = 0 + lane_offset, delinearize into (4, 128)
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %[[LIN0]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128 + lane_offset
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR128:.+]]:2 = affine.delinearize_index %[[LIN128]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1], %[[DST]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256 + lane_offset
    // CHECK: %[[C256:.+]] = arith.constant 256 : index
    // CHECK: %[[LIN256:.+]] = arith.addi %[[C256]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR256:.+]]:2 = affine.delinearize_index %[[LIN256]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR256]]#0, %[[DELINEAR256]]#1], %[[DST]][%[[DELINEAR256]]#0, %[[DELINEAR256]]#1] : vector<4xf32>
    //
    // Transfer 4: linearOffset = 384 + lane_offset
    // CHECK: %[[C384:.+]] = arith.constant 384 : index
    // CHECK: %[[LIN384:.+]] = arith.addi %[[C384]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR384:.+]]:2 = affine.delinearize_index %[[LIN384]] into (4, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR384]]#0, %[[DELINEAR384]]#1], %[[DST]][%[[DELINEAR384]]#0, %[[DELINEAR384]]#1] : vector<4xf32>
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
//   - Source offset = lane_id * 2
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
    // Transfer 1: linearOffset = 0 + lane_offset
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %[[LIN0]] into (2, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<2xf16>
    //
    // Transfer 2: linearOffset = 64 + lane_offset
    // CHECK: %[[C64:.+]] = arith.constant 64 : index
    // CHECK: %[[LIN64:.+]] = arith.addi %[[C64]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR64:.+]]:2 = affine.delinearize_index %[[LIN64]] into (2, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR64]]#0, %[[DELINEAR64]]#1], %[[DST]][%[[DELINEAR64]]#0, %[[DELINEAR64]]#1] : vector<2xf16>
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
    // CHECK: %[[LIN_OFFSET:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[DELIN:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[LIN_OFFSET]] into (128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELIN]]], %[[DST]][%[[DELIN]]] : vector<4xf32>
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
    // Single transfer: linearOffset = 0 + lane_offset
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR:.+]]:2 = affine.delinearize_index %[[LIN0]] into (1, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR]]#0, %[[DELINEAR]]#1], %[[DST]][%[[DELINEAR]]#0, %[[DELINEAR]]#1] : vector<4xf32>
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
    // Transfer 1: linearOffset = 0 + lane_offset
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:3 = affine.delinearize_index %[[LIN0]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1, %[[DELINEAR0]]#2], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1, %[[DELINEAR0]]#2] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128 + lane_offset
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR128:.+]]:3 = affine.delinearize_index %[[LIN128]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1, %[[DELINEAR128]]#2], %[[DST]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1, %[[DELINEAR128]]#2] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256 + lane_offset
    // CHECK: %[[C256:.+]] = arith.constant 256 : index
    // CHECK: %[[LIN256:.+]] = arith.addi %[[C256]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR256:.+]]:3 = affine.delinearize_index %[[LIN256]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR256]]#0, %[[DELINEAR256]]#1, %[[DELINEAR256]]#2], %[[DST]][%[[DELINEAR256]]#0, %[[DELINEAR256]]#1, %[[DELINEAR256]]#2] : vector<4xf32>
    //
    // Transfer 4: linearOffset = 384 + lane_offset
    // CHECK: %[[C384:.+]] = arith.constant 384 : index
    // CHECK: %[[LIN384:.+]] = arith.addi %[[C384]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR384:.+]]:3 = affine.delinearize_index %[[LIN384]] into (2, 2, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR384]]#0, %[[DELINEAR384]]#1, %[[DELINEAR384]]#2], %[[DST]][%[[DELINEAR384]]#0, %[[DELINEAR384]]#1, %[[DELINEAR384]]#2] : vector<4xf32>
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
    // Transfer 1: linearOffset = 0 + lane_offset
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %[[LIN0]] into (2, 1024)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 1024 + lane_offset
    // CHECK: %[[C1024:.+]] = arith.constant 1024 : index
    // CHECK: %[[LIN1024:.+]] = arith.addi %[[C1024]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR1024:.+]]:2 = affine.delinearize_index %[[LIN1024]] into (2, 1024)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR1024]]#0, %[[DELINEAR1024]]#1], %[[DST]][%[[DELINEAR1024]]#0, %[[DELINEAR1024]]#1] : vector<4xf32>
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
    // Transfer 1: linearOffset = 0 + lane_offset, load row_indices[delinear#0]
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %[[LIN0]] into (2, 128)
    // CHECK: %[[LOADED_ROW0:.+]] = memref.load %[[IDX]][%[[DELINEAR0]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW0]], %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128 + lane_offset
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR128:.+]]:2 = affine.delinearize_index %[[LIN128]] into (2, 128)
    // CHECK: %[[LOADED_ROW1:.+]] = memref.load %[[IDX]][%[[DELINEAR128]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW1]], %[[DELINEAR128]]#1], %[[DST]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1] : vector<4xf32>
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
// Should lower to exactly 3.
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
    // Transfer 1: linearOffset = 0 + lane_offset
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %[[LIN0]] into (3, 128)
    // CHECK: %[[LOADED_ROW0:.+]] = memref.load %[[IDX]][%[[DELINEAR0]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW0]], %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128 + lane_offset
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR128:.+]]:2 = affine.delinearize_index %[[LIN128]] into (3, 128)
    // CHECK: %[[LOADED_ROW1:.+]] = memref.load %[[IDX]][%[[DELINEAR128]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW1]], %[[DELINEAR128]]#1], %[[DST]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256 + lane_offset
    // CHECK: %[[C256:.+]] = arith.constant 256 : index
    // CHECK: %[[LIN256:.+]] = arith.addi %[[C256]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR256:.+]]:2 = affine.delinearize_index %[[LIN256]] into (3, 128)
    // CHECK: %[[LOADED_ROW2:.+]] = memref.load %[[IDX]][%[[DELINEAR256]]#0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW2]], %[[DELINEAR256]]#1], %[[DST]][%[[DELINEAR256]]#0, %[[DELINEAR256]]#1] : vector<4xf32>
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
    // Transfer 1: elements [0, 128), tile offset = 0
    // CHECK: %[[LIN0:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[DELIN0:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[LIN0]] into (256)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELIN0]]], %[[DST]][%[[DELIN0]]] : vector<4xf32>
    //
    // Transfer 2: elements [128, 256), tile offset = 128
    // CHECK: %[[LIN1:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[DELIN1:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[LIN1]] into (256)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELIN1]]], %[[DST]][%[[DELIN1]]] : vector<4xf32>
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
    // Transfer 1: 128-bit DMA using LANE_OFFSET_4
    // CHECK: %[[LIN0:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET_4]]
    // CHECK: %[[DELIN0:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[LIN0]] into (160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELIN0]]], %[[DST]][%[[DELIN0]]] : vector<4xf32>
    //
    // Transfer 2: 32-bit DMA using LANE_OFFSET_1
    // CHECK: %[[LIN1:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET_1]]
    // CHECK: %[[DELIN1:[a-zA-Z0-9_]+]] = affine.delinearize_index %[[LIN1]] into (160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELIN1]]], %[[DST]][%[[DELIN1]]] : vector<1xf32>
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
    // Transfer 1: linearOffset = 0 + lane_offset_4, 128-bit DMA
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128 + lane_offset_4, 128-bit DMA
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[DELINEAR128:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1], %[[DST]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1] : vector<4xf32>
    //
    // Transfer 3: linearOffset = 256 + lane_offset_1, 32-bit DMA
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[DELINEAR256:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR256]]#0, %[[DELINEAR256]]#1], %[[DST]][%[[DELINEAR256]]#0, %[[DELINEAR256]]#1] : vector<1xf32>
    //
    // Transfer 4: linearOffset = 288 + lane_offset_1, 32-bit DMA
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: %[[DELINEAR288:.+]]:2 = affine.delinearize_index %{{.+}} into (2, 160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR288]]#0, %[[DELINEAR288]]#1], %[[DST]][%[[DELINEAR288]]#0, %[[DELINEAR288]]#1] : vector<1xf32>
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
    // Transfer 1: linearOffset = 0 + lane_offset
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %[[LIN0]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<4xf32>
    //
    // Transfer 2: linearOffset = 128 + lane_offset
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR128:.+]]:2 = affine.delinearize_index %[[LIN128]] into (4, 64)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1], %[[DST]][%[[DELINEAR128]]#0, %[[DELINEAR128]]#1] : vector<4xf32>
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
    // Outer iteration 0 (dim 0 = 0):
    // CHECK: %[[OUTER0:[a-zA-Z0-9_]+]] = arith.constant 0 : index
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[DELIN0:.+]]:2 = affine.delinearize_index %{{.+}} into (4, 32)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[OUTER0]], %[[DELIN0]]#0, %[[DELIN0]]#1], %[[DST]][%[[OUTER0]], %[[DELIN0]]#0, %[[DELIN0]]#1] : vector<4xf32>
    //
    // Outer iteration 1 (dim 0 = 1):
    // CHECK: %[[OUTER1:[a-zA-Z0-9_]+]] = arith.constant 1 : index
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: %[[DELIN1:.+]]:2 = affine.delinearize_index %{{.+}} into (4, 32)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[OUTER1]], %[[DELIN1]]#0, %[[DELIN1]]#1], %[[DST]][%[[OUTER1]], %[[DELIN1]]#0, %[[DELIN1]]#1] : vector<4xf32>
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
//
// For a 1D gather from source[1024] to dest[256] with 32 lanes:
//   - 32 lanes * 4 elements/lane = 128 elements per transfer
//   - 256 / 128 = 2 transfers needed
//   - Each lane loads indices[its_dest_position] for true per-element mapping
//
// Example for transfer 0:
//   - Lane 0:  linearOffset = 0 + 0 = 0,   srcIdx = indices[0]
//   - Lane 16: linearOffset = 0 + 64 = 64, srcIdx = indices[64]
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
    // Transfer 1: linearOffset = 0 + lane_offset, each lane loads its own index
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]] = affine.delinearize_index %[[LIN0]] into (256)
    // CHECK: %[[LOADED0:.+]] = memref.load %{{.+}}[%[[DELINEAR0]]]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[LOADED0]]], %{{.+}}[%[[DELINEAR0]]]
    //
    // Transfer 2: linearOffset = 128 + lane_offset
    // CHECK: %[[C128:.+]] = arith.constant 128 : index
    // CHECK: %[[LIN128:.+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR128:.+]] = affine.delinearize_index %[[LIN128]] into (256)
    // CHECK: %[[LOADED128:.+]] = memref.load %{{.+}}[%[[DELINEAR128]]]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[LOADED128]]], %{{.+}}[%[[DELINEAR128]]]
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
//   - delinearize(64) = (1, 0) → correctly spreads across rows
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
    // Transfer 1: linearOffset = 0 + lane_offset
    // For lane 16: 0 + 64 = 64 → delinearize → (1, 0)
    // CHECK: %[[C0:.+]] = arith.constant 0 : index
    // CHECK: %[[LIN0:.+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: %[[DELINEAR0:.+]]:2 = affine.delinearize_index %[[LIN0]] into (16, 64)
    // The source indices MUST use delinearized values (not lane_offset directly)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1], %[[DST]][%[[DELINEAR0]]#0, %[[DELINEAR0]]#1] : vector<4xf32>
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
