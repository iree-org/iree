// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds,canonicalize))" \
// RUN:   --verify-diagnostics %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:.+]]) in (32)
  scf.forall (%arg6) in (32) {
    // lane_offset = lane_id * 4
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer for each row: src[row, lane_offset], dst[row, 0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]]] : vector<4xf32>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C2]], %[[LANE_OFFSET]]], %[[DST]][%[[C2]], %[[C0]]] : vector<4xf32>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C3]], %[[LANE_OFFSET]]], %[[DST]][%[[C3]], %[[C0]]] : vector<4xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: 2x64 memref with 32 lanes.
//   - Elements per lane = 64 / 32 = 2 (each lane reads 2 contiguous f16s)
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: scf.forall (%[[LANE_ID:.+]]) in (32)
  scf.forall (%arg6) in (32) {
    // lane_offset = lane_id * 2
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C2]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<2xf16>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]]] : vector<2xf16>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<2x64xf16, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LANE_OFFSET]]], %[[DST]][%[[C0]]] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Single transfer: src[0, lane_offset], dst[0, 0]
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // 4 transfers for [0,0], [0,1], [1,0], [1,1]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]], %[[C0]]] : vector<4xf32>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[C1]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C1]], %[[C0]]] : vector<4xf32>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]], %[[C0]]] : vector<4xf32>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %[[C1]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C1]], %[[C0]]] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x2x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x2x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb_wide = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [256, 256],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (256)
  scf.forall (%arg6) in (256) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]]] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x1024xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x1024xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test gather 2D: load row indices from index memref
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer 1: load index[0], gather from source[loaded_idx, lane_offset]
    // CHECK: %[[LOADED0:.+]] = memref.load %[[IDX]][%[[C0]]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
    // Transfer 2: load index[1], gather from source[loaded_idx, lane_offset]
    // CHECK: %[[LOADED1:.+]] = memref.load %[[IDX]][%[[C1]]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED1]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]]] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source[%row_indices] into %dest lane(%arg6) : memref<1024x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2xindex>, memref<2x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Exactly 3 transfers (matching dest shape, not source shape)
    // CHECK: %[[ROW0:.+]] = memref.load %[[IDX]][%[[C0]]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
    // CHECK: %[[ROW1:.+]] = memref.load %[[IDX]][%[[C1]]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW1]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]]] : vector<4xf32>
    // CHECK: %[[ROW2:.+]] = memref.load %[[IDX]][%[[C2]]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW2]], %[[LANE_OFFSET]]], %[[DST]][%[[C2]], %[[C0]]] : vector<4xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer 1: elements [0, 128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LANE_OFFSET]]], %[[DST]][%[[C0]]] : vector<4xf32>
    // Transfer 2: elements [128, 256)
    // CHECK: %[[SRC_IDX1:.+]] = arith.addi %[[LANE_OFFSET]], %[[C128]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX1]]], %[[DST]][%[[C128]]] : vector<4xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Row 0, Transfer 1: [0, 0:128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
    // Row 0, Transfer 2: [0, 128:256)
    // CHECK: %[[COL1:.+]] = arith.addi %[[LANE_OFFSET]], %[[C128]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[COL1]]], %[[DST]][%[[C0]], %[[C128]]] : vector<4xf32>
    // Row 1, Transfer 1: [1, 0:128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]]] : vector<4xf32>
    // Row 1, Transfer 2: [1, 128:256)
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %{{.+}}], %[[DST]][%[[C1]], %[[C128]]] : vector<4xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // 128-bit DMA: lane_offset = lane_id * 4
    // CHECK: %[[LANE_OFFSET_4:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer 1: 128-bit DMA at offset 0
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LANE_OFFSET_4]]], %[[DST]][%[[C0]]] : vector<4xf32>
    // Transfer 2: 32-bit DMA at offset 128
    // For 32-bit DMA, src_idx = lane_id + 128
    // CHECK: %[[SRC_IDX1:.+]] = arith.addi %[[LANE_ID]], %[[C128]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX1]]], %[[DST]][%[[C128]]] : vector<1xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C96:.+]] = arith.constant 96 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer 1: [0, 0], 128-bit
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
    // Transfer 2: [0, 128], 128-bit
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %{{.+}}], %[[DST]][%[[C0]], %[[C128]]] : vector<4xf32>
    // Transfer 3: [1, 96], 32-bit
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %{{.+}}], %[[DST]][%[[C1]], %[[C96]]] : vector<1xf32>
    // Transfer 4: [1, 128], 32-bit
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %{{.+}}], %[[DST]][%[[C1]], %[[C128]]] : vector<1xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
// CHECK-LABEL: func.func @lower_coalesced_dma_linearized_2_rows_per_transfer
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<4x64xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<4x64xf32, #gpu.address_space<workgroup>>
func.func @lower_coalesced_dma_linearized_2_rows_per_transfer(
    %source: memref<4x64xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<4x64xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer 1: covers rows 0-1, starts at [0, 0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]]] : vector<4xf32>
    // Transfer 2: covers rows 2-3, starts at [2, 0]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C2]], %[[LANE_OFFSET]]], %[[DST]][%[[C2]], %[[C0]]] : vector<4xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
// CHECK-LABEL: func.func @lower_3d_partial_contiguous
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: memref<2x4x32xf32, strided<[256, 32, 1]>, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: memref<2x4x32xf32, strided<[256, 32, 1]>, #gpu.address_space<workgroup>>
func.func @lower_3d_partial_contiguous(
    %source: memref<2x4x32xf32, strided<[256, 32, 1]>, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x4x32xf32, strided<[256, 32, 1]>, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb_3d,
    translation_info = #translation_3d} {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Outer iteration 0: [0, 0, lane_offset]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0]], %[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C0]], %[[C0]], %[[C0]]] : vector<4xf32>
    // Outer iteration 1: [1, 0, lane_offset]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1]], %[[C0]], %[[LANE_OFFSET]]], %[[DST]][%[[C1]], %[[C0]], %[[C0]]] : vector<4xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
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
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: scf.forall
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %{{.+}}, %[[C4]]
    // Outer 0: 128-bit DMA at [0, 0, lane_offset]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[C0]], %[[C0]], %[[LANE_OFFSET]]], %{{.+}}[%[[C0]], %[[C0]], %[[C0]]] : vector<4xf32>
    // Outer 0: 32-bit DMA at [0, 4, lane_id]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[C0]], %[[C4]], %{{.+}}], %{{.+}}[%[[C0]], %[[C4]], %[[C0]]] : vector<1xf32>
    // Outer 1: 128-bit DMA at [1, 0, lane_offset]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[C1]], %[[C0]], %[[LANE_OFFSET]]], %{{.+}}[%[[C1]], %[[C0]], %[[C0]]] : vector<4xf32>
    // Outer 1: 32-bit DMA at [1, 4, lane_id]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[C1]], %[[C4]], %{{.+}}], %{{.+}}[%[[C1]], %[[C4]], %[[C0]]] : vector<1xf32>
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
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test gather with indices on the innermost (only) dimension.
// This tests the code path where dim < numIndexDims AND isInnermost is true.
// The simplified implementation computes:
//   srcIdx = load(indices[dimOffset]) + laneOffset
//
// For a 1D gather from source[1024] to dest[256] with 32 lanes:
//   - 32 lanes * 4 elements/lane = 128 elements per transfer
//   - 256 / 128 = 2 transfers needed
//   - Transfer 0: dimOffset = 0, srcIdx = indices[0] + lane_offset
//   - Transfer 1: dimOffset = 128, srcIdx = indices[128] + lane_offset
//
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_innermost_1d
func.func @lower_coalesced_gather_dma_innermost_1d(
    %source: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %col_indices: memref<256xindex>,
    %dest: memref<256xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_32} {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // Transfer 1: load indices[0], srcIdx = loaded + lane_offset
    // CHECK: %[[LOADED0:.+]] = memref.load %{{.+}}[%[[C0]]]
    // CHECK: %[[SRC0:.+]] = arith.addi %[[LOADED0]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[SRC0]]], %{{.+}}[%[[C0]]]
    // Transfer 2: load indices[128], srcIdx = loaded + lane_offset
    // CHECK: %[[LOADED128:.+]] = memref.load %{{.+}}[%[[C128]]]
    // CHECK: %[[SRC128:.+]] = arith.addi %[[LOADED128]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %{{.+}}[%[[SRC128]]], %{{.+}}[%[[C128]]]
    // CHECK-NOT: amdgpu.gather_to_lds
    iree_gpu.coalesced_gather_dma %source[%col_indices] into %dest lane(%arg6) :
      memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<256xindex>,
      memref<256xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}
