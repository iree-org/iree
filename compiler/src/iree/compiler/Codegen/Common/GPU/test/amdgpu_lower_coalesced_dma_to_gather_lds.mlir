// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds))" \
// RUN:   --verify-diagnostics %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx1250", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192>>}>

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
    // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
    // CHECK-DAG: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0: source[0, offset + lane_offset], dest[0, 0]
    // CHECK-DAG: %[[ROW0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_COL0:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW0]], %[[SRC_COL0]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
    //
    // Row 1: source[1, offset + lane_offset], dest[1, 0]
    // CHECK: %[[ROW1:.+]] = arith.constant 1 : index
    // CHECK: %[[SRC_COL1:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW1]], %[[SRC_COL1]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
    //
    // Row 2: source[2, offset + lane_offset], dest[2, 0]
    // CHECK: %[[ROW2:.+]] = arith.constant 2 : index
    // CHECK: %[[SRC_COL2:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW2]], %[[SRC_COL2]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
    //
    // Row 3: source[3, offset + lane_offset], dest[3, 0]
    // CHECK: %[[ROW3:.+]] = arith.constant 3 : index
    // CHECK: %[[SRC_COL3:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW3]], %[[SRC_COL3]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
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
  arch = "gfx1250", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192>>}>

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
    // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
    // CHECK-DAG: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C2]]
    //
    // Row 0: source[0, offset + lane_offset], dest[0, 0]
    // CHECK-DAG: %[[ROW0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_COL0:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW0]], %[[SRC_COL0]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<2xf16>
    //
    // Row 1: source[1, offset + lane_offset], dest[1, 0]
    // CHECK: %[[ROW1:.+]] = arith.constant 1 : index
    // CHECK: %[[SRC_COL1:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW1]], %[[SRC_COL1]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<2xf16>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<2x64xf16, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx1250", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192>>}>

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

    // CHECK: %[[SRC_IDX:[a-zA-Z0-9_]+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX]]], %[[DST]][%[[C0]]] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx1250", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test: 3D memref with shape 2x2x128 and 32 lanes.
//   * Elements per lane = 128 / 33 = 4
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
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Tile [0, 0, 0]
    // CHECK: %[[C0_D0:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[C0_D1:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[C0_D2:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_IDX_0:[a-zA-Z0-9_]+]] = arith.addi %[[C0_D2]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C0_D0]], %[[C0_D1]], %[[SRC_IDX_0]]], %[[DST]][%[[C0_D0]], %[[C0_D1]], %[[C0_D2]]] : vector<4xf32>
    //
    // Tile [0, 1, 0]
    // CHECK: %[[C1_D1:[a-zA-Z0-9_]+]] = arith.constant 1
    // CHECK: %[[C0_D2_1:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_IDX_1:[a-zA-Z0-9_]+]] = arith.addi %[[C0_D2_1]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{[a-zA-Z0-9_]+}}, %[[C1_D1]], %[[SRC_IDX_1]]], %[[DST]][%{{[a-zA-Z0-9_]+}}, %[[C1_D1]], %[[C0_D2_1]]] : vector<4xf32>
    //
    // Tile [1, 0, 0]
    // CHECK: %[[C1_D0:[a-zA-Z0-9_]+]] = arith.constant 1
    // CHECK: %[[C0_D1_2:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[C0_D2_2:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_IDX_2:[a-zA-Z0-9_]+]] = arith.addi %[[C0_D2_2]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1_D0]], %[[C0_D1_2]], %[[SRC_IDX_2]]], %[[DST]][%[[C1_D0]], %[[C0_D1_2]], %[[C0_D2_2]]] : vector<4xf32>
    //
    // Tile [1, 1, 0]
    // CHECK: %[[C1_D0_3:[a-zA-Z0-9_]+]] = arith.constant 1
    // CHECK: %[[C1_D1_3:[a-zA-Z0-9_]+]] = arith.constant 1
    // CHECK: %[[C0_D2_3:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_IDX_3:[a-zA-Z0-9_]+]] = arith.addi %[[C0_D2_3]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[C1_D0_3]], %[[C1_D1_3]], %[[SRC_IDX_3]]], %[[DST]][%[[C1_D0_3]], %[[C1_D1_3]], %[[C0_D2_3]]] : vector<4xf32>
    //
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x2x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x2x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb_wide = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx1250", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [256, 256],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192>>}>

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
    // elementsPerTransfer = 1024 / 256 = 4
    // CHECK-DAG: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK-DAG: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0: source[0, offset + lane_offset], dest[0, 0]
    // CHECK-DAG: %[[ROW0:.+]] = arith.constant 0 : index
    // CHECK: %[[SRC_COL0:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW0]], %[[SRC_COL0]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
    //
    // Row 1: source[1, offset + lane_offset], dest[1, 0]
    // CHECK: %[[ROW1:.+]] = arith.constant 1 : index
    // CHECK: %[[SRC_COL1:.+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW1]], %[[SRC_COL1]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x1024xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x1024xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx1250", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192>>}>

#translation_32 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// Test gather 2D:
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
    // CHECK-DAG: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK-DAG: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0: load row_indices[0], then gather source[loaded_idx, col+lane_offset]
    // CHECK-DAG: %[[ROW0:[a-zA-Z0-9_]+]] = arith.constant 0 : index
    // CHECK: %[[LOADED_ROW0:[a-zA-Z0-9_]+]] = memref.load %[[IDX]][%[[ROW0]]]
    // CHECK: %[[SRC_COL0:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW0]], %[[SRC_COL0]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
    //
    // Row 1: load row_indices[1], then gather source[loaded_idx, col+lane_offset]
    // CHECK: %[[ROW1:[a-zA-Z0-9_]+]] = arith.constant 1 : index
    // CHECK: %[[LOADED_ROW1:[a-zA-Z0-9_]+]] = memref.load %[[IDX]][%[[ROW1]]]
    // CHECK: %[[SRC_COL1:[a-zA-Z0-9_]+]] = arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW1]], %[[SRC_COL1]]], %[[DST]][%{{.+}}, %{{.+}}] : vector<4xf32>
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source[%row_indices] into %dest lane(%arg6) : memref<1024x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2xindex>, memref<2x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx1250", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192>>}>

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
    // CHECK-DAG: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK-DAG: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0: load row_indices[0], then gather source[loaded_idx, col+lane_offset]
    // CHECK-DAG: %[[ROW0:[a-zA-Z0-9_]+]] = arith.constant 0 : index
    // CHECK: %[[LOADED_ROW0:[a-zA-Z0-9_]+]] = memref.load %[[IDX]][%[[ROW0]]]
    // CHECK-NEXT: %[[COL0_BASE:[a-zA-Z0-9_]+]] = arith.constant 0 : index
    // CHECK-NEXT: %[[SRC_COL0:[a-zA-Z0-9_]+]] = arith.addi %[[COL0_BASE]], %[[LANE_OFFSET]]
    // CHECK-NEXT: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW0]], %[[SRC_COL0]]], %[[DST]][%[[ROW0]], %[[COL0_BASE]]] : vector<4xf32>
    //
    // Row 1: load row_indices[1], then gather source[loaded_idx, col+lane_offset]
    // CHECK-NEXT: %[[ROW1:[a-zA-Z0-9_]+]] = arith.constant 1 : index
    // CHECK-NEXT: %[[LOADED_ROW1:[a-zA-Z0-9_]+]] = memref.load %[[IDX]][%[[ROW1]]]
    // CHECK-NEXT: %[[COL1_BASE:[a-zA-Z0-9_]+]] = arith.constant 0 : index
    // CHECK-NEXT: %[[SRC_COL1:[a-zA-Z0-9_]+]] = arith.addi %[[COL1_BASE]], %[[LANE_OFFSET]]
    // CHECK-NEXT: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW1]], %[[SRC_COL1]]], %[[DST]][%[[ROW1]], %[[COL1_BASE]]] : vector<4xf32>
    //
    // Row 2: load row_indices[2], then gather source[loaded_idx, col+lane_offset]
    // CHECK-NEXT: %[[ROW2:[a-zA-Z0-9_]+]] = arith.constant 2 : index
    // CHECK-NEXT: %[[LOADED_ROW2:[a-zA-Z0-9_]+]] = memref.load %[[IDX]][%[[ROW2]]]
    // CHECK-NEXT: %[[COL2_BASE:[a-zA-Z0-9_]+]] = arith.constant 0 : index
    // CHECK-NEXT: %[[SRC_COL2:[a-zA-Z0-9_]+]] = arith.addi %[[COL2_BASE]], %[[LANE_OFFSET]]
    // CHECK-NEXT: amdgpu.gather_to_lds %[[SRC]][%[[LOADED_ROW2]], %[[SRC_COL2]]], %[[DST]][%[[ROW2]], %[[COL2_BASE]]] : vector<4xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source[%row_indices] into %dest lane(%arg6) : memref<256x128xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<3xindex>, memref<3x128xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}
