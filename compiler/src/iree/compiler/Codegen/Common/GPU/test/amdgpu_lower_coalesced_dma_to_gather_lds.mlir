// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds))" \
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
    // CHECK: %[[C4:.+]] = arith.constant 4 : index
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0: source[0, offset + lane_offset], dest[0, 0]
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 1: source[1, offset + lane_offset], dest[1, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 2: source[2, offset + lane_offset], dest[2, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 3: source[3, offset + lane_offset], dest[3, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
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
    // CHECK: %[[C2:.+]] = arith.constant 2 : index
    // CHECK: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C2]]
    //
    // Row 0: source[0, offset + lane_offset], dest[0, 0]
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<2xf16>
    //
    // Row 1: source[1, offset + lane_offset], dest[1, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<2xf16>
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
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // CHECK: %[[C0:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_IDX:[a-zA-Z0-9_]+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX]]], %[[DST]][%{{.+}}] : vector<4xf32>
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
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Tile [0, 0, 0]
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Tile [0, 1, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Tile [1, 0, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Tile [1, 1, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
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
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (256)
  scf.forall (%arg6) in (256) {
    // elementsPerTransfer = 1024 / 256 = 4
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0: source[0, offset + lane_offset], dest[0, 0]
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 1: source[1, offset + lane_offset], dest[1, 0]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
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
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0: load row_indices[0], then gather source[loaded_idx, col+lane_offset]
    // CHECK: memref.load %[[IDX]]
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 1: load row_indices[1], then gather source[loaded_idx, col+lane_offset]
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: memref.load %[[IDX]]
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
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
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Row 0
    // CHECK: memref.load %[[IDX]]
    // CHECK: arith.addi %{{.+}}, %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 1
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: memref.load %[[IDX]]
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 2
    // CHECK: arith.muli %[[LANE_ID]], %{{.+}}
    // CHECK: memref.load %[[IDX]]
    // CHECK: arith.addi %{{.+}}, %{{.+}}
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
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
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Each lane reads 4 elements per transfer.
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    //
    // Transfer 1: elements [0, 128), tile offset = 0
    // CHECK: %[[C0:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_IDX0:[a-zA-Z0-9_]+]] = arith.addi %[[C0]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX0]]], %[[DST]][%{{.+}}] : vector<4xf32>
    //
    // Transfer 2: elements [128, 256), tile offset = 128
    // CHECK: %[[C128:[a-zA-Z0-9_]+]] = arith.constant 128
    // CHECK: %[[SRC_IDX1:[a-zA-Z0-9_]+]] = arith.addi %[[C128]], %[[LANE_OFFSET]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX1]]], %[[DST]][%{{.+}}] : vector<4xf32>
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
  // CHECK: scf.forall (%[[LANE_ID:[a-zA-Z0-9]+]]) in (32)
  scf.forall (%arg6) in (32) {
    // Transfer 1: 128-bit DMA, elements [0, 128)
    // CHECK: %[[C4:[a-zA-Z0-9_]+]] = arith.constant 4
    // CHECK: %[[LANE_OFFSET_4:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C4]]
    // CHECK: %[[C0:[a-zA-Z0-9_]+]] = arith.constant 0
    // CHECK: %[[SRC_IDX0:[a-zA-Z0-9_]+]] = arith.addi %[[C0]], %[[LANE_OFFSET_4]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX0]]], %[[DST]][%{{.+}}] : vector<4xf32>
    //
    // Transfer 2: 32-bit DMA, elements [128, 160)
    // CHECK: %[[C1:[a-zA-Z0-9_]+]] = arith.constant 1
    // CHECK: %[[LANE_OFFSET_1:[a-zA-Z0-9_]+]] = arith.muli %[[LANE_ID]], %[[C1]]
    // CHECK: %[[C128:[a-zA-Z0-9_]+]] = arith.constant 128
    // CHECK: %[[SRC_IDX1:[a-zA-Z0-9_]+]] = arith.addi %[[C128]], %[[LANE_OFFSET_1]]
    // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SRC_IDX1]]], %[[DST]][%{{.+}}] : vector<1xf32>
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

// Test: Mixed DMA sizes with 2D memref.
//   - Shape: 2x160 f32s
//   - subgroupSize = 32, dma_sizes = [32, 128]
//   - Per row: 1×128-bit transfer + 1×32-bit transfer
//   - Total: 4 transfers (2 rows × 2 transfers/row)
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
    // Row 0, Transfer 1: 128-bit DMA, [0, 0:128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 0, Transfer 2: 32-bit DMA, [0, 128:160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<1xf32>
    //
    // Row 1, Transfer 1: 128-bit DMA, [1, 0:128)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<4xf32>
    //
    // Row 1, Transfer 2: 32-bit DMA, [1, 128:160)
    // CHECK: amdgpu.gather_to_lds %[[SRC]]{{.+}} : vector<1xf32>
    // CHECK-NOT: amdgpu.gather_to_lds
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) : memref<2x160xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<2x160xf32, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}
