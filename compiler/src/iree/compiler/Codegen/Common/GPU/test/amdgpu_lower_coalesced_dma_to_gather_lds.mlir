// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds))" \
// RUN:   %s | FileCheck %s

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
