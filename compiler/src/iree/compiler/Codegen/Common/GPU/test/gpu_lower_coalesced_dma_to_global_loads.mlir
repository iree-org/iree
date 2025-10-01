// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-coalesced-dma-to-global-loads))" \
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

#translation_info = #iree_codegen.translation_info<
  pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1]
  subgroup_size = 32, {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_shared_memory = true,
      no_reduce_shared_memory_bank_conflicts = false,
      use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @lower_coalesced_gather_dma_basic
func.func @lower_coalesced_gather_dma_basic(
    %indices: memref<32xindex>,
    %source: memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<128xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_info} {
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK: %[[LANE_ID:.+]] = gpu.lane_id
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[C1_STEP:.+]] = arith.constant 1 : index
  // CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[C1]] step %[[C1_STEP]]
  // CHECK:   %[[C32:.+]] = arith.constant 32 : index
  // CHECK:   %[[IDX_OFFSET:.+]] = arith.muli %[[IV]], %[[C32]]
  // CHECK:   %[[IDX_POS:.+]] = arith.addi %[[LANE_ID]], %[[IDX_OFFSET]]
  // CHECK:   %[[DELINEAR_IDX:.+]] =
  // CHECK-SAME: affine.delinearize_index %[[IDX_POS]] into (32)
  // CHECK:   %[[LOADED_IDX:.+]] = memref.load %arg0[%[[DELINEAR_IDX]]]
  // CHECK:   %[[C32_1:.+]] = arith.constant 32 : index
  // CHECK:   %[[DEST_OFFSET_MUL:.+]] = arith.muli %[[IV]], %[[C32_1]]
  // CHECK:   %[[DELINEAR_DEST:.+]] =
  // CHECK-SAME: affine.delinearize_index %[[DEST_OFFSET_MUL]] into (128)
  // CHECK:   amdgpu.gather_to_lds %arg1[%[[LOADED_IDX]]],
  // CHECK-SAME: %arg2[%[[DELINEAR_DEST]]] : vector<4xf32>
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest :
      memref<32xindex>,
      memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<128xf32, #gpu.address_space<workgroup>> ->
      memref<128xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
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

#translation_info = #iree_codegen.translation_info<
  pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1]
  subgroup_size = 32, {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_shared_memory = true,
      no_reduce_shared_memory_bank_conflicts = false,
      use_igemm_convolution = false>}>

// Test case where each index loads 4xf32 per iteration across 4 iterations
// 512 f32 elements = 2048 bytes, 128 indices,
// bytesPerIndexCopy = 2048/128 = 16 bytes = 4xf32
// 128 indices / 32 threads = 4 iterations per thread
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_multiple
func.func @lower_coalesced_gather_dma_multiple(
    %indices: memref<128xindex>,
    %source: memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<512xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_info} {
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK: %[[LANE_ID:.+]] = gpu.lane_id
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
  // CHECK:   %[[C32:.+]] = arith.constant 32 : index
  // CHECK:   %[[IDX_OFFSET:.+]] = arith.muli %[[IV]], %[[C32]]
  // CHECK:   %[[IDX_POS:.+]] = arith.addi %[[LANE_ID]], %[[IDX_OFFSET]]
  // CHECK:   %[[DELINEAR_IDX:.+]] =
  // CHECK-SAME: affine.delinearize_index %[[IDX_POS]] into (128)
  // CHECK:   %[[LOADED_IDX:.+]] = memref.load %arg0[%[[DELINEAR_IDX]]]
  // CHECK:   %[[C32_0:.+]] = arith.constant 32 : index
  // CHECK:   %[[DEST_OFFSET_MUL:.+]] = arith.muli %[[IV]], %[[C32_0]]
  // CHECK:   %[[DELINEAR_DEST:.+]] =
  // CHECK-SAME: affine.delinearize_index %[[DEST_OFFSET_MUL]] into (512)
  // CHECK:   amdgpu.gather_to_lds %arg1[%[[LOADED_IDX]]],
  // CHECK-SAME: %arg2[%[[DELINEAR_DEST]]] : vector<4xf32>
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest :
      memref<128xindex>,
      memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<512xf32, #gpu.address_space<workgroup>> ->
      memref<512xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
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

#translation_info = #iree_codegen.translation_info<
  pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1]
  subgroup_size = 32, {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_shared_memory = true,
      no_reduce_shared_memory_bank_conflicts = false,
      use_igemm_convolution = false>}>

// 32x32 = 1024 indices, 128x32 = 4096 f32 elements = 16384 bytes
// bytesPerIndexCopy = 16384/1024 = 16 bytes = 4xf32
// 1024 indices / 32 threads = 32 iterations per thread
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_2d
func.func @lower_coalesced_gather_dma_2d(
    %indices: memref<32x32xindex>,
    %source: memref<128x32xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<128x32xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_info} {
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK: %[[LANE_ID:.+]] = gpu.lane_id
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
  // CHECK:   %[[C32_0:.+]] = arith.constant 32 : index
  // CHECK:   %[[IDX_OFFSET:.+]] = arith.muli %[[IV]], %[[C32_0]]
  // CHECK:   %[[IDX_POS:.+]] = arith.addi %[[LANE_ID]], %[[IDX_OFFSET]]
  // CHECK:   %[[DELINEAR_IDX:.+]]:2 = affine.delinearize_index %[[IDX_POS]]
  // CHECK-SAME: into (32, 32) : index, index
  // CHECK:   %[[LOADED_IDX:.+]] = memref.load %arg0[%[[DELINEAR_IDX]]#0,
  // CHECK-SAME: %[[DELINEAR_IDX]]#1]
  // CHECK:   %[[C32_1:.+]] = arith.constant 32 : index
  // CHECK:   %[[DEST_OFFSET_MUL:.+]] = arith.muli %[[IV]], %[[C32_1]]
  // CHECK:   %[[DELINEAR_DEST:.+]]:2 = affine.delinearize_index
  // CHECK-SAME: %[[DEST_OFFSET_MUL]] into (128, 32) : index, index
  // CHECK:   amdgpu.gather_to_lds %arg1[%[[LOADED_IDX]]],
  // CHECK-SAME: %arg2[%[[DELINEAR_DEST]]#0, %[[DELINEAR_DEST]]#1] :
  // CHECK-SAME: vector<4xf32>
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest :
      memref<32x32xindex>,
      memref<128x32xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<128x32xf32, #gpu.address_space<workgroup>> ->
      memref<128x32xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return
}
