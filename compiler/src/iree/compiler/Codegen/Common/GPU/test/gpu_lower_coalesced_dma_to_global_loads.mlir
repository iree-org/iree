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
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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

// CHECK-LABEL: func.func @lower_loop_nest
func.func @lower_loop_nest(%arg0: memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>,
                           %arg1: memref<32x32xindex>,
                           %arg2: memref<32x32xf32, #gpu.address_space<workgroup>>) -> memref<32x32xf32, #gpu.address_space<workgroup>>
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb,
    translation_info = #translation_info} {
  // CHECK: scf.forall (%[[WG_I:.+]], %[[WG_J:.+]]) = (0, 0) to (32, 32) step (1, 32)
  // CHECK:   %[[INDICES_SLICE:.+]] = memref.subview %arg1[%[[WG_I]], %[[WG_J]]] [1, 32] [1, 1]
  // CHECK-SAME:   memref<32x32xindex> to memref<1x32xindex, strided<[32, 1], offset: ?>>
  // CHECK:   %[[DEST_SLICE:.+]] = memref.subview %arg2[%[[WG_I]], %[[WG_J]]] [1, 32] [1, 1]
  // CHECK-SAME:   memref<32x32xf32, #gpu.address_space<workgroup>> to memref<1x32xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
  // CHECK:   %[[C0:.+]] = arith.constant 0 : index
  // CHECK:   %[[C1:.+]] = arith.constant 1 : index
  // CHECK:   %[[C1_STEP:.+]] = arith.constant 1 : index
  // CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[C1]] step %[[C1_STEP]]
  // CHECK:     %[[C32:.+]] = arith.constant 32 : index
  // CHECK:     %[[IDX_OFFSET:.+]] = arith.muli %[[IV]], %[[C32]]
  // CHECK:     %[[IDX_POS:.+]] = arith.addi %[[LANE_ID]], %[[IDX_OFFSET]]
  // CHECK:     %[[DELINEAR_IDX:.+]]:2 = affine.delinearize_index %[[IDX_POS]] into (1, 32)
  // CHECK:     %[[LOADED_IDX:.+]] = memref.load %[[INDICES_SLICE]][%[[DELINEAR_IDX]]#0, %[[DELINEAR_IDX]]#1]
  // CHECK:     %[[C32_1:.+]] = arith.constant 32 : index
  // CHECK:     %[[DEST_OFFSET_MUL:.+]] = arith.muli %[[IV]], %[[C32_1]]
  // CHECK:     %[[DELINEAR_DEST:.+]]:2 = affine.delinearize_index %[[DEST_OFFSET_MUL]] into (1, 32)
  // CHECK:     amdgpu.gather_to_lds %arg0[%[[LOADED_IDX]]], %[[DEST_SLICE]][%[[DELINEAR_DEST]]#0, %[[DELINEAR_DEST]]#1]
  // CHECK:     }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  scf.forall (%arg3, %arg4) = (0, 0) to (32, 32) step (1, 32) {
    %subview = memref.subview %arg1[%arg3, %arg4] [1, 32] [1, 1] : memref<32x32xindex> to memref<1x32xindex, strided<[32, 1], offset: ?>>
    %subview_0 = memref.subview %arg2[%arg3, %arg4] [1, 32] [1, 1] : memref<32x32xf32, #gpu.address_space<workgroup>> to memref<1x32xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
    scf.forall (%arg5, %arg6) in (1, 32) {
      %0 = iree_gpu.coalesced_gather_dma %subview, %arg0 into %subview_0 :
        memref<1x32xindex, strided<[32, 1], offset: ?>>,
        memref<1024xf32, #amdgpu.address_space<fat_raw_buffer>>,
        memref<1x32xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>> ->
        memref<1x32xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return %arg2 : memref<32x32xf32, #gpu.address_space<workgroup>>
}
