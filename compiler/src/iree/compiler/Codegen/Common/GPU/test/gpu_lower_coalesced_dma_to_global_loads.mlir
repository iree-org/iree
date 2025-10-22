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

// CHECK-LABEL: func.func @lower_coalesced_gather_dma_multiple
func.func @lower_coalesced_gather_dma_multiple(
    %source: memref<512xf32, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<512xf32, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb} {
  // CHECK: scf.forall
  scf.forall (%arg5, %arg6) in (32, 1) {
    // CHECK: iree_gpu.coalesced_gather_dma
    iree_gpu.coalesced_gather_dma %source into %dest dest_size = [2, 8] lane(%arg6) :
      memref<512xf32, #amdgpu.address_space<fat_raw_buffer>>,
      memref<512xf32, #gpu.address_space<workgroup>>, index
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

// CHECK-LABEL: func.func @lower_loop_nest
func.func @lower_loop_nest(%arg0: memref<1x4xf32, #amdgpu.address_space<fat_raw_buffer>>,
                           %arg2: memref<32x32xf32, #gpu.address_space<workgroup>>) -> memref<32x32xf32, #gpu.address_space<workgroup>>
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb} {
  // CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (32, 32) step (1, 32)
  scf.forall (%arg3, %arg4) = (0, 0) to (32, 32) step (1, 32) {
    // CHECK: %[[SUBVIEW:.+]] = memref.subview
    %subview_0 = memref.subview %arg2[%arg3, %arg4] [1, 4] [1, 1] : memref<32x32xf32, #gpu.address_space<workgroup>> to memref<1x4xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
    // CHECK: scf.forall
    scf.forall (%arg5, %arg6) in (1, 32) {
      // CHECK-NOT: iree_gpu.coalesced_gather_dma
      // CHECK: %[[C0:.*]] = arith.constant 0 : index
      // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
      // CHECK: amdgpu.gather_to_lds %{{.*}}[%[[C0]], %[[C0_0]]], %[[SUBVIEW]][%[[C0]], %[[C0_0]]] : vector<4xf32>
      iree_gpu.coalesced_gather_dma %arg0 into %subview_0 dest_size = [2, 8] lane(%arg6) :
        memref<1x4xf32, #amdgpu.address_space<fat_raw_buffer>>,
        memref<1x4xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>, index
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return %arg2 : memref<32x32xf32, #gpu.address_space<workgroup>>
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

// Test case for coalesced DMA without explicit indices (copy operation)
// CHECK-LABEL: func.func @lower_coalesced_copy_dma_basic
func.func @lower_coalesced_copy_dma_basic(
    %source: memref<2x8xf16>,
    %dest: memref<2x8xf16, #gpu.address_space<workgroup>>)
  attributes {
    hal.executable.target = #executable_target_rocm_hsaco_fb} {
  // CHECK: scf.forall
  scf.forall (%arg5, %arg6) in (1, 64) {
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
    // CHECK: amdgpu.gather_to_lds %{{.*}}[%[[C0]], %[[C0_0]]], %{{.*}}[%[[C0]], %[[C0_0]]] : vector<8xf16>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
    // CHECK: amdgpu.gather_to_lds %{{.*}}[%[[C1]], %[[C0_1]]], %{{.*}}[%[[C1]], %[[C0_1]]] : vector<8xf16>
    iree_gpu.coalesced_gather_dma %source into %dest dest_size = [2, 8] lane(%arg6) : memref<2x8xf16>, memref<2x8xf16, #gpu.address_space<workgroup>>, index
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return
}
