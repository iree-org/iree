// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-coalesced-dma-to-global-loads))" %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1250", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 8192>>}>

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32,
                                                  {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true,
                                                                                                     no_reduce_shared_memory_bank_conflicts = false,
                                                                                                     use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @lower_coalesced_gather_dma_basic
func.func @lower_coalesced_gather_dma_basic(%indices: memref<32xindex>, %source: memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, %dest: memref<128xf32, #gpu.address_space<workgroup>>) attributes {hal.executable.target = #executable_target_rocm_hsaco_fb, translation_info = #translation_info} {
  // CHECK-NOT: scf.forall
  // CHECK: scf.for
  // CHECK: amdgpu.gather_to_lds
  // CHECK: gpu.barrier
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<128xf32, #gpu.address_space<workgroup>> -> memref<128xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return
}

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1250", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32, 32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 8192>>}>

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32,
                                                  {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true,
                                                                                                     no_reduce_shared_memory_bank_conflicts = false,
                                                                                                     use_igemm_convolution = false>}>

// Test case with larger destination that requires multiple loads per thread
// 256 f32 elements = 1024 bytes, 1024/32 threads = 32 bytes per thread, 32/16 bytes per load = 2 loads per thread
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_multiple
func.func @lower_coalesced_gather_dma_multiple(%indices: memref<32xindex>, %source: memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, %dest: memref<256xf32, #gpu.address_space<workgroup>>) attributes {hal.executable.target = #executable_target_rocm_hsaco_fb, translation_info = #translation_info} {
  // CHECK-NOT: scf.forall
  // CHECK: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: scf.for %{{.+}} = %{{.+}} to %[[C2]]
  // CHECK: amdgpu.gather_to_lds
  // CHECK: gpu.barrier
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<256xf32, #gpu.address_space<workgroup>> -> memref<256xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return
}
