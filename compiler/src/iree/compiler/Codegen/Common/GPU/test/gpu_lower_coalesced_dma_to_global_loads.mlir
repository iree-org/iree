// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-coalesced-dma-to-global-loads))" %s | FileCheck %s

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32,
                                                  {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true,
                                                                                                     no_reduce_shared_memory_bank_conflicts = false,
                                                                                                     use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @lower_coalesced_gather_dma_basic
func.func @lower_coalesced_gather_dma_basic(%indices: memref<32xindex>, %source: memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, %dest: memref<128xf32, #gpu.address_space<workgroup>>) attributes {translation_info = #translation_info} {
  // CHECK-NOT: scf.forall
  // CHECK: scf.for
  // CHECK: amdgpu.gather_to_lds
  // CHECK: gpu.barrier
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<128xf32, #gpu.address_space<workgroup>> -> memref<128xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return
}

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32,
                                                  {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true,
                                                                                                     no_reduce_shared_memory_bank_conflicts = false,
                                                                                                     use_igemm_convolution = false>}>

// Test case with nested forall loops - only inner loop should be transformed
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_nested
func.func @lower_coalesced_gather_dma_nested(%indices: memref<32xindex>, %source: memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, %dest: memref<128xf32, #gpu.address_space<workgroup>>) attributes {translation_info = #translation_info} {
  // CHECK: scf.forall (%{{.+}}, %{{.+}}) in (4, 1)
  scf.forall (%wg_i, %wg_j) in (4, 1) {
    // CHECK-NOT: scf.forall (%{{.+}}, %{{.+}}) in (32, 1)
    // CHECK: scf.for
    // CHECK: amdgpu.gather_to_lds
    // CHECK: gpu.barrier
    // CHECK-NOT: iree_gpu.coalesced_gather_dma
    scf.forall (%sg_i, %sg_j) in (32, 1) {
      %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<128xf32, #gpu.address_space<workgroup>> -> memref<128xf32, #gpu.address_space<workgroup>>
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  // CHECK: mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]
  return
}

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32,
                                                  {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true,
                                                                                                     no_reduce_shared_memory_bank_conflicts = false,
                                                                                                     use_igemm_convolution = false>}>

// Test case with larger destination that requires multiple loads per thread
// 256 f32 elements = 1024 bytes, 1024/32 threads = 32 bytes per thread, 32/16 bytes per load = 2 loads per thread
// CHECK-LABEL: func.func @lower_coalesced_gather_dma_multiple
func.func @lower_coalesced_gather_dma_multiple(%indices: memref<32xindex>, %source: memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, %dest: memref<256xf32, #gpu.address_space<workgroup>>) attributes {translation_info = #translation_info} {
  // CHECK-NOT: scf.forall
  // CHECK: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: scf.for %{{.+}} = %{{.+}} to %[[C2]]
  // CHECK: amdgpu.gather_to_lds
  // CHECK: gpu.barrier
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<256xf32, #gpu.address_space<workgroup>> -> memref<256xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return
}
