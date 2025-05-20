// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-to-global-loads))" %s | FileCheck %s

func.func @matmul_config_1() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x64xi8, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<16x64xi8, #hal.descriptor_type<storage_buffer>> to memref<16x64xi8, #amdgpu.address_space<fat_raw_buffer>>
  %alloc = memref.alloc() : memref<16x64xi8, #gpu.address_space<workgroup>>
  linalg.copy {lowering_config = #iree_gpu.use_global_load_dma} ins(%1 : memref<16x64xi8, #amdgpu.address_space<fat_raw_buffer>>) outs(%alloc : memref<16x64xi8, #gpu.address_space<workgroup>>)
  return
}

// To gather a memref<16x64xi8> buffer:
// number of subgroups: 64 / 64 = 1
// each subgroup loads: 16 * 64 / 1 = 1024 elements
// each subgroup load 64 * (32 / bitwidth(i8)) = 256 elements
// number of loads per subgroup: 1024 / 256 = 4

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 1024 + s2 * 256)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 * 1024 + s1 * 256)>
// CHECK: func.func @matmul_config_1

// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[BUFFER:.*]] = amdgpu.fat_raw_buffer_cast
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: %[[SGID:.*]] = gpu.subgroup_id : index
// CHECK: %[[LID:.*]] = gpu.lane_id
// CHECK: scf.for %[[ARG0:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:   %[[GATHER_ADDR:.*]] = affine.apply #[[MAP]]()[%[[LID]], %[[SGID]], %[[ARG0]]]
// CHECK:   %[[DELI_GATHER:.*]]:2 = affine.delinearize_index %[[GATHER_ADDR]] into (16, 64) : index, index
// CHECK:   %[[STORE_ADDR:.*]] = affine.apply #[[MAP1]]()[%[[SGID]], %[[ARG0]]]
// CHECK:   %[[DELI_STORE:.*]]:2 = affine.delinearize_index %[[STORE_ADDR]] into (16, 64) : index, index
// CHECK:   iree_gpu.global_load_dma %[[BUFFER]][%[[DELI_GATHER]]#0, %[[DELI_GATHER]]#1] -> %[[ALLOC]][%[[DELI_STORE]]#0, %[[DELI_STORE]]#1]

// -----

func.func @matmul_config_2() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x16xi8, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<64x16xi8, #hal.descriptor_type<storage_buffer>> to memref<64x16xi8, #amdgpu.address_space<fat_raw_buffer>>
  %alloc = memref.alloc() : memref<64x16xi8, #gpu.address_space<workgroup>>
  linalg.copy {lowering_config = #iree_gpu.use_global_load_dma} ins(%1 : memref<64x16xi8, #amdgpu.address_space<fat_raw_buffer>>) outs(%alloc : memref<64x16xi8, #gpu.address_space<workgroup>>)
  return
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 1024 + s2 * 256)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 * 1024 + s1 * 256)>
// CHECK: func.func @matmul_config_2
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[BUFFER:.*]] = amdgpu.fat_raw_buffer_cast
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: %[[SGID:.*]] = gpu.subgroup_id : index
// CHECK: %[[LID:.*]] = gpu.lane_id
// CHECK: scf.for %[[ARG0:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:   %[[GATHER_ADDR:.*]] = affine.apply #[[MAP]]()[%[[LID]], %[[SGID]], %[[ARG0]]]
// CHECK:   %[[DELI_GATHER:.*]]:2 = affine.delinearize_index %[[GATHER_ADDR]] into (64, 16) : index, index
// CHECK:   %[[STORE_ADDR:.*]] = affine.apply #[[MAP1]]()[%[[SGID]], %[[ARG0]]]
// CHECK:   %[[DELI_STORE:.*]]:2 = affine.delinearize_index %[[STORE_ADDR]] into (64, 16) : index, index
// CHECK:   iree_gpu.global_load_dma %[[BUFFER]][%[[DELI_GATHER]]#0, %[[DELI_GATHER]]#1] -> %[[ALLOC]][%[[DELI_STORE]]#0, %[[DELI_STORE]]#1]

// -----

func.func @matmul_config_3() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x64xi16, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<32x64xi16, #hal.descriptor_type<storage_buffer>> to memref<32x64xi16, #amdgpu.address_space<fat_raw_buffer>>
  %alloc = memref.alloc() : memref<32x64xi16, #gpu.address_space<workgroup>>
  linalg.copy {lowering_config = #iree_gpu.use_global_load_dma} ins(%1 : memref<32x64xi16, #amdgpu.address_space<fat_raw_buffer>>) outs(%alloc : memref<32x64xi16, #gpu.address_space<workgroup>>)
  return
}

// To gather a memref<32x64xi16> buffer:
// number of subgroups: 64 / 32 = 2
// each subgroup loads: 32 * 64 / 2 = 1024 elements
// each subgroup load 32 * (32 / bitwidth(i16)) = 64 elements
// number of loads per subgroup: 1024 / 64 = 16

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 2 + s1 * 1024 + s2 * 64)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 * 1024 + s1 * 64)>
// CHECK: func.func @matmul_config_3
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[BUFFER:.*]] = amdgpu.fat_raw_buffer_cast
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: %[[SGID:.*]] = gpu.subgroup_id : index
// CHECK: %[[LID:.*]] = gpu.lane_id
// CHECK: scf.for %[[ARG0:.*]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK:   %[[GATHER_ADDR:.*]] = affine.apply #[[MAP]]()[%[[LID]], %[[SGID]], %[[ARG0]]]
// CHECK:   %[[DELI_GATHER:.*]]:2 = affine.delinearize_index %[[GATHER_ADDR]] into (32, 64) : index, index
// CHECK:   %[[STORE_ADDR:.*]] = affine.apply #[[MAP1]]()[%[[SGID]], %[[ARG0]]]
// CHECK:   %[[DELI_STORE:.*]]:2 = affine.delinearize_index %[[STORE_ADDR]] into (32, 64) : index, index
// CHECK:   iree_gpu.global_load_dma %[[BUFFER]][%[[DELI_GATHER]]#0, %[[DELI_GATHER]]#1] -> %[[ALLOC]][%[[DELI_STORE]]#0, %[[DELI_STORE]]#1]
