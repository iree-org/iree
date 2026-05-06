// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-amdgpu-lower-async-dma, cse, canonicalize))" \
// RUN:   --verify-diagnostics \
// RUN:   %s | FileCheck %s

#executable_target = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none,
    subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    dma_sizes = [32, 128]>>}>

#translation = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<VectorDistribute>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>

// CHECK-LABEL: @lower_async_dma_basic
// CHECK-SAME:    %[[SRC:.+]]: memref<16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:.+]]: memref<16x64xf16, #gpu.address_space<workgroup>>
// CHECK-SAME:    %[[I:.+]]: index, %[[J:.+]]: index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[TID:.+]] = gpu.thread_id x
// CHECK:         %[[DELIN:.+]]:3 = affine.delinearize_index %[[TID]] into (8, 8)
//   Batch 0: source indices are divergent, dest indices are uniform.
// CHECK:         %[[SRC_D0_B0:.+]] = affine.linearize_index [%[[DELIN]]#1, %[[I]]] by (8, 1)
// CHECK:         %[[SRC_D1:.+]] = affine.linearize_index [%[[DELIN]]#2, %[[J]]] by (8, 8)
// CHECK:         amdgpu.gather_to_lds %[[SRC]][%[[SRC_D0_B0]], %[[SRC_D1]]], %[[DST]][%[[I]], %[[J]]] : vector<8xf16>
//   Batch 1: source dim0 includes batch offset + thread, dest dim0 includes batch offset only.
// CHECK:         %[[SRC_D0_B1:.+]] = affine.linearize_index [%[[C1]], %[[DELIN]]#1, %[[I]]] by (2, 8, 1)
// CHECK:         %[[DST_D0_B1:.+]] = affine.linearize_index [%[[C1]], %[[C0]], %[[I]]] by (2, 8, 1)
// CHECK:         amdgpu.gather_to_lds %[[SRC]][%[[SRC_D0_B1]], %[[SRC_D1]]], %[[DST]][%[[DST_D0_B1]], %[[J]]] : vector<8xf16>
// CHECK-NOT:     amdgpu.gather_to_lds
// CHECK-NOT:     iree_gpu.async_dma
func.func @lower_async_dma_basic(
    %source: memref<16x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<16x64xf16, #gpu.address_space<workgroup>>,
    %i: index, %j: index)
  attributes {
    hal.executable.target = #executable_target,
    translation_info = #translation} {
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  iree_gpu.async_dma %source[%i, %j] to %dest[%i, %j], vector<16x64xf16>
      : memref<16x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
        memref<16x64xf16, #gpu.address_space<workgroup>>
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  return
}

// -----

#executable_target_2 = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none,
    subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    dma_sizes = [32, 128]>>}>

#translation_2 = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<VectorDistribute>
  workgroup_size = [256, 1, 1]
  subgroup_size = 64>

// CHECK-LABEL: func.func @lower_async_dma_multi_subgroup
// CHECK-SAME:    %[[SRC:.+]]: memref<64x64xf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:.+]]: memref<64x64xf16, #gpu.address_space<workgroup>>
// CHECK-SAME:    %[[I:.+]]: index, %[[J:.+]]: index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[TID:.+]] = gpu.thread_id x
//   Subgroup delinearization: tid -> (_, subgroup_id, lane_id).
// CHECK:         %[[SG_DELIN:.+]]:3 = affine.delinearize_index %[[TID]] into (4, 64)
//   Thread delinearization: tid -> (_, thread_d0, thread_d1).
// CHECK:         %[[TH_DELIN:.+]]:3 = affine.delinearize_index %[[TID]] into (8, 8)
//   Batch 0: src dim0 = subgroup*batch*thread + thread_d0 + i (divergent).
// CHECK:         %[[SRC_D0_B0:.+]] = affine.linearize_index [%[[SG_DELIN]]#1, %[[C0]], %[[TH_DELIN]]#1, %[[I]]] by (4, 2, 8, 1)
// CHECK:         %[[SRC_D1:.+]] = affine.linearize_index [%[[TH_DELIN]]#2, %[[J]]] by (8, 8)
//   Batch 0: dst dim0 = subgroup*batch*thread + i (uniform, no thread offset).
// CHECK:         %[[DST_D0_B0:.+]] = affine.linearize_index [%[[SG_DELIN]]#1, %[[C0]], %[[C0]], %[[I]]] by (4, 2, 8, 1)
// CHECK:         amdgpu.gather_to_lds %[[SRC]][%[[SRC_D0_B0]], %[[SRC_D1]]], %[[DST]][%[[DST_D0_B0]], %[[J]]] : vector<8xf16>
//   Batch 1: batch index advances by 1 in the batch dimension.
// CHECK:         %[[SRC_D0_B1:.+]] = affine.linearize_index [%[[SG_DELIN]]#1, %[[C1]], %[[TH_DELIN]]#1, %[[I]]] by (4, 2, 8, 1)
// CHECK:         %[[DST_D0_B1:.+]] = affine.linearize_index [%[[SG_DELIN]]#1, %[[C1]], %[[C0]], %[[I]]] by (4, 2, 8, 1)
// CHECK:         amdgpu.gather_to_lds %[[SRC]][%[[SRC_D0_B1]], %[[SRC_D1]]], %[[DST]][%[[DST_D0_B1]], %[[J]]] : vector<8xf16>
// CHECK-NOT:     amdgpu.gather_to_lds
// CHECK-NOT:     iree_gpu.async_dma
func.func @lower_async_dma_multi_subgroup(
    %source: memref<64x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<64x64xf16, #gpu.address_space<workgroup>>,
    %i: index, %j: index)
  attributes {
    hal.executable.target = #executable_target_2,
    translation_info = #translation_2} {
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  iree_gpu.async_dma %source[%i, %j] to %dest[%i, %j], vector<64x64xf16>
      : memref<64x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
        memref<64x64xf16, #gpu.address_space<workgroup>>
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  return
}

// -----

#executable_target_3 = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none,
    subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    dma_sizes = [32, 128]>>}>

#translation_3 = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<VectorDistribute>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>

// CHECK-LABEL: func.func @lower_async_dma_fallback_dma_size
// CHECK-SAME:    %[[SRC:.+]]: memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[DST:.+]]: memref<2x64xf16, #gpu.address_space<workgroup>>
// CHECK:         amdgpu.gather_to_lds %[[SRC]]{{.+}}, %[[DST]]{{.+}} : vector<2xf16>
// CHECK-NOT:     amdgpu.gather_to_lds
// CHECK-NOT:     iree_gpu.async_dma
func.func @lower_async_dma_fallback_dma_size(
    %source: memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<2x64xf16, #gpu.address_space<workgroup>>,
    %i: index, %j: index)
  attributes {
    hal.executable.target = #executable_target_3,
    translation_info = #translation_3} {
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  iree_gpu.async_dma %source[%i, %j] to %dest[%i, %j], vector<2x64xf16>
      : memref<2x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
        memref<2x64xf16, #gpu.address_space<workgroup>>
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  return
}

// -----

#executable_target_4 = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none,
    subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    dma_sizes = [32, 128]>>}>

#translation_4 = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<VectorDistribute>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>

func.func @lower_async_dma_rejects_oob_in_bounds(
    %source: memref<16x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<16x64xf16, #gpu.address_space<workgroup>>,
    %i: index, %j: index)
  attributes {
    hal.executable.target = #executable_target_4,
    translation_info = #translation_4} {
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  // expected-error @+1 {{failed to lower async_dma to gather_to_lds}}
  iree_gpu.async_dma %source[%i, %j] to %dest[%i, %j], vector<16x64xf16> in_bounds [false, true]
      : memref<16x64xf16, #amdgpu.address_space<fat_raw_buffer>>,
        memref<16x64xf16, #gpu.address_space<workgroup>>
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  return
}

// -----

#executable_target_5 = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32, storage = b32, subgroup = none,
    subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    dma_sizes = [32, 128]>>}>

#translation_5 = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<VectorDistribute>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>

func.func @lower_async_dma_rejects_noncontiguous_source(
    %source: memref<64x64xf16, strided<[129, 2]>, #amdgpu.address_space<fat_raw_buffer>>,
    %dest: memref<16x64xf16, #gpu.address_space<workgroup>>,
    %i: index, %j: index)
  attributes {
    hal.executable.target = #executable_target_5,
    translation_info = #translation_5} {
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  // expected-error @+1 {{failed to lower async_dma to gather_to_lds}}
  iree_gpu.async_dma %source[%i, %j] to %dest[%i, %j], vector<16x64xf16>
      : memref<64x64xf16, strided<[129, 2]>, #amdgpu.address_space<fat_raw_buffer>>,
        memref<16x64xf16, #gpu.address_space<workgroup>>
  gpu.barrier {addr_space = #gpu.address_space<workgroup>}
  return
}
