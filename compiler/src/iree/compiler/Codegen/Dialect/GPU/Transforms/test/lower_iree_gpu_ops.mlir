// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-gpu-lower-ops))" %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @lower_coalesced_gather_dma_in_forall
func.func @lower_coalesced_gather_dma_in_forall(%indices: memref<32xindex>, %source: memref<2048xf32>, %dest: memref<128xf32, #gpu.address_space<workgroup>>) {
  // CHECK-NOT: scf.forall
  // CHECK: amdgpu.gather_to_lds %{{.*}}[], %{{.*}}[] : i32, memref<2048xf32>, memref<128xf32, #gpu.address_space<workgroup>>
  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK-NOT: scf.forall
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32>, memref<128xf32, #gpu.address_space<workgroup>> -> memref<128xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return
}

// -----

// scf.forall with wrong loop bounds should not be lowered
// CHECK-LABEL: func.func @forall_wrong_bounds_not_lowered
func.func @forall_wrong_bounds_not_lowered(%indices: memref<32xindex>, %source: memref<2048xf32>, %dest: memref<128xf32, #gpu.address_space<workgroup>>) {
  // CHECK: scf.forall
  // CHECK: iree_gpu.coalesced_gather_dma
  scf.forall (%arg5, %arg6) in (16, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32>, memref<128xf32, #gpu.address_space<workgroup>> -> memref<128xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return
}

// -----

// scf.forall with wrong mapping should not be lowered
// CHECK-LABEL: func.func @forall_wrong_mapping_not_lowered
func.func @forall_wrong_mapping_not_lowered(%indices: memref<32xindex>, %source: memref<2048xf32>, %dest: memref<128xf32, #gpu.address_space<workgroup>>) {
  // CHECK: scf.forall
  // CHECK: iree_gpu.coalesced_gather_dma
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<2048xf32>, memref<128xf32, #gpu.address_space<workgroup>> -> memref<128xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
  return
}


// -----

// scf.forall with dynamic shapes should not be lowered
// CHECK-LABEL: func.func @forall_dynamic_shapes_not_lowered
func.func @forall_dynamic_shapes_not_lowered(%indices: memref<32xindex>, %source: memref<?xf32>, %dest: memref<128xf32, #gpu.address_space<workgroup>>) {
  // CHECK: scf.forall
  // CHECK: iree_gpu.coalesced_gather_dma
  scf.forall (%arg5, %arg6) in (32, 1) {
    %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest : memref<32xindex>, memref<?xf32>, memref<128xf32, #gpu.address_space<workgroup>> -> memref<128xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return
}
