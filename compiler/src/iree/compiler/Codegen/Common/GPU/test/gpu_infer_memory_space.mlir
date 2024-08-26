// RUN: iree-opt %s --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-infer-memory-space))" | FileCheck %s

func.func @write_in_lane_forall(%dest : tensor<4x3xi32>) -> tensor<4x3xi32> {
  %alloc = bufferization.alloc_tensor() : tensor<2x3xi32>
  %cst = arith.constant dense<0> : vector<2x3xi32>
  %c0 = arith.constant 0 : index
  %res = scf.forall (%arg0) in (2) shared_outs(%arg1 = %dest) -> tensor<4x3xi32> {
    %w = vector.transfer_write %cst, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<2x3xi32>, tensor<2x3xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %w into %arg1[%arg0, 0] [2, 3] [1, 1] : tensor<2x3xi32> into tensor<4x3xi32>
    }
  } {mapping = [#iree_gpu.lane_id<0>]}
  return %res : tensor<4x3xi32>
}

// CHECK: func @write_in_lane_forall
// CHECK:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<private>}
// CHECK:   vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @forall_shared_dest(%w : tensor<2x3xi32>) -> tensor<4x3xi32> {
  %dest = bufferization.alloc_tensor() : tensor<4x3xi32>
  %res = scf.forall (%arg0) in (2) shared_outs(%arg1 = %dest) -> tensor<4x3xi32> {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %w into %arg1[%arg0, 0] [2, 3] [1, 1] : tensor<2x3xi32> into tensor<4x3xi32>
    }
  } {mapping = [#gpu.warp<x>]}
  return %res : tensor<4x3xi32>
}

// CHECK: func @forall_shared_dest
// CHECK:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>}
// CHECK:   scf.forall {{.*}} shared_outs(%{{.*}} = %[[ALLOC]])

// -----

func.func @already_annotated_alloc() -> tensor<2x3xi32> {
  %alloc = bufferization.alloc_tensor() {memory_space = #gpu.address_space<private>} : tensor<2x3xi32>
  return %alloc : tensor<2x3xi32>
}

// CHECK: func @already_annotated_alloc
// CHECK:   bufferization.alloc_tensor() {memory_space = #gpu.address_space<private>}

// -----

// expected-error@+1 {{failed to set the gpu memory space for all `bufferization.alloc_tensor` ops}}
func.func @unknown_memory_space() -> tensor<2x3xi32> {
  // expected-error@+1 {{unexpected gpu memory space must be private or workgroup.}}
  %alloc = bufferization.alloc_tensor() {memory_space = "bad"} : tensor<2x3xi32>
  return %alloc : tensor<2x3xi32>
}
