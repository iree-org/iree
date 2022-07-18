// RUN: iree-opt --split-input-file --iree-util-test-conversion --cse --canonicalize --verify-diagnostics %s | FileCheck %s

// Must be rank-0 or rank-1.

// expected-error @-5 {{conversion to util failed}}
func.func @verify_invalid_rank_2(%buffer: memref<4x2xf32>, %idx: index) {
  // expected-error @below {{failed to legalize operation 'memref.load'}}
  memref.load %buffer[%idx, %idx] : memref<4x2xf32>
  return
}

// -----

// Must have an identity map.

#map = affine_map<(d0)[s0] -> (d0 * s0)>
// expected-error @-6 {{conversion to util failed}}
func.func @verify_invalid_non_identity_map(%buffer: memref<4xf32, #map>, %idx: index) {
  // expected-error @below {{failed to legalize operation 'memref.load'}}
  memref.load %buffer[%idx] : memref<4xf32, #map>
  return
}

// -----

// CHECK-LABEL: @assume_alignment
func.func @assume_alignment(%buffer: memref<?xf32>) {
  // CHECK-NOT: assume_alignment
  memref.assume_alignment %buffer, 64 : memref<?xf32>
  func.return
}

// -----

// CHECK-LABEL: @cast
func.func @cast(%buffer: memref<?xf32>) -> memref<5xf32> {
  // CHECK-NOT: memref.cast
  %0 = memref.cast %buffer : memref<?xf32> to memref<5xf32>
  // CHECK: return %arg0 : !util.buffer
  func.return %0 : memref<5xf32>
}

// -----

// CHECK-LABEL: @alloca() -> !util.buffer
func.func @alloca() -> memref<16xi32> {
  // CHECK: %[[ALLOCATION_SIZE:.+]] = arith.constant 64 : index
  // CHECK: %[[BUFFER:.+]] = util.buffer.alloc uninitialized : !util.buffer{%[[ALLOCATION_SIZE]]}
  %0 = memref.alloca() : memref<16xi32>
  // CHECK: return %[[BUFFER]]
  return %0 : memref<16xi32>
}

// -----

// CHECK-LABEL: @load_store_f32
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index) -> f32 {
func.func @load_store_f32(%buffer: memref<?xf32>, %idx0: index, %idx1: index) -> f32 {
  // CHECK: %[[BUFFER_SIZE:.+]] = util.buffer.size %[[BUFFER]]
  // CHECK: %[[IDX0_BYTES:.+]] = arith.muli %[[IDX0]], %c4
  // CHECK: %[[VALUE:.+]] = util.buffer.load %[[BUFFER]][%[[IDX0_BYTES]]] : !util.buffer{%[[BUFFER_SIZE]]} -> f32
  %0 = memref.load %buffer[%idx0] : memref<?xf32>
  // CHECK: %[[IDX1_BYTES:.+]] = arith.muli %[[IDX1]], %c4
  // CHECK: util.buffer.store %[[VALUE]], %[[BUFFER]][%[[IDX1_BYTES]]] : f32 -> !util.buffer{%[[BUFFER_SIZE]]}
  memref.store %0, %buffer[%idx1] : memref<?xf32>
  // CHECK: return %[[VALUE]] : f32
  return %0 : f32
}

// -----

// CHECK: util.global private @__constant_f32 : !util.buffer
// CHECK: util.initializer {
// CHECK:   %[[BUFFER:.+]] = util.buffer.constant : !util.buffer = dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
// CHECK:   util.global.store %[[BUFFER]], @__constant_f32 : !util.buffer
memref.global "private" constant @__constant_f32 : memref<2xf32> = dense<[0.0287729427, 0.0297581609]>

// CHECK-LABEL: @constant_global_f32
// CHECK-SAME: (%[[IDX:.+]]: index) -> f32 {
func.func @constant_global_f32(%idx: index) -> f32 {
  // CHECK: %[[BUFFER:.+]] = util.global.load @__constant_f32 : !util.buffer
  %0 = memref.get_global @__constant_f32 : memref<2xf32>
  // CHECK: %[[BUFFER_SIZE:.+]] = util.buffer.size %[[BUFFER]]
  // CHECK: %[[IDX_BYTES:.+]] = arith.muli %[[IDX]], %c4
  // CHECK: %[[VALUE:.+]] = util.buffer.load %[[BUFFER]][%[[IDX_BYTES]]] : !util.buffer{%[[BUFFER_SIZE]]} -> f32
  %1 = memref.load %0[%idx] : memref<2xf32>
  // CHECK: return %[[VALUE]] : f32
  return %1 : f32
}
