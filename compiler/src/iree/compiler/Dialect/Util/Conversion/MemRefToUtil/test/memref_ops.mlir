// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(iree-util-test-conversion{widen-integers}, cse, canonicalize)' \
// RUN:   --verify-diagnostics %s | FileCheck %s

// -----
// Must be rank-0 or rank-1.
// expected-error @-3 {{conversion to util failed}}
util.func @verify_invalid_rank_2(%buffer: memref<4x2xf32>, %idx: index) -> f32{
  // expected-error @below {{failed to legalize operation 'memref.load'}}
  %0 = memref.load %buffer[%idx, %idx] : memref<4x2xf32>
  util.return %0 : f32
}

// -----
// Must have an identity map.
// expected-error @-3 {{conversion to util failed}}
#map = affine_map<(d0)[s0] -> (d0 * s0)>
util.func @verify_invalid_non_identity_map(%buffer: memref<4xf32, #map>, %idx: index) -> f32 {
  // expected-error @below {{failed to legalize operation 'memref.load'}}
  %0 = memref.load %buffer[%idx] : memref<4xf32, #map>
  util.return %0 : f32
}

// -----
// CHECK-LABEL: @assume_alignment
util.func @assume_alignment(%buffer: memref<?xf32>) {
  // CHECK-NOT: assume_alignment
  memref.assume_alignment %buffer, 64 : memref<?xf32>
  util.return
}

// -----
// CHECK-LABEL: @cast
util.func @cast(%buffer: memref<?xf32>) -> memref<5xf32> {
  // CHECK-NOT: memref.cast
  %0 = memref.cast %buffer : memref<?xf32> to memref<5xf32>
  // CHECK: util.return %arg0 : !util.buffer
  util.return %0 : memref<5xf32>
}

// -----
// CHECK-LABEL: @alloca() -> !util.buffer
util.func @alloca() -> memref<16xi32> {
  // CHECK: %[[ALLOCATION_SIZE:.+]] = arith.constant 64 : index
  // CHECK: %[[BUFFER:.+]] = util.buffer.alloc uninitialized : !util.buffer{%[[ALLOCATION_SIZE]]}
  %0 = memref.alloca() : memref<16xi32>
  // CHECK: util.return %[[BUFFER]]
  util.return %0 : memref<16xi32>
}

// -----
// CHECK-LABEL: @alloca_dynamic_size
// CHECK-SAME: (%[[LENGTH:.+]]: index)
util.func @alloca_dynamic_size(%length : index) -> memref<?xi32> {
  // CHECK: %[[ELEM_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[ALLOCATION_SIZE:.+]] = arith.muli %[[LENGTH]], %[[ELEM_SIZE]] : index
  // CHECK: %[[BUFFER:.+]] = util.buffer.alloc uninitialized : !util.buffer{%[[ALLOCATION_SIZE]]}
  %0 = memref.alloca(%length) : memref<?xi32>
  // CHECK: util.return %[[BUFFER]]
  util.return %0 : memref<?xi32>
}

// -----
// CHECK-LABEL: @alloc_i16
// CHECK-SAME: (%[[IDX0:.+]]: index) -> !util.buffer {
util.func @alloc_i16(%idx0: index) -> memref<4xi16> {
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[BUFFER:.*]] = util.buffer.alloc uninitialized : !util.buffer{%[[C8]]}
  %0 = memref.alloca() : memref<4xi16>
  // CHECK: util.return %[[BUFFER]]
  util.return %0 : memref<4xi16>
}

// -----
// CHECK-LABEL: @alloc_index
// CHECK-SAME: (%[[IDX0:.+]]: index) -> !util.buffer {
util.func @alloc_index(%idx0: index) -> memref<4xindex> {
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[SIZEOF:.*]] = util.sizeof index
  // CHECK: %[[SZ:.*]] = arith.muli %[[SIZEOF]], %[[C4]]
  // CHECK: %[[BUFFER:.*]] = util.buffer.alloc uninitialized : !util.buffer{%[[SZ]]}
  %0 = memref.alloca() : memref<4xindex>
  // CHECK: util.return %[[BUFFER]]
  util.return %0 : memref<4xindex>
}

// -----
// CHECK-LABEL: @load_store_f32
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index) -> f32 {
util.func @load_store_f32(%buffer: memref<?xf32>, %idx0: index, %idx1: index) -> f32 {
  // CHECK: %[[BUFFER_SIZE:.+]] = util.buffer.size %[[BUFFER]]
  // CHECK: %[[IDX0_BYTES:.+]] = arith.muli %[[IDX0]], %c4
  // CHECK: %[[VALUE:.+]] = util.buffer.load %[[BUFFER]][%[[IDX0_BYTES]] for %c4] : !util.buffer{%[[BUFFER_SIZE]]} -> f32
  %0 = memref.load %buffer[%idx0] : memref<?xf32>
  // CHECK: %[[IDX1_BYTES:.+]] = arith.muli %[[IDX1]], %c4
  // CHECK: util.buffer.store %[[VALUE]], %[[BUFFER]][%[[IDX1_BYTES]] for %c4] : f32 -> !util.buffer{%[[BUFFER_SIZE]]}
  memref.store %0, %buffer[%idx1] : memref<?xf32>
  // CHECK: util.return %[[VALUE]] : f32
  util.return %0 : f32
}

// -----
// CHECK: util.global private @__constant_f32 : !util.buffer
// CHECK: util.initializer {
// CHECK:   %[[BUFFER:.+]] = util.buffer.constant : !util.buffer = dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
// CHECK:   util.global.store %[[BUFFER]], @__constant_f32 : !util.buffer
memref.global "private" constant @__constant_f32 : memref<2xf32> = dense<[0.0287729427, 0.0297581609]>

// CHECK-LABEL: @constant_global_f32
// CHECK-SAME: (%[[IDX:.+]]: index) -> f32 {
util.func @constant_global_f32(%idx: index) -> f32 {
  // CHECK: %[[BUFFER:.+]] = util.global.load @__constant_f32 : !util.buffer
  %0 = memref.get_global @__constant_f32 : memref<2xf32>
  // CHECK: %[[BUFFER_SIZE:.+]] = util.buffer.size %[[BUFFER]]
  // CHECK: %[[IDX_BYTES:.+]] = arith.muli %[[IDX]], %c4
  // CHECK: %[[VALUE:.+]] = util.buffer.load %[[BUFFER]][%[[IDX_BYTES]] for %c4] : !util.buffer{%[[BUFFER_SIZE]]} -> f32
  %1 = memref.load %0[%idx] : memref<2xf32>
  // CHECK: util.return %[[VALUE]] : f32
  util.return %1 : f32
}

// -----
// CHECK-LABEL: @load_store_i16
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[VALUE:.+]]: i32) -> i32 {
util.func @load_store_i16(%buffer: memref<?xi16>, %idx0: index, %idx1: index, %value: i16) -> i16 {
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[SZ:.*]] = util.buffer.size %[[BUFFER]]
  // CHECK-DAG: %[[OFS0:.*]] = arith.muli %[[IDX0]], %[[C2]] : index
  // CHECK-DAG: %[[UCST0:.*]] = builtin.unrealized_conversion_cast %[[VALUE]] : i32 to i16
  // CHECK: util.buffer.store %[[UCST0]], %[[BUFFER]][%[[OFS0]] for %[[C2]]] : i16 -> !util.buffer{%[[SZ]]}
  memref.store %value, %buffer[%idx0] : memref<?xi16>
  // CHECK: %[[OFS1:.*]] = arith.muli %[[IDX1]], %[[C2]] : index
  // CHECK: %[[LD:.*]] = util.buffer.load %[[BUFFER]][%[[OFS1]] for %c2] : !util.buffer{%[[SZ]]} -> i16
  // CHECK: %[[UCST1:.*]] = builtin.unrealized_conversion_cast %[[LD]] : i16 to i32
  %1 = memref.load %buffer[%idx1] : memref<?xi16>
  // CHECK: util.return %[[UCST1]]
  util.return %1 : i16
}

// -----
// CHECK-LABEL: @load_store_index
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[VALUE:.+]]: index) -> index {
util.func @load_store_index(%buffer: memref<?xindex>, %idx0: index, %idx1: index, %value: index) -> index {
  // CHECK-DAG: %[[SIZEOF:.*]] = util.sizeof index
  // CHECK-DAG: %[[SZ:.*]] = util.buffer.size %[[BUFFER]]
  // CHECK-DAG: %[[OFS0:.*]] = arith.muli %[[SIZEOF]], %[[IDX0]] : index
  // CHECK: util.buffer.store %[[VALUE]], %[[BUFFER]][%[[OFS0]] for %[[SIZEOF]]] : index -> !util.buffer{%[[SZ]]}
  memref.store %value, %buffer[%idx0] : memref<?xindex>
  // CHECK: %[[OFS1:.*]] = arith.muli %[[SIZEOF]], %[[IDX1]] : index
  // CHECK: %[[LD:.*]] = util.buffer.load %[[BUFFER]][%[[OFS1]] for %[[SIZEOF]]] : !util.buffer{%[[SZ]]} -> index
  %1 = memref.load %buffer[%idx1] : memref<?xindex>
  // CHECK: util.return %[[LD]]
  util.return %1 : index
}

// -----
// CHECK-LABEL: @dim_i16
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[IDX0:.+]]: index) -> index {
util.func @dim_i16(%buffer: memref<?xi16>, %idx0: index) -> index {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[SZ:.*]] = util.buffer.size %[[BUFFER]] : !util.buffer
  // CHECK: %[[DV:.*]] = arith.floordivsi %[[SZ]], %[[C2]] : index
  %0 = memref.dim %buffer, %idx0 : memref<?xi16>
  // CHECK: util.return %[[DV]]
  util.return %0 : index
}

// -----
// CHECK-LABEL: @dim_index
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[IDX0:.+]]: index) -> index {
util.func @dim_index(%buffer: memref<?xindex>, %idx0: index) -> index {
  // CHECK: %[[SIZEOF:.*]] = util.sizeof index
  // CHECK: %[[SZ:.*]] = util.buffer.size %[[BUFFER]] : !util.buffer
  // CHECK: %[[DV:.*]] = arith.floordivsi %[[SZ]], %[[SIZEOF]] : index
  %0 = memref.dim %buffer, %idx0 : memref<?xindex>
  // CHECK: util.return %[[DV]]
  util.return %0 : index
}
