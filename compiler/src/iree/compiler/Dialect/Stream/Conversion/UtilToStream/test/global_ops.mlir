// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK: util.global public mutable @var_i32 : !stream.resource<variable>
// CHECK: util.global public mutable @var_i32__size : index
util.global public mutable @var_i32 : tensor<i32>
// CHECK-LABEL: @mutableGlobal
func.func @mutableGlobal() {
  // CHECK-DAG: %[[VAR:.+]] = util.global.load @var_i32 : !stream.resource<variable>
  // CHECK-DAG: %[[SIZE:.+]] = util.global.load @var_i32__size : index
  //     CHECK: %[[LOAD_T:.+]] = stream.async.transfer %[[VAR]] : !stream.resource<variable>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = util.global.load @var_i32 : tensor<i32>
  //     CHECK: %[[STORE_T:.+]] = stream.async.transfer %[[LOAD_T]] : !stream.resource<*>{%[[SIZE]]} -> !stream.resource<variable>{%[[SIZE]]}
  // CHECK-DAG: util.global.store %[[STORE_T]], @var_i32 : !stream.resource<variable>
  // CHECK-DAG: util.global.store %[[SIZE]], @var_i32__size : index
  util.global.store %0, @var_i32 : tensor<i32>
  return
}

// -----

// TODO(#7432): add indirect global expansion support to streams.
// util.global public mutable @var_indirect : tensor<i32>
// func.func @mutableGlobalIndirect() {
//   %0 = util.global.address @var_indirect : !util.ptr<tensor<i32>>
//   %1 = util.global.load.indirect %0 : !util.ptr<tensor<i32>> -> tensor<i32>
//   util.global.store.indirect %1, %0 : tensor<i32> -> !util.ptr<tensor<i32>>
//   return
// }

// -----

//  CHECK-DAG: util.global public mutable @var_with_tensor_initializer : !stream.resource<variable>
//  CHECK-DAG: util.global public mutable @var_with_tensor_initializer__size : index
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[CST:.+]] = stream.tensor.constant : tensor<f32> in !stream.resource<variable> = dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %[[SIZE:.+]] = stream.resource.size %[[CST]] : !stream.resource<variable>
//  CHECK-DAG:   util.global.store %[[CST]], @var_with_tensor_initializer : !stream.resource<variable>
//  CHECK-DAG:   util.global.store %[[SIZE]], @var_with_tensor_initializer__size : index
util.global public mutable @var_with_tensor_initializer = dense<0.000000e+00> : tensor<f32>
// CHECK-LABEL: @initializedGlobal
func.func @initializedGlobal() {
  // CHECK-DAG: = util.global.load @var_with_tensor_initializer : !stream.resource<variable>
  // CHECK-DAG: = util.global.load @var_with_tensor_initializer__size : index
  %0 = util.global.load @var_with_tensor_initializer : tensor<f32>
  // CHECK-DAG: util.global.store %{{.+}}, @var_with_tensor_initializer : !stream.resource<variable>
  // CHECK-DAG: util.global.store %{{.+}}, @var_with_tensor_initializer__size : index
  util.global.store %0, @var_with_tensor_initializer : tensor<f32>
  return
}

// -----

//  CHECK-DAG: util.global private mutable @var_with_tensor_uninitialized : !stream.resource<variable>
//  CHECK-DAG: util.global private mutable @var_with_tensor_uninitialized__size : index
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[SIZE:.+]] = stream.tensor.sizeof tensor<4xf32>
// CHECK-NEXT:   %[[EMPTY:.+]] = stream.tensor.empty : tensor<4xf32> in !stream.resource<variable>{%[[SIZE]]}
//  CHECK-DAG:   util.global.store %[[EMPTY]], @var_with_tensor_uninitialized : !stream.resource<variable>
//  CHECK-DAG:   util.global.store %[[SIZE]], @var_with_tensor_uninitialized__size : index
util.global private mutable @var_with_tensor_uninitialized = #util.uninitialized : tensor<4xf32>
// CHECK-LABEL: @uninitializedGlobalTensor
func.func @uninitializedGlobalTensor() {
  // CHECK-DAG: = util.global.load @var_with_tensor_uninitialized : !stream.resource<variable>
  // CHECK-DAG: = util.global.load @var_with_tensor_uninitialized__size : index
  %0 = util.global.load @var_with_tensor_uninitialized : tensor<4xf32>
  // CHECK-DAG: util.global.store %{{.+}}, @var_with_tensor_uninitialized : !stream.resource<variable>
  // CHECK-DAG: util.global.store %{{.+}}, @var_with_tensor_uninitialized__size : index
  util.global.store %0, @var_with_tensor_uninitialized : tensor<4xf32>
  return
}

// -----

// Checks that the implicit cast allowing a buffer_view to store into a variable
// that maps to a buffer is permitted.

// CHECK-DAG: util.global public mutable @var_with_buffer_view_store : !stream.resource<variable>
// CHECK-DAG: util.global public mutable @var_with_buffer_view_store__size : index
util.global public mutable @var_with_buffer_view_store : tensor<?x4xf32>
// CHECK-LABEL: @globalStoreFromExternal
func.func @globalStoreFromExternal(%arg0: !hal.buffer_view) {
  // CHECK: %[[DIM0:.+]] = hal.buffer_view.dim
  %dim0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  // CHECK: %[[SIZE:.+]] = stream.tensor.sizeof tensor<?x4xf32>{%[[DIM0]]} : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x4xf32>{%[[DIM0]]} in !stream.resource<external>{%[[SIZE]]}
  // CHECK: %[[T:.+]] = stream.async.transfer %[[IMPORT]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x4xf32>{%dim0}
  // CHECK: %[[VAR:.+]] = stream.async.transfer %[[T]] : !stream.resource<*>{%[[SIZE]]} -> !stream.resource<variable>{%[[SIZE]]}
  // CHECK: util.global.store %[[VAR]], @var_with_buffer_view_store : !stream.resource<variable>
  // CHECK: util.global.store %[[SIZE]], @var_with_buffer_view_store__size : index
  util.global.store %0, @var_with_buffer_view_store : tensor<?x4xf32>
  return
}

// -----

// Checks that the implicit cast allowing a buffer_view to indirect store into
// a variable that maps to a buffer is permitted.

// TODO(#7432): add indirect global expansion support to streams.
// util.global public mutable @var_indirect_with_buffer_view_store : tensor<i32>
// func.func @globalStoreFromExternalIndirect(%arg0: !hal.buffer_view) {
//   %0 = util.global.address @var_indirect_with_buffer_view_store : !util.ptr<tensor<i32>>
//   %1 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<i32>
//   util.global.store.indirect %1, %0 : tensor<i32> -> !util.ptr<tensor<i32>>
//   return
// }
