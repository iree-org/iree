// RUN: iree-opt --split-input-file --iree-stream-conversion --canonicalize %s | FileCheck %s

// Basic extern taking a tensor and returning a new tensor.
// During lowering the returned tensor will be turned into an output argument.

// CHECK: stream.async.func private @basicExtern(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*>
flow.func private @basicExtern(%arg0: tensor<?xf32>, %arg1: index) -> tensor<?xf32>

// CHECK-LABEL: @basicCall
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[SIZE0:.+]]: index, %[[DIM0:.+]]: index)
util.func public @basicCall(%arg0: tensor<?xf32>, %dim0: index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<?xf32>{%[[DIM0]]}
  // CHECK: %[[CALL:.+]] = stream.async.call @basicExtern
  // CHECK-SAME: (%[[ARG0]][%c0 to %[[SIZE0]] for %[[SIZE0]]], %[[DIM0]]) : (!stream.resource<*>{%[[SIZE0]]}, index) -> !stream.resource<*>{%[[RESULT_SIZE]]}
  %call = flow.call @basicExtern(%arg0, %dim0) : (tensor<?xf32>{%dim0}, index) -> tensor<?xf32>{%dim0}
  // CHECK: util.return %[[CALL]], %[[RESULT_SIZE]]
  util.return %call : tensor<?xf32>
}

// -----

// Extern that performs an in-place operation on its argument.
// No new tensors will be allocated and the call will receive a single argument.

// CHECK: stream.async.func private @inplaceExtern(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*>
flow.func private @inplaceExtern(%arg0: tensor<?xf32>, %arg1: index) -> tensor<?xf32>

// CHECK-LABEL: @inplaceCall
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[SIZE0:.+]]: index, %[[DIM0:.+]]: index)
util.func public @inplaceCall(%arg0: tensor<?xf32>, %dim0: index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[CALL:.+]] = stream.async.call @inplaceExtern(%[[ARG0]][%c0 to %[[SIZE0]] for %[[SIZE0]]], %[[DIM0]]) : (!stream.resource<*>{%[[SIZE0]]}, index) -> %[[ARG0]]{%[[SIZE0]]}
  %call = flow.call @inplaceExtern(%arg0, %dim0) : (tensor<?xf32>{%dim0}, index) -> %arg0{%dim0}
  // CHECK: util.return %[[CALL]], %[[SIZE0]]
  util.return %call : tensor<?xf32>
}

// -----

// Extern that performs an in-place operation that changes the type as seen by
// the compiler. Here we change both the dimensions and the element type.

// CHECK: stream.async.func private @inplaceTypeChangeExtern(%arg0: !stream.resource<*>, %arg1: index) -> %arg0
flow.func private @inplaceTypeChangeExtern(%arg0: tensor<?x4xf32>, %arg1: index) -> %arg0 as tensor<4x?xi32>

// CHECK-LABEL: @inplaceTypeChangeCall
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[SIZE0:.+]]: index, %[[DIM0:.+]]: index)
util.func public @inplaceTypeChangeCall(%arg0: tensor<?x4xf32>, %dim0: index) -> tensor<4x?xi32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[CALL:.+]] = stream.async.call @inplaceTypeChangeExtern(%[[ARG0]][%c0 to %[[SIZE0]] for %[[SIZE0]]], %[[DIM0]]) : (!stream.resource<*>{%[[SIZE0]]}, index) -> %[[ARG0]]{%[[SIZE0]]}
  %call = flow.call @inplaceTypeChangeExtern(%arg0, %dim0) : (tensor<?x4xf32>{%dim0}, index) -> %arg0 as tensor<4x?xi32>{%dim0}
  // CHECK: util.return %[[CALL]], %[[SIZE0]]
  util.return %call : tensor<4x?xi32>
}
