// RUN: iree-opt --split-input-file %s --verify-diagnostics | FileCheck %s

// CHECK: flow.func private @externArg0()
flow.func private @externArg0()
// CHECK-NEXT: flow.func private @externArg1(%arg0: tensor<4x?xi32>)
flow.func private @externArg1(%arg0: tensor<4x?xi32>)
// CHECK-NEXT: flow.func private @externArg1Attrs(%arg0: tensor<4x?xi32> {arg.attr})
flow.func private @externArg1Attrs(%arg0: tensor<4x?xi32> {arg.attr})
// CHECK-NEXT: flow.func private @externArg2(%arg0: tensor<4x?xi32>, %arg1: i32)
flow.func private @externArg2(%arg0: tensor<4x?xi32>, %arg1: i32)
// CHECK-NEXT: flow.func private @externArg2Attrs(%arg0: tensor<4x?xi32> {arg.attr0}, %arg1: i32 {arg.attr1})
flow.func private @externArg2Attrs(%arg0: tensor<4x?xi32> {arg.attr0}, %arg1: i32 {arg.attr1})

// -----

// CHECK: flow.func private @externRet0()
flow.func private @externRet0()
// CHECK-NEXT: flow.func private @externRet1() -> tensor<4x?xi32>
flow.func private @externRet1() -> tensor<4x?xi32>
// CHECK-NEXT: flow.func private @externRet1Attrs() -> (tensor<4x?xi32> {ret.attr})
flow.func private @externRet1Attrs() -> (tensor<4x?xi32> {ret.attr})
// CHECK-NEXT: flow.func private @externRet2() -> (tensor<4x?xi32>, i32)
flow.func private @externRet2() -> (tensor<4x?xi32>, i32)
// CHECK-NEXT: flow.func private @externRet2Attrs() -> (tensor<4x?xi32> {ret.attr0}, i32 {ret.attr1})
flow.func private @externRet2Attrs() -> (tensor<4x?xi32> {ret.attr0}, i32 {ret.attr1})
// CHECK-NEXT: flow.func private @externRetAttributes() -> (tensor<4x?xi32> {ret.attr}) attributes {some.attr = 123 : index}
flow.func private @externRetAttributes() -> (tensor<4x?xi32> {ret.attr}) attributes {some.attr = 123 : index}

// -----

// CHECK: flow.func private @externTied(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1, %arg2 as tensor<?x4xf32>)
flow.func private @externTied(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1, %arg2 as tensor<?x4xf32>)
// CHECK-NEXT: flow.func private @externTiedAttrs(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1 {ret.attr0}, %arg2 as tensor<?x4xf32> {ret.attr1})
flow.func private @externTiedAttrs(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1 {ret.attr0}, %arg2 as tensor<?x4xf32> {ret.attr1})

// -----

// Basic extern taking a tensor and returning a new tensor.
// During lowering the returned tensor will be turned into an output argument.
flow.func private @basicExtern(%arg0: tensor<?xf32>, %arg1: index) -> (tensor<?xf32>, i32)

// CHECK-LABEL: @basicCall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>)
func.func @basicCall(%arg0: tensor<?xf32>) -> (tensor<?xf32>, i32) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %c0
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  // CHECK: %[[CALL:.+]]:2 = flow.call @basicExtern(%[[ARG0]], %[[DIM]]) : (tensor<?xf32>{%[[DIM]]}, index) -> (tensor<?xf32>{%[[DIM]]}, i32)
  %call:2 = flow.call @basicExtern(%arg0, %dim) : (tensor<?xf32>{%dim}, index) -> (tensor<?xf32>{%dim}, i32)
  // CHECK: return %[[CALL]]#0, %[[CALL]]#1
  return %call#0, %call#1 : tensor<?xf32>, i32
}

// -----

// Extern that performs an in-place operation on its argument.
// No new tensors will be allocated and the call will receive a single argument.
flow.func private @inplaceExtern(%arg0: tensor<?xf32>, %arg1: index) -> %arg0

// CHECK-LABEL: @inplaceCall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>)
func.func @inplaceCall(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %c0
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  // CHECK: %[[CALL:.+]] = flow.call @inplaceExtern(%[[ARG0]], %[[DIM]]) : (tensor<?xf32>{%[[DIM]]}, index) -> %[[ARG0]]{%[[DIM]]}
  %call = flow.call @inplaceExtern(%arg0, %dim) : (tensor<?xf32>{%dim}, index) -> %arg0{%dim}
  // CHECK: return %[[CALL]]
  return %call : tensor<?xf32>
}

// -----

// Extern that performs an in-place operation that changes the type as seen by
// the compiler. Here we change both the dimensions and the element type.
flow.func private @inplaceTypeChangeExtern(%arg0: tensor<?x4xf32>, %arg1: index) -> %arg0 as tensor<4x?xi32>

// CHECK-LABEL: @inplaceTypeChangeCall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x4xf32>)
func.func @inplaceTypeChangeCall(%arg0: tensor<?x4xf32>) -> tensor<4x?xi32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %c0
  %dim = tensor.dim %arg0, %c0 : tensor<?x4xf32>
  // CHECK: %[[CALL:.+]] = flow.call @inplaceTypeChangeExtern(%[[ARG0]], %[[DIM]]) : (tensor<?x4xf32>{%[[DIM]]}, index) -> %[[ARG0]] as tensor<4x?xi32>{%[[DIM]]}
  %call = flow.call @inplaceTypeChangeExtern(%arg0, %dim) : (tensor<?x4xf32>{%dim}, index) -> %arg0 as tensor<4x?xi32>{%dim}
  // CHECK: return %[[CALL]]
  return %call : tensor<4x?xi32>
}
