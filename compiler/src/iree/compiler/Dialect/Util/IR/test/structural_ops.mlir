// RUN: iree-opt --split-input-file %s | FileCheck %s

//      CHECK: util.initializer {
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer {
  util.return
}

// -----

//      CHECK: util.initializer attributes {foo} {
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer attributes {foo} {
  util.return
}

// -----

// CHECK: util.initializer {
util.initializer {
  // CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0 : i32
  %zero = arith.constant 0 : i32
  // CHECK-NEXT:   cf.br ^bb1(%[[ZERO]] : i32)
  cf.br ^bb1(%zero: i32)
  // CHECK-NEXT: ^bb1(%0: i32):
^bb1(%0: i32):
  // CHECK-NEXT:   util.return
  util.return
}

// -----

// CHECK: util.func private @externArg0()
util.func private @externArg0()
// CHECK-NEXT: util.func private @externArg1(%arg0: tensor<4x?xi32>)
util.func private @externArg1(%arg0: tensor<4x?xi32>)
// CHECK-NEXT: util.func private @externArg1Attrs(%arg0: tensor<4x?xi32> {arg.attr})
util.func private @externArg1Attrs(%arg0: tensor<4x?xi32> {arg.attr})
// CHECK-NEXT: util.func private @externArg2(%arg0: tensor<4x?xi32>, %arg1: i32)
util.func private @externArg2(%arg0: tensor<4x?xi32>, %arg1: i32)
// CHECK-NEXT: util.func private @externArg2Attrs(%arg0: tensor<4x?xi32> {arg.attr0}, %arg1: i32 {arg.attr1})
util.func private @externArg2Attrs(%arg0: tensor<4x?xi32> {arg.attr0}, %arg1: i32 {arg.attr1})

// -----

// CHECK: util.func private @externRet0()
util.func private @externRet0()
// CHECK-NEXT: util.func private @externRet1() -> tensor<4x?xi32>
util.func private @externRet1() -> tensor<4x?xi32>
// CHECK-NEXT: util.func private @externRet1Attrs() -> (tensor<4x?xi32> {ret.attr})
util.func private @externRet1Attrs() -> (tensor<4x?xi32> {ret.attr})
// CHECK-NEXT: util.func private @externRet2() -> (tensor<4x?xi32>, i32)
util.func private @externRet2() -> (tensor<4x?xi32>, i32)
// CHECK-NEXT: util.func private @externRet2Attrs() -> (tensor<4x?xi32> {ret.attr0}, i32 {ret.attr1})
util.func private @externRet2Attrs() -> (tensor<4x?xi32> {ret.attr0}, i32 {ret.attr1})
// CHECK-NEXT: util.func private @externRetAttributes() -> (tensor<4x?xi32> {ret.attr}) attributes {some.attr = 123 : index}
util.func private @externRetAttributes() -> (tensor<4x?xi32> {ret.attr}) attributes {some.attr = 123 : index}

// -----

// CHECK: util.func private @externTied(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1, %arg2 as tensor<?x4xf32>)
util.func private @externTied(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1, %arg2 as tensor<?x4xf32>)
// CHECK-NEXT: util.func private @externTiedAttrs(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1 {ret.attr0}, %arg2 as tensor<?x4xf32> {ret.attr1})
util.func private @externTiedAttrs(%arg0: i32, %arg1: tensor<4x?xi32>, %arg2: tensor<4x?xi32>) -> (%arg1 {ret.attr0}, %arg2 as tensor<?x4xf32> {ret.attr1})

// -----

// CHECK: util.func private @basicFunc(%arg0: tensor<?xf32>) -> i32
util.func private @basicFunc(%arg0: tensor<?xf32>) -> i32 {
  %c0 = arith.constant 0 : i32
  // CHECK: util.return
  util.return %c0 : i32
}

// -----

// CHECK: util.func private @no_args_callee
util.func private @no_args_callee() -> ()
// CHECK: util.func private @no_args_caller
util.func private @no_args_caller() -> () {
  // CHECK: util.call @no_args_callee() : () -> ()
  util.call @no_args_callee() : () -> ()
  util.return
}

// -----

// CHECK: util.func private @basicExtern
util.func private @basicExtern(%arg0: tensor<?xf32>) -> (tensor<?xf32>, i32)

// CHECK-LABEL: util.func public @basicCall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>) -> (tensor<?xf32>, i32)
util.func @basicCall(%arg0: tensor<?xf32>) -> (tensor<?xf32>, i32) {
  // CHECK: %[[CALL:.+]]:2 = util.call @basicExtern(%[[ARG0]]) : (tensor<?xf32>) -> (tensor<?xf32>, i32)
  %call:2 = util.call @basicExtern(%arg0) : (tensor<?xf32>) -> (tensor<?xf32>, i32)
  // CHECK: util.return %[[CALL]]#0, %[[CALL]]#1
  util.return %call#0, %call#1 : tensor<?xf32>, i32
}

// -----

// CHECK: util.func private @inplaceExtern
util.func private @inplaceExtern(%arg0: tensor<?xf32>) -> %arg0

// CHECK-LABEL: util.func public @inplaceCall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>) -> tensor<?xf32>
util.func @inplaceCall(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[CALL:.+]] = util.call @inplaceExtern(%[[ARG0]]) : (tensor<?xf32>) -> %[[ARG0]]
  %call = util.call @inplaceExtern(%arg0) : (tensor<?xf32>) -> %arg0
  // CHECK: util.return %[[CALL]]
  util.return %call : tensor<?xf32>
}

// -----

// CHECK: util.func private @inplaceTypeChangeExtern
util.func private @inplaceTypeChangeExtern(%arg0: tensor<?x4xf32>) -> %arg0 as tensor<4x?xi32>

// CHECK-LABEL: util.func public @inplaceTypeChangeCall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x4xf32>) -> tensor<4x?xi32>
util.func public @inplaceTypeChangeCall(%arg0: tensor<?x4xf32>) -> tensor<4x?xi32> {
  // CHECK: %[[CALL:.+]] = util.call @inplaceTypeChangeExtern(%[[ARG0]]) : (tensor<?x4xf32>) -> %[[ARG0]] as tensor<4x?xi32>
  %call = util.call @inplaceTypeChangeExtern(%arg0) : (tensor<?x4xf32>) -> %arg0 as tensor<4x?xi32>
  // CHECK: util.return %[[CALL]]
  util.return %call : tensor<4x?xi32>
}
