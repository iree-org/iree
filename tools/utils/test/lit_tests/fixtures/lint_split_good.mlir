// RUN: iree-opt --split-input-file %s | FileCheck %s
//
// This file tests that proper split boundary usage doesn't trigger warnings.

// First case - CHECK-LABEL first (no split boundary before it).
// CHECK-LABEL: @test0
util.func @test0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// GOOD: CHECK-LABEL first, then other CHECKs.
// CHECK-LABEL: @test1
// CHECK: arith.constant
util.func @test1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}

// -----

// GOOD: CHECK-LABEL first, then CHECK-DAG.
// CHECK-LABEL: @test2
// CHECK-DAG: util.global
util.global private @global2 : index
util.func @test2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// GOOD: CHECK-LABEL on module is valid anchoring.
// CHECK-LABEL: module
module {
  // CHECK: util.global
  util.global private @nested_global : index
}
