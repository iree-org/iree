// RUN: iree-opt --split-input-file %s | FileCheck %s
//
// This file tests split boundary CHECK warnings.

// First case is fine (no split boundary before it).
// CHECK-LABEL: @test0
util.func @test0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// BAD: first CHECK after split is not CHECK-LABEL (warning: first_check_not_label_after_split).
// Also BAD: CHECK-DAG before CHECK-LABEL (warning: unanchored_check_dag_after_split).
// CHECK-DAG: util.global
// CHECK-LABEL: @test1
util.global private @global1 : index
util.func @test1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// BAD: first CHECK after split is CHECK (not LABEL).
// CHECK-LABEL: @test2_anchor
// CHECK: some.op
// CHECK-LABEL: @test2
util.func @test2_anchor() {}
util.func @test2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  "some.op"() : () -> ()
  util.return %arg0 : tensor<4xf32>
}

// -----

// BAD: first CHECK after split is CHECK-SAME (not LABEL).
// CHECK-SAME: something
// CHECK-LABEL: @test3
util.func @test3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// BAD: first CHECK after split is CHECK-NEXT (not LABEL).
// CHECK-NEXT: arith.constant
// CHECK-LABEL: @test4
util.func @test4(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}

// -----

// GOOD: CHECK-LABEL first, then CHECK-DAG.
// CHECK-LABEL: @test5
// CHECK-DAG: util.global
util.global private @global5 : index
util.func @test5(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// GOOD: CHECK-LABEL on module is valid anchoring.
// CHECK-LABEL: module
module {
  // CHECK: util.global
  util.global private @nested_global : index
}

// -----

// GOOD: CHECK-LABEL first (this case tests that first case with LABEL is fine).
// CHECK-LABEL: @test_good_label
util.func @test_good_label(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}
