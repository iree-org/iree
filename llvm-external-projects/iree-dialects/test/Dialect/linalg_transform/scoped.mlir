// RUN: iree-dialects-opt --test-wrap-scope='opname=arith.addi' %s | FileCheck %s --check-prefix WRAP
// RUN: iree-dialects-opt --test-unwrap-scope %s | FileCheck %s --check-prefix UNWRAP

// WRAP-LABEL: @test_wrap
// WRAP-SAME: (%[[ARG0:.*]]: i32) -> i32
func.func @test_wrap(%arg0: i32) -> i32 {
  // WRAP: %[[V:.*]] = iree_linalg_transform.util.scope(%[[ARG0]], %[[ARG0]]) {
  // WRAP-NEXT: ^[[B:.*]](%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32):
  // WRAP-NEXT: %[[ADD:.*]] = arith.addi %[[ARG2]], %[[ARG2]]
  // WRAP-NEXT: iree_linalg_transform.util.forward %[[ADD]]
  // WRAP-NEXT: } : (i32, i32) -> i32
  %0 = arith.addi %arg0, %arg0 : i32
  // WRAP: return %[[V]]
  return %0 : i32
}

// UNWRAP-LABEL: @test_unwrap
// UNWRAP-SAME: (%[[ARG0:.*]]: i32) -> (i32, i32)
func.func @test_unwrap(%arg0: i32) -> (i32, i32) {
  // UNWRAP: %[[V0:.*]] = arith.addi %[[ARG0]], %[[ARG0]]
  // UNWRAP-NEXT: %[[V1:.*]] = arith.addi %[[V0]], %[[ARG0]]
  %0:2 = iree_linalg_transform.util.scope(%arg0) {
  ^bb0(%arg1: i32):
    %1 = arith.addi %arg1, %arg1 : i32
    %2 = arith.addi %1, %arg1 : i32
    iree_linalg_transform.util.forward %1, %2 : i32, i32
  } : (i32) -> (i32, i32)
  // UNWRAP-NEXT: return %[[V0]], %[[V1]]
  return %0#0, %0#1 : i32, i32
}
