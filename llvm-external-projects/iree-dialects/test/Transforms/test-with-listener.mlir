// RUN: iree-dialects-opt --test-listener-canonicalize='listener=1' %s | FileCheck %s --check-prefix CANON
// RUN: iree-dialects-opt --test-listener-cse='listener=1' %s | FileCheck %s --check-prefix CSE

func.func @test_canonicalize(%arg0: i32) -> (i32, i32) {
  // CANON: REPLACED arith.addi
  // CANON: REMOVED arith.addi
  %c5 = arith.constant -5 : i32
  %0 = arith.addi %arg0, %c5 : i32
  %1 = arith.addi %0, %c5 : i32
  return %0, %1 : i32, i32
}

func.func @test_cse(%arg0: i32) -> (i32, i32) {
  // CSE: REPLACED arith.addi
  // CSE: REMOVED arith.addi
  %c5 = arith.constant -5 : i32
  %0 = arith.addi %c5, %arg0 : i32
  %1 = arith.addi %c5, %arg0 : i32
  return %0, %1 : i32, i32
}
