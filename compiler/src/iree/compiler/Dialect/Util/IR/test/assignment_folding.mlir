// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @foldSwitchI32Nop
func.func @foldSwitchI32Nop(%arg0 : index) -> i32 {
  // CHECK: %[[DEFAULT:.+]] = arith.constant 5
  %c5 = arith.constant 5 : i32
  %0 = util.switch i32 from [] at %arg0 else %c5 : i32
  // CHECK: return %[[DEFAULT]] : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @foldSwitchI32Identical
func.func @foldSwitchI32Identical(%arg0 : index) -> i32 {
  // CHECK: %[[C100:.+]] = arith.constant 100
  %c100 = arith.constant 100 : i32
  %0 = util.switch i32 from [%c100, %c100, %c100] at %arg0 else %c100 : i32
  // CHECK: return %[[C100]] : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @foldSwitchI32ConstantIndex
func.func @foldSwitchI32ConstantIndex() -> (i32, i32, i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK-DAG: %[[C100:.+]] = arith.constant 100
  %c100 = arith.constant 100 : i32
  // CHECK-DAG: %[[C200:.+]] = arith.constant 200
  %c200 = arith.constant 200 : i32
  // CHECK-DAG: %[[C300:.+]] = arith.constant 300
  %c300 = arith.constant 300 : i32
  // CHECK-DAG: %[[C400:.+]] = arith.constant 400
  %c400 = arith.constant 400 : i32
  %0 = util.switch i32 from [%c100, %c200, %c300] at %c0 else %c400 : i32
  %1 = util.switch i32 from [%c100, %c200, %c300] at %c1 else %c400 : i32
  %2 = util.switch i32 from [%c100, %c200, %c300] at %c2 else %c400 : i32
  %3 = util.switch i32 from [%c100, %c200, %c300] at %c3 else %c400 : i32
  // CHECK: return %[[C100]], %[[C200]], %[[C300]], %[[C400]] : i32, i32, i32, i32
  return %0, %1, %2, %3 : i32, i32, i32, i32
}
