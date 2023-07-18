// RUN: iree-opt --split-input-file --iree-vm-conversion --cse --iree-vm-target-index-bits=64 %s | FileCheck %s

// CHECK-LABEL: @switch_index
//  CHECK-SAME: (%[[INDEX_I64:.+]]: i64)
func.func @switch_index(%index: index) -> index {
  // CHECK-DAG: %[[C100:.+]] = vm.const.i64 100
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[C200:.+]] = vm.const.i64 200
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[C300:.+]] = vm.const.i64 300
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[INDEX_I32:.+]] = vm.trunc.i64.i32 %[[INDEX_I64]]
  // CHECK: = vm.switch.i64 %[[INDEX_I32]][%[[C100]], %[[C200]]] else %[[C300]] : i64
  %0 = util.switch index from [%c100, %c200] at %index else %c300 : index
  return %0 : index
}

// -----

// CHECK-LABEL: @switch_i32
//  CHECK-SAME: (%[[INDEX_I64:.+]]: i64)
func.func @switch_i32(%index: index) -> i32 {
  // CHECK-DAG: %[[C100:.+]] = vm.const.i32 100
  %c100 = arith.constant 100 : i32
  // CHECK-DAG: %[[C200:.+]] = vm.const.i32 200
  %c200 = arith.constant 200 : i32
  // CHECK-DAG: %[[C300:.+]] = vm.const.i32 300
  %c300 = arith.constant 300 : i32
  // CHECK-DAG: %[[INDEX_I32:.+]] = vm.trunc.i64.i32 %[[INDEX_I64]]
  // CHECK: = vm.switch.i32 %[[INDEX_I32]][%[[C100]], %[[C200]]] else %[[C300]] : i32
  %0 = util.switch i32 from [%c100, %c200] at %index else %c300 : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @switch_f32
//  CHECK-SAME: (%[[INDEX_I64:.+]]: i64)
func.func @switch_f32(%index: index) -> f32 {
  // CHECK-DAG: %[[C100:.+]] = vm.const.f32 1
  %c100 = arith.constant 100.0 : f32
  // CHECK-DAG: %[[C200:.+]] = vm.const.f32 2
  %c200 = arith.constant 200.0 : f32
  // CHECK-DAG: %[[C300:.+]] = vm.const.f32 3
  %c300 = arith.constant 300.0 : f32
  // CHECK-DAG: %[[INDEX_I32:.+]] = vm.trunc.i64.i32 %[[INDEX_I64]]
  // CHECK: = vm.switch.f32 %[[INDEX_I32]][%[[C100]], %[[C200]]] else %[[C300]] : f32
  %0 = util.switch f32 from [%c100, %c200] at %index else %c300 : f32
  return %0 : f32
}
