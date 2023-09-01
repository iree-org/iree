// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @switch
// CHECK-SAME: (%[[INDEX:.+]]: index)
func.func @switch(%index: index) -> i32 {
  // CHECK-DAG: %[[C100:.+]] = arith.constant 100
  %c100 = arith.constant 100 : i32
  // CHECK-DAG: %[[C200:.+]] = arith.constant 200
  %c200 = arith.constant 200 : i32
  // CHECK-DAG: %[[C300:.+]] = arith.constant 300
  %c300 = arith.constant 300 : i32
  // CHECK-DAG: %[[DEFAULT:.+]] = arith.constant 400
  %default = arith.constant 400 : i32
  // CHECK: = util.switch i32 from [%[[C100]], %[[C200]], %[[C300]]] at %[[INDEX]] else %[[DEFAULT]] : i32
  %0 = util.switch i32 from [%c100, %c200, %c300] at %index else %default : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @cast
// CHECK-SAME: (%[[SOURCE:.+]]: !util.buffer)
func.func @cast(%source: !util.buffer) -> !util.object {
  // CHECK: = util.cast %[[SOURCE]] : !util.buffer to !util.object
  %0 = util.cast %source : !util.buffer to !util.object
  return %0 : !util.object
}
