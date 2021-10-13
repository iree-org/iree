// RUN: iree-opt -split-input-file -iree-convert-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: @tensorLoad
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer
func @tensorLoad(%tensor : tensor<2x3xi32>) {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset<%allocator : !hal.allocator>
  // CHECK-SAME:  indices([%[[C0]], %[[C1]]])
  // CHECK-SAME:  shape([%[[C2]], %[[C3]]])
  // CHECK-SAME:  type(%c16777248_i32)
  // CHECK-NEXT: = hal.buffer.load<%[[BUFFER]] : !hal.buffer>[%[[OFF]]] : i32
  %0 = flow.tensor.load %tensor[%i0, %i1] : tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: @tensorLoad1
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer
func @tensorLoad1(%tensor : tensor<i1>) {
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset<%allocator : !hal.allocator>
  // CHECK-SAME:  indices([])
  // CHECK-SAME:  shape([])
  // CHECK-SAME:  type(%c16777217_i32)
  // CHECK-NEXT: = hal.buffer.load<%[[BUFFER]] : !hal.buffer>[%[[OFF]]] : i1
  %0 = flow.tensor.load %tensor : tensor<i1>
  return
}

// -----

// CHECK-LABEL: @tensorStore
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer
func @tensorStore(%tensor : tensor<2x3xi32>) {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C9:.+]] = arith.constant 9 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %c9 = arith.constant 9 : i32
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset<%allocator : !hal.allocator>
  // CHECK-SAME:  indices([%[[C0]], %[[C1]]])
  // CHECK-SAME:  shape([%[[C2]], %[[C3]]])
  // CHECK-SAME:  type(%c16777248_i32)
  // CHECK-NEXT: hal.buffer.store<%[[BUFFER]] : !hal.buffer>[%[[OFF]]] value(%[[C9]] : i32)
  flow.tensor.store %c9, %tensor[%i0, %i1] : tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: @tensorStore1
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer
func @tensorStore1(%tensor : tensor<i1>) {
  // CHECK-DAG: %[[C1:.+]] = arith.constant true
  %c1 = arith.constant true
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset<%allocator : !hal.allocator>
  // CHECK-SAME:  indices([])
  // CHECK-SAME:  shape([])
  // CHECK-SAME:  type(%c16777217_i32)
  // CHECK-NEXT: hal.buffer.store<%[[BUFFER]] : !hal.buffer>[%[[OFF]]] value(%[[C1]] : i1)
  flow.tensor.store %c1, %tensor : tensor<i1>
  return
}
