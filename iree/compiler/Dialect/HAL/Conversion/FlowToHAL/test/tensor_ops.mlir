// RUN: iree-opt -split-input-file -iree-convert-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: @constantTensor
func @constantTensor() {
  // CHECK-NEXT: %dev = hal.ex.shared_device
  // CHECK-NEXT: %allocator = hal.device.allocator %dev
  // CHECK-NEXT: %cbuffer = hal.allocator.allocate.const %allocator, {{.+}} = dense<[1, 2]> : tensor<2xi32>
  %0 = constant dense<[1, 2]> : tensor<2xi32>
  return
}

// -----

// CHECK-LABEL: @constantTensor1
func @constantTensor1() {
  // CHECK-NEXT: %dev = hal.ex.shared_device
  // CHECK-NEXT: %allocator = hal.device.allocator %dev
  // CHECK-NEXT: %cbuffer = hal.allocator.allocate.const %allocator, {{.+}} = dense<[1, 0]> : tensor<2xi8>
  %0 = constant dense<[1, 0]> : tensor<2xi1>
  return
}

// -----

// CHECK-LABEL: @tensorLoad
func @tensorLoad(%arg0 : tensor<2x3xi32>) {
  // CHECK-DAG: %[[C0:.+]] = constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = constant 3 : index
  %i0 = constant 0 : index
  %i1 = constant 1 : index
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset %allocator, shape = [
  // CHECK-SAME:   %[[C2]], %[[C3]]
  // CHECK-SAME: ], element_type = 16777248, indices = [
  // CHECK-SAME:   %[[C0]], %[[C1]]
  // CHECK-SAME: ]
  // CHECK-NEXT: = hal.buffer.load %arg0[
  // CHECK-SAME:   %[[OFF]]
  // CHECK-SAME: ] : i32
  %0 = flow.tensor.load %arg0[%i0, %i1] : tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: @tensorLoad1
func @tensorLoad1(%arg0 : tensor<i1>) {
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset %allocator, shape = [], element_type = 16777217, indices = []
  // CHECK-NEXT: = hal.buffer.load %arg0[
  // CHECK-SAME:   %[[OFF]]
  // CHECK-SAME: ] : i1
  %0 = flow.tensor.load %arg0 : tensor<i1>
  return
}

// -----

// CHECK-LABEL: @tensorStore
func @tensorStore(%arg0 : tensor<2x3xi32>) {
  // CHECK-DAG: %[[C0:.+]] = constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = constant 1 : index
  // CHECK-DAG: %[[C9:.+]] = constant 9 : i32
  // CHECK-DAG: %[[C2:.+]] = constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = constant 3 : index
  %i0 = constant 0 : index
  %i1 = constant 1 : index
  %c9 = constant 9 : i32
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset %allocator, shape = [
  // CHECK-SAME:   %[[C2]], %[[C3]]
  // CHECK-SAME: ], element_type = 16777248, indices = [
  // CHECK-SAME:   %[[C0]], %[[C1]]
  // CHECK-SAME: ]
  // CHECK-NEXT: hal.buffer.store %[[C9]], %arg0[
  // CHECK-SAME:   %[[OFF]]
  // CHECK-SAME: ] : i32
  flow.tensor.store %c9, %arg0[%i0, %i1] : tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: @tensorStore1
func @tensorStore1(%arg0 : tensor<i1>) {
  // CHECK-DAG: %[[C1:.+]] = constant true
  %c1 = constant true
  // CHECK: %[[OFF:.+]] = hal.allocator.compute_offset %allocator, shape = [], element_type = 16777217, indices = []
  // CHECK-NEXT: hal.buffer.store %[[C1]], %arg0[
  // CHECK-SAME:   %[[OFF]]
  // CHECK-SAME: ] : i1
  flow.tensor.store %c1, %arg0 : tensor<i1>
  return
}
