// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers -canonicalize %s | IreeFileCheck %s

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1 + 2)>
// CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x5xi32>
// CHECK-DAG: %[[IN0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x2xi32>
// CHECK-DAG: %[[IN1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<2x3xi32>
//     CHECK: %[[SUB0:.+]] = memref.subview %[[OUT]][0, 0] [2, 2] [1, 1]  : memref<2x5xi32> to memref<2x2xi32, #[[MAP0]]>
//     CHECK: linalg.copy(%[[IN0]], %[[SUB0]])
//     CHECK: %[[SUB1:.+]] = memref.subview %[[OUT]][0, 2] [2, 3] [1, 1]  : memref<2x5xi32> to memref<2x3xi32, #[[MAP1]]>
//     CHECK: linalg.copy(%[[IN1]], %[[SUB1]])
module  {
  func @concatenate() {
    %c1 = constant 1 : index
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x3xi32>
    %2 = linalg.init_tensor [2, 5] : tensor<2x5xi32>
    %3 = linalg.fill(%2, %c0_i32) : tensor<2x5xi32>, i32 -> tensor<2x5xi32>
    %4 = subtensor_insert %0 into %3[%c0, %c0] [%c2, %c2] [%c1, %c1] : tensor<2x2xi32> into tensor<2x5xi32>
    %5 = subtensor_insert %1 into %4[%c0, %c2] [%c2, %c3] [%c1, %c1] : tensor<2x3xi32> into tensor<2x5xi32>
    hal.interface.store.tensor %5, @legacy_io::@ret0, offset = %c0 : tensor<2x5xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1 + 4)>
// CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<5x2xi32>
// CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x2xi32>
// CHECK-DAG: %[[CST:.+]] = constant 42 : i32
//     CHECK: %[[SUB0:.+]] = memref.subview %[[OUT]][0, 0] [2, 2] [1, 1]  : memref<5x2xi32> to memref<2x2xi32, #[[MAP0]]>
//     CHECK: linalg.copy(%[[IN]], %[[SUB0]])
//     CHECK: %[[SUB1:.+]] = memref.subview %[[OUT]][2, 0] [3, 2] [1, 1]  : memref<5x2xi32> to memref<3x2xi32, #[[MAP1]]>
//     CHECK: linalg.fill(%[[SUB1]], %[[CST]])
module  {
  func @concatenate() {
    %cst = constant dense<42> : tensor<3x2xi32>
    %c1 = constant 1 : index
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = linalg.init_tensor [5, 2] : tensor<5x2xi32>
    %2 = linalg.fill(%1, %c0_i32) : tensor<5x2xi32>, i32 -> tensor<5x2xi32>
    %3 = subtensor_insert %0 into %2[%c0, %c0] [%c2, %c2] [%c1, %c1] : tensor<2x2xi32> into tensor<5x2xi32>
    %4 = subtensor_insert %cst into %3[%c2, %c0] [%c3, %c2] [%c1, %c1] : tensor<3x2xi32> into tensor<5x2xi32>
    hal.interface.store.tensor %4, @legacy_io::@ret0, offset = %c0 : tensor<5x2xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
