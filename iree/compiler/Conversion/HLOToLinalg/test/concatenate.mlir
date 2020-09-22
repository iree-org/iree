// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers -canonicalize %s | IreeFileCheck %s

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1 + 2)>
// CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x5xi32>
// CHECK-DAG: %[[IN0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x2xi32>
// CHECK-DAG: %[[IN1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<2x3xi32>
//     CHECK: %[[SUB0:.+]] = subview %[[OUT]][0, 0] [2, 2] [1, 1]  : memref<2x5xi32> to memref<2x2xi32, #[[MAP0]]>
//     CHECK: linalg.copy(%[[IN0]], %[[SUB0]])
//     CHECK: %[[SUB1:.+]] = subview %[[OUT]][0, 2] [2, 3] [1, 1]  : memref<2x5xi32> to memref<2x3xi32, #[[MAP1]]>
//     CHECK: linalg.copy(%[[IN1]], %[[SUB1]])
module {
  func @concatenate() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x3xi32>
    %2 = "mhlo.concatenate"(%0, %1) {
      dimension = 1
    } : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<2x5xi32>
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
//     CHECK: %[[SUB0:.+]] = subview %[[OUT]][0, 0] [2, 2] [1, 1]  : memref<5x2xi32> to memref<2x2xi32, #[[MAP0]]>
//     CHECK: linalg.copy(%[[IN]], %[[SUB0]])
//     CHECK: %[[SUB1:.+]] = subview %[[OUT]][2, 0] [3, 2] [1, 1]  : memref<5x2xi32> to memref<3x2xi32, #[[MAP1]]>
//     CHECK: linalg.fill(%[[SUB1]], %[[CST]])

module {
  func @concatenate() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = constant dense<42> : tensor<3x2xi32>
    %2 = "mhlo.concatenate"(%0, %1) {
      dimension = 0
    } : (tensor<2x2xi32>, tensor<3x2xi32>) -> tensor<5x2xi32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<5x2xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
