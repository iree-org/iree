// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers -canonicalize %s | IreeFileCheck %s

module {
  func @pad_cst() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    %2 = "mhlo.pad"(%0, %1) {
      edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
      edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
      interior_padding = dense<0> : tensor<2xi64>
    } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<18x12xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
// CHECK_LABEL: @pad_cst
//   CHECK-DAG: %[[CST:.+]] = constant 0.000000e+00 : f32
//   CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<18x12xf32>
//   CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<12x4xf32>
//       CHECK: linalg.fill(%[[OUT]], %[[CST]])
//       CHECK: %[[SUBVIEW:.+]] = subview %[[OUT]][4, 5] [12, 4] [1, 1]
//       CHECK: linalg.copy(%[[IN]], %[[SUBVIEW]])

// -----

module {
  func @pad_memref() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.pad"(%0, %1) {
      edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
      edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
      interior_padding = dense<0> : tensor<2xi64>
    } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<18x12xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK_LABEL: @pad_memref
//   CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<18x12xf32>
//   CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<12x4xf32>
//   CHECK-DAG: %[[PAD_BUF:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
//       CHECK: %[[PAD_VAL:.+]] = load %[[PAD_BUF]][] : memref<f32>
//       CHECK: linalg.fill(%[[OUT]], %[[PAD_VAL]])
//       CHECK: %[[SUBVIEW:.+]] = subview %[[OUT]][4, 5] [12, 4] [1, 1]
//       CHECK: linalg.copy(%[[IN]], %[[SUBVIEW]])

// -----

module {
  func @pad_no_op() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    %2 = "mhlo.pad"(%0, %1) {
      edge_padding_high = dense<0> : tensor<2xi64>,
      edge_padding_low = dense<0> : tensor<2xi64>,
      interior_padding = dense<0> : tensor<2xi64>
    } : (tensor<12x4xf32>, tensor<f32>) -> tensor<12x4xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<12x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
// TODO(hanchung): Make it just a copy op.
// CHECK_LABEL: @pad_no_op
//   CHECK-DAG: %[[CST:.+]] = constant 0.000000e+00 : f32
//   CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<12x4xf32>
//   CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<12x4xf32>
//       CHECK: linalg.fill(%[[OUT]], %[[CST]])
//       CHECK: %[[SUBVIEW:.+]] = subview %[[OUT]][0, 0] [12, 4] [1, 1]
//       CHECK: linalg.copy(%[[IN]], %[[SUBVIEW]])

// -----

module {
  func @cst_pad_cst() {
    %c0 = constant 0 : index
    %0 = constant dense<1.0> : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    %2 = "mhlo.pad"(%0, %1) {
      edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
      edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
      interior_padding = dense<0> : tensor<2xi64>
    } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<18x12xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @ret0, set=0, binding=0, type="StorageBuffer", access="Write"
  }
}
// CHECK_LABEL: @cst_pad_cst
//   CHECK-DAG: %[[ZERO:.+]] = constant 0.000000e+00 : f32
//   CHECK-DAG: %[[ONE:.+]] = constant 1.000000e+00 : f32
//   CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<18x12xf32>
//       CHECK: linalg.fill(%[[OUT]], %[[ZERO]])
//       CHECK: %[[SUBVIEW:.+]] = subview %[[OUT]][4, 5] [12, 4] [1, 1]
//       CHECK: linalg.fill(%[[SUBVIEW]], %[[ONE]])
