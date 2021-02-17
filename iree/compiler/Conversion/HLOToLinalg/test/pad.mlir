// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors -canonicalize %s | IreeFileCheck %s

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
// CHECK-LABEL: func @pad_cst
//   CHECK-DAG: %[[PAD:.+]] = constant 0.000000e+00 : f32
//   CHECK-DAG: %[[C4:.+]] = constant 4 : index
//   CHECK-DAG: %[[C2:.+]] = constant 2 : index
//   CHECK-DAG: %[[C5:.+]] = constant 5 : index
//   CHECK-DAG: %[[C3:.+]] = constant 3 : index
//   CHECK-DAG: %[[IN:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
//       CHECK: linalg.pad_tensor %[[IN]] low[%[[C4]], %[[C5]]] high[%[[C2]], %[[C3]]]
//       CHECK:  linalg.yield %[[PAD]] : f32
//       CHECK: } : tensor<12x4xf32> to tensor<18x12xf32>

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
// CHECK-LABEL: func @pad_memref
//   CHECK-DAG: %[[C4:.+]] = constant 4 : index
//   CHECK-DAG: %[[C2:.+]] = constant 2 : index
//   CHECK-DAG: %[[C5:.+]] = constant 5 : index
//   CHECK-DAG: %[[C3:.+]] = constant 3 : index
//   CHECK-DAG: %[[IN1:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
//   CHECK-DAG: %[[IN2:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
//   CHECK-DAG: %[[PAD:.+]] = tensor.extract %[[IN2]][] : tensor<f32>
//       CHECK: linalg.pad_tensor %[[IN1]] low[%[[C4]], %[[C5]]] high[%[[C2]], %[[C3]]]
//       CHECK:  linalg.yield %[[PAD]] : f32
//       CHECK: } : tensor<12x4xf32> to tensor<18x12xf32>

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
// CHECK-LABEL: func @pad_no_op
//   CHECK-NOT:   linalg.pad_tensor

// -----

module {
  func @cst_pad_cst() {
    %c0 = constant 0 : index
    %0 = constant dense<1.0> : tensor<1xf32>
    %1 = constant dense<0.0> : tensor<f32>
    %2 = "mhlo.pad"(%0, %1) {
      edge_padding_high = dense<[1]> : tensor<1xi64>,
      edge_padding_low = dense<[2]> : tensor<1xi64>,
      interior_padding = dense<0> : tensor<1xi64>
    } : (tensor<1xf32>, tensor<f32>) -> tensor<4xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @ret0, set=0, binding=0, type="StorageBuffer", access="Write"
  }
}
// CHECK-LABEL: func @cst_pad_cst
//       CHECK:   constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<4xf32>
//   CHECK-NOT:   linalg.pad_tensor
