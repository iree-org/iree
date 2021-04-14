// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-pipeline %s | FileCheck %s

module {
  func @bug_2882_repro() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<10xf32>
    %1 = hal.interface.load.tensor @io::@arg1, offset = %c0 : tensor<5xf32>
    %2 = "mhlo.reshape"(%0) : (tensor<10xf32>) -> tensor<1x2x5xf32>
    %3 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<1x2x5xf32>
    %4 =  mhlo.add %2, %3 : tensor<1x2x5xf32>
    %5 = "mhlo.reshape"(%4) : (tensor<1x2x5xf32>) -> (tensor<10xf32>)
    hal.interface.store.tensor %5, @io::@ret0, offset = %c0 : tensor<10xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// CHECK-LABEL: func @bug_2882_repro
//       CHECK:   linalg.generic
//   CHECK-NOT:   linalg.generic
//       CHECK:   return

// -----

module {
  func @bug_2882_repro2() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<1x1x1x1000xf32>
    %1 = hal.interface.load.tensor @io::@arg1, offset = %c0 : tensor<1000xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1000xf32>) -> tensor<1x1x1x1000xf32>
    %3 = mhlo.add %0, %2 : tensor<1x1x1x1000xf32>
    %4 = "mhlo.reshape"(%3) : (tensor<1x1x1x1000xf32>) -> tensor<1x1000xf32>
    hal.interface.store.tensor %3, @io::@ret0, offset = %c0 : tensor<1x1x1x1000xf32>
    hal.interface.store.tensor %4, @io::@ret1, offset = %c0 : tensor<1x1000xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: func @bug_2882_repro2
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0}
//   CHECK-DAG:   %[[RET1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret1}
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg1}
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0}
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] :
//  CHECK-SAME:     outs(%[[RET1]] :
//       CHECK:   linalg.copy(%[[RET1]], %[[RET0]])

// -----

module {
  func @issue_3188() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<4x1x1x512xf32>
    %1 = hal.interface.load.tensor @io::@arg1, offset = %c0 : tensor<512xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<4x1x1x512xf32>
    %3 = mhlo.add %0, %2 : tensor<4x1x1x512xf32>
    %4 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<4x1x1x512xf32>
    %5 = "mhlo.reshape"(%3) : (tensor<4x1x1x512xf32>) -> tensor<1x4x1x512xf32>
    %6 = "mhlo.transpose"(%5) {permutation = dense<[1, 0, 2, 3]> : tensor<4xi64>} : (tensor<1x4x1x512xf32>) -> tensor<4x1x1x512xf32>
    %7 = mhlo.subtract %6, %4 : tensor<4x1x1x512xf32>
    %8 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<4x1x1x512xf32>
    %9 = mhlo.multiply %7, %8 : tensor<4x1x1x512xf32>
    hal.interface.store.tensor %9, @io::@ret0, offset = %c0 : tensor<4x1x1x512xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func @issue_3188()
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4x512xf32>
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<512xf32>
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4x512xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] :
//  CHECK-SAME:     outs(%[[RET0]] :

// -----

module {
  func @issue_3579() {
    %c0 = constant 0 : index
    %cst_1 = constant dense<1.000000e+00> : tensor<1x10xf32>
    %4 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<5x1x1xf32>
    %5 = hal.interface.load.tensor @io::@arg1, offset = %c0 : tensor<i32>
    %6 = "mhlo.torch_index_select"(%4, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<5x1x1xf32>, tensor<i32>) -> tensor<1x1xf32>
    %7 = "mhlo.reshape"(%6) : (tensor<1x1xf32>) -> tensor<1xf32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
    %9 = mhlo.multiply %8, %cst_1 : tensor<1x10xf32>
    hal.interface.store.tensor %9, @io::@ret0, offset = %c0 : tensor<1x10xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// CHECK-LABEL: func @issue_3579
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<5x1x1xf32>
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<i32>
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<10xf32>
//       CHECK:   linalg.indexed_generic
//  CHECK-SAME:     ins(%[[ARG1]] : memref<i32>)
//  CHECK-SAME:     outs(%[[RET0]] : memref<10xf32>
//       CHECK:       load %[[ARG0]]
//       CHECK:       linalg.yield
