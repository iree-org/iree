// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-pipeline -canonicalize %s | IreeFileCheck %s

module {
  // CHECK_LABEL: @slice_whole_buffer
  //  CHECK-NOT: subview
  //      CHECK: linalg.copy
  func @slice_whole_buffer()
    attributes {signature = (tensor<3x4xi32>) -> (tensor<3x4xi32>)} {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<3x4xi32>
    %1 = "mhlo.slice"(%0) {
      start_indices = dense<[0, 0]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<3x4xi32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<3x4xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

// -----

module {
  //      CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1 + 4)>
  //      CHECK: @slice_whole_stride
  //  CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<3x4xi32>
  //      CHECK: subview %[[IN]][1, 0] [1, 4] [1, 1]  : memref<3x4xi32> to memref<1x4xi32, #[[MAP]]>
  //      CHECK: linalg.copy
  func @slice_whole_stride()
    attributes {signature = (tensor<3x4xi32>) -> (tensor<1x4xi32>)} {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<3x4xi32>
    %1 = "mhlo.slice"(%0) {
      start_indices = dense<[1, 0]> : tensor<2xi64>,
      limit_indices = dense<[2, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<1x4xi32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<1x4xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

// -----

module {
  //      CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1 + 5)>
  //      CHECK: @slice_stride_part
  //  CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<3x4xi32>
  //       CHECK: subview %[[IN]][1, 1] [1, 2] [1, 1]  : memref<3x4xi32> to memref<1x2xi32, #[[MAP]]>
  //       CHECK: linalg.copy
  func @slice_stride_part()
    attributes {signature = (tensor<3x4xi32>) -> (tensor<1x2xi32>)} {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<3x4xi32>
    %1 = "mhlo.slice"(%0) {
      start_indices = dense<[1, 1]> : tensor<2xi64>,
      limit_indices = dense<[2, 3]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<1x2xi32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<1x2xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

// -----

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1 + 5)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func @slice_stride_part
//      CHECK: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<1x2xi32>
//      CHECK: %[[IN0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<3x4xi32>
//      CHECK: %[[SUBVIEW:.+]] = subview %[[IN0]][1, 1] [1, 2] [1, 1]  : memref<3x4xi32> to memref<1x2xi32, #[[MAP0]]>
//      CHECK: %[[IN1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<1x2xi32>
//      CHECK: linalg.generic
// CHECK-SAME: ins(%[[SUBVIEW]], %[[IN1]]
// CHECK-SAME: outs(%[[OUT]]
module {
  func @slice_stride_part() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 {operand_result_index = 0 : i32} : tensor<3x4xi32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 {operand_result_index = 1 : i32} : tensor<1x2xi32>
    %2 = "mhlo.slice"(%0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
    %3 = mhlo.add %2, %1 : tensor<1x2xi32>
    hal.interface.store.tensor %3, @legacy_io::@ret0, offset = %c0 {operand_result_index = 2 : i32} : tensor<1x2xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
