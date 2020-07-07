// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers -cse %s | IreeFileCheck %s

module {
  // CHECK_LABEL: @slice_whole_buffer
  //  CHECK-NOT: subview
  //      CHECK: linalg.copy
  func @slice_whole_buffer() {
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
  //      CHECK: #[[MAP:.+]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
  //      CHECK: @slice_whole_stride
  //  CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x4xi32>
  //  CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<3x4xi32>
  //  CHECK-DAG: %[[ZERO:.+]] = constant 0 : index
  //  CHECK-DAG: %[[ONE:.+]] = constant 1 : index
  //  CHECK-DAG: %[[DIM0:.+]] = dim %[[OUT]], %[[ZERO]] : memref<1x4xi32>
  //  CHECK-DAG: %[[DIM1:.+]] = dim %[[OUT]], %[[ONE]] : memref<1x4xi32>
  //      CHECK: subview %[[IN]]
  // CHECK-SAME:   [%[[ONE]], %[[ZERO]]]
  // CHECK-SAME:   [%[[DIM0]], %[[DIM1]]]
  // CHECK-SAME:   [%[[ONE]], %[[ONE]]]
  // CHECK-SAME: : memref<3x4xi32> to memref<?x?xi32, #[[MAP]]>
  //      CHECK: linalg.copy
  func @slice_whole_stride() {
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
  //      CHECK: #[[MAP:.+]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
  //      CHECK: @slice_stride_part
  //  CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x2xi32>
  //  CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<3x4xi32>
  //  CHECK-DAG: %[[ZERO:.+]] = constant 0 : index
  //  CHECK-DAG: %[[ONE:.+]] = constant 1 : index
  //  CHECK-DAG: %[[DIM0:.+]] = dim %[[OUT]], %[[ZERO]] : memref<1x2xi32>
  //  CHECK-DAG: %[[DIM1:.+]] = dim %[[OUT]], %[[ONE]] : memref<1x2xi32>
  //      CHECK: subview %[[IN]]
  // CHECK-SAME:   [%[[ONE]], %[[ONE]]]
  // CHECK-SAME:   [%[[DIM0]], %[[DIM1]]]
  // CHECK-SAME:   [%[[ONE]], %[[ONE]]]
  // CHECK-SAME: : memref<3x4xi32> to memref<?x?xi32, #map0>
  //       CHECK: linalg.copy
  func @slice_stride_part() {
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
