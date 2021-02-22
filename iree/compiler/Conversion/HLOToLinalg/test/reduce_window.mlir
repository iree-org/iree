// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module {
  func @reduce_window_min_nhwc() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x18x18x64xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce_window"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.minimum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
        window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x18x18x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<1x8x8x64xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK-LABEL: func @reduce_window_min_nhwc
// CHECK-DAG:     %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x18x18x64xf32>
// CHECK-DAG:     %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
// CHECK-DAG:     %[[RES:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x8x8x64xf32>
// CHECK:         %[[WINDOW:.+]] = alloc() : memref<1x3x3x1xi32>
// CHECK:         %[[INIT:.+]] = load %[[ARG1]][] : memref<f32>
// CHECK:         linalg.fill(%[[RES]], %[[INIT]]) : memref<1x8x8x64xf32>, f32
// CHECK:         linalg.pooling_min(%[[ARG0]], %[[WINDOW]], %[[RES]])
// CHECK-SAME:      strides = [1, 2, 2, 1]

// -----

module {
  func @reduce_window_max_nhwc() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x18x18x64xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce_window"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
        window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x18x18x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<1x8x8x64xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK-LABEL: func @reduce_window_max_nhwc
// CHECK-DAG:     %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x18x18x64xf32>
// CHECK-DAG:     %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
// CHECK-DAG:     %[[RES:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x8x8x64xf32>
// CHECK:         %[[WINDOW:.+]] = alloc() : memref<1x3x3x1xi32>
// CHECK:         %[[INIT:.+]] = load %[[ARG1]][] : memref<f32>
// CHECK:         linalg.fill(%[[RES]], %[[INIT]]) : memref<1x8x8x64xf32>, f32
// CHECK:         linalg.pooling_max(%[[ARG0]], %[[WINDOW]], %[[RES]])
// CHECK-SAME:      strides = [1, 2, 2, 1]

// -----

module {
  func @reduce_window_add_nhwc() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x18x18x64xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce_window"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
        window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x18x18x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<1x8x8x64xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK-LABEL: func @reduce_window_add_nhwc
// CHECK-DAG:     %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x18x18x64xf32>
// CHECK-DAG:     %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
// CHECK-DAG:     %[[RES:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x8x8x64xf32>
// CHECK:         %[[WINDOW:.+]] = alloc() : memref<1x3x3x1xi32>
// CHECK:         %[[INIT:.+]] = load %[[ARG1]][] : memref<f32>
// CHECK:         linalg.fill(%[[RES]], %[[INIT]]) : memref<1x8x8x64xf32>, f32
// CHECK:         linalg.pooling_sum(%[[ARG0]], %[[WINDOW]], %[[RES]])
// CHECK-SAME:      strides = [1, 2, 2, 1]

// -----

module {
  func @reduce_window_max_nhwc_with_cst() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x18x18x64xf32>
    %1 = constant dense<0xFF800000> : tensor<f32>
    %2 = "mhlo.reduce_window"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
        window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x18x18x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<1x8x8x64xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK-LABEL: func @reduce_window_max_nhwc
// CHECK-DAG:     %[[INIT:.+]] = constant 0xFF800000 : f32
// CHECK-DAG:     %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x18x18x64xf32>
// CHECK-DAG:     %[[RES:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x8x8x64xf32>
// CHECK:         %[[WINDOW:.+]] = alloc() : memref<1x3x3x1xi32>
// CHECK:         linalg.fill(%[[RES]], %[[INIT]]) : memref<1x8x8x64xf32>, f32
// CHECK:         linalg.pooling_max(%[[ARG0]], %[[WINDOW]], %[[RES]])
// CHECK-SAME:      strides = [1, 2, 2, 1]
