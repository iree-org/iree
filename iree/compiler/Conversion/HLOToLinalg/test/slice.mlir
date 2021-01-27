// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

module {
  // CHECK-LABEL: @slice_whole_buffer
  //   CHECK-NOT: subtensor
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
  // CHECK-LABEL: @slice_whole_stride
  //       CHECK: subtensor %{{.*}}[1, 0] [1, 4] [1, 1] : tensor<3x4xi32> to tensor<1x4xi32>
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
  // CHECK-LABEL: @slice_stride_part
  //       CHECK: subtensor %{{.*}}[1, 1] [1, 2] [1, 1]  : tensor<3x4xi32> to tensor<1x2xi32>
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
