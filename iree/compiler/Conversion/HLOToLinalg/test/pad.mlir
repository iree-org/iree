// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module {
  // CHECK_LABEL: @pad_cst
  func @pad_cst() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
    // CHECK: linalg.indexed_generic
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

// -----

module {
  // CHECK_LABEL: @pad_memref
  func @pad_memref() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    // CHECK: linalg.indexed_generic
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

// -----

module {
  // CHECK_LABEL: @pad_no_op
  func @pad_no_op() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12x4xf32>
    // CHECK: linalg.indexed_generic
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
