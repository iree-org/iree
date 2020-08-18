// RUN: iree-opt -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module {
  // CHECK: func @dot_general
  func @dot_general() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2x3xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x3x4xf32>
    // CHECK: linalg.batch_matmul %{{.+}}, %{{.+}}, %{{.+}} : (memref<2x2x3xf32>, memref<2x3x4xf32>, memref<2x2x4xf32>)
    %result ="mhlo.dot_general"(%0, %1) {
        dot_dimension_numbers = {
            lhs_batching_dimensions = dense<0> : tensor<1xi64>,
            lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
            rhs_batching_dimensions = dense<0> : tensor<1xi64>,
            rhs_contracting_dimensions = dense<1> : tensor<1xi64>
        },
        precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<2x2x3xf32>, tensor<2x3x4xf32>) -> tensor<2x2x4xf32>
    hal.interface.store.tensor %result, @legacy_io::@ret0, offset = %c0 : tensor<2x2x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
