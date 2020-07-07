// RUN: iree-opt -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module {
  // CHECK: func @dot
  func @dot() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x3xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<3x2xf32>
    // CHECK: linalg.matmul %{{.+}}, %{{.+}}, %{{.+}} : (memref<2x3xf32>, memref<3x2xf32>, memref<2x2xf32>)
    %result = "mhlo.dot"(%0, %1) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    hal.interface.store.tensor %result, @legacy_io::@ret0, offset = %c0 : tensor<2x2xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
