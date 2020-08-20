// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-pipeline %s | FileCheck %s

module {
  func @bug_2882_repro() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<10xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<5xf32>
    %2 = "mhlo.reshape"(%0) : (tensor<10xf32>) -> tensor<1x2x5xf32>
    %3 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<1x2x5xf32>
    %4 =  mhlo.add %2, %3 : tensor<1x2x5xf32>
    %5 = "mhlo.reshape"(%4) : (tensor<1x2x5xf32>) -> (tensor<10xf32>)
    hal.interface.store.tensor %5, @legacy_io::@ret0, offset = %c0 : tensor<10xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// CHECK-LABEL: func @bug_2882_repro
//       CHECK:   linalg.generic
//   CHECK-NOT:   linalg.generic
//       CHECK:   return
