// RUN: iree-opt -split-input-file -iree-hal-rewrite-legacy-io %s | IreeFileCheck %s

flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {workload = dense<[4, 1, 1]> : vector<3xi32>}
  module {
    func @simpleMath_rgn_dispatch_0() {
      %c0_i32 = constant 0 : i32
      %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0_i32 : tensor<4xf32>
      %1 = call @simpleMath_rgn_dispatch_0_impl(%0) : (tensor<4xf32>) -> tensor<4xf32>
      hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0_i32 : tensor<4xf32>
      return
    }
    func @simpleMath_rgn_dispatch_0_impl(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {sym_visibility = "private"} {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    hal.interface @legacy_io attributes {sym_visibility = "private"} {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
    }
  }
}

// CHECK-LABEL: flow.executable @simpleMath_ex_dispatch_0 {
// CHECK-NEXT:   flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {workload = dense<[4, 1, 1]> : vector<3xi32>}
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @simpleMath_rgn_dispatch_0(%arg0: memref<4xf32>, %arg1: memref<4xf32>) attributes {iree.executable.export} {
// CHECK-NEXT:       %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
// CHECK-NEXT:       %1 = xla_hlo.add %0, %0 : tensor<4xf32>
// CHECK-NEXT:       iree.store_output(%1 : tensor<4xf32>, %arg1 : memref<4xf32>)
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
