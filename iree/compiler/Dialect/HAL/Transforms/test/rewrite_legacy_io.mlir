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

// -----

flow.executable @reduction_ex_reduce_0_dim_0 {
  flow.reduction.entry @reduction_rgn_reduce_0_dim_0_entry apply(@reduction_rgn_reduce_0_dim_0) attributes {dimension = 1 : i32, workgroup_size = dense<[32, 1, 1]> : vector<3xi32>, workload = dense<[4, 1, 1]> : vector<3xi32>}
  module {
    func @reduction_rgn_reduce_0_dim_0_entry() {
      %c0_i32 = constant 0 : i32
      %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0_i32 : tensor<4x8xf32>
      %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0_i32 : tensor<f32>
      %2 = call @reduction_rgn_reduce_0_dim_0_entry_impl(%0, %1) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0_i32 : tensor<4xf32>
      return
    }
    func @reduction_rgn_reduce_0_dim_0_entry_impl(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32> attributes {sym_visibility = "private"}
    func @reduction_rgn_reduce_0_dim_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
      return %0 : tensor<f32>
    }
    hal.interface @legacy_io attributes {sym_visibility = "private"} {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
  }
}

// CHECK-LABEL: flow.executable @reduction_ex_reduce_0_dim_0 {
// CHECK-NEXT:   flow.reduction.entry @reduction_rgn_reduce_0_dim_0_entry apply(@reduction_rgn_reduce_0_dim_0) attributes {dimension = 1 : i32, workgroup_size = dense<[32, 1, 1]> : vector<3xi32>, workload = dense<[4, 1, 1]> : vector<3xi32>}
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @reduction_rgn_reduce_0_dim_0_entry(memref<4x8xf32>, memref<f32>, memref<4xf32>) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_rgn_reduce_0_dim_0, iree.executable.reduction.dimension = 1 : i32}
// CHECK-NEXT:     func @reduction_rgn_reduce_0_dim_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
// CHECK-NEXT:       return %0 : tensor<f32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
