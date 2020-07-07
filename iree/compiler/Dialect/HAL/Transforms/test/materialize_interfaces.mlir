// RUN: iree-opt -split-input-file -iree-hal-materialize-interfaces -iree-hal-target-backends=vmla %s | IreeFileCheck %s

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
//   CHECK-DAG: hal.interface @legacy_io {
//  CHECK-NEXT:   hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
//   CHECK-DAG: hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {
//  CHECK-SAME:   interface = @legacy_io,
//  CHECK-SAME:   ordinal = 0 : i32,
//  CHECK-SAME:   signature = (tensor<4xf32>) -> tensor<4xf32>
//  CHECK-SAME: }
//   CHECK-DAG: hal.executable.target "vmla" {
flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  // CHECK: module {
  module {
    // CHECK-NEXT: func @simpleMath_rgn_dispatch_0()
    // CHECK-NEXT:   %[[ZERO:.+]] = constant 0 : index
    // CHECK-NEXT:   %[[ARG0:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %[[ZERO]] : tensor<4xf32>
    // CHECK-NEXT:   %[[RET0:.+]] = call @simpleMath_rgn_dispatch_0_impl(%[[ARG0]]) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT:   hal.interface.store.tensor %[[RET0]], @legacy_io::@ret0, offset = %[[ZERO]] : tensor<4xf32>
    // CHECK-NEXT:   return
    // CHECK-NEXT: }
    // CHECK-NEXT: func @simpleMath_rgn_dispatch_0_impl
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    // CHECK: hal.interface @legacy_io attributes {sym_visibility = "private"}
  }
}

// -----

// CHECK-LABEL: hal.executable @shaped_dispatch
//  CHECK-NEXT: hal.interface @legacy_io attributes {push_constants = 2 : i32} {
//  CHECK-NEXT:   hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
//  CHECK-NEXT: hal.executable.entry_point @entry attributes {
//  CHECK-SAME:   interface = @legacy_io,
//  CHECK-SAME:   ordinal = 0 : i32,
//  CHECK-SAME:   signature = (tensor<?x7x10xf32>, index, index) -> tensor<7x?x10xf32>
//  CHECK-SAME: }
flow.executable @shaped_dispatch {
  flow.dispatch.entry @entry
  // CHECK: module {
  module {
    //      CHECK: func @entry() {
    //  CHECK-NEXT:   %[[ZERO:.+]] = constant 0 : index
    // Invariant: Constant loads emitted before binding (tensor) loads.
    //  CHECK-NEXT:   %[[DIM0:.+]] = hal.interface.load.constant offset = 0 : index
    //  CHECK-NEXT:   %[[DIM1:.+]] = hal.interface.load.constant offset = 1 : index
    //  CHECK-NEXT:   %[[ARG0:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %[[ZERO]] : tensor<?x7x10xf32>
    //  CHECK-NEXT:   %[[RET0:.+]] = call @entry_impl(%[[ARG0]], %[[DIM0]], %[[DIM1]]) : (tensor<?x7x10xf32>, index, index) -> tensor<7x?x10xf32>
    //  CHECK-NEXT:   hal.interface.store.tensor %[[RET0]], @legacy_io::@ret0, offset = %[[ZERO]] : tensor<7x?x10xf32>
    //  CHECK-NEXT:   return
    //  CHECK-NEXT: }
    //  CHECK-NEXT: func @entry_impl
    func @entry(%arg0: tensor<?x7x10xf32>, %arg1: index, %arg2: index) -> tensor<7x?x10xf32> {
      %0 = shapex.make_ranked_shape %arg1 : (index) -> !shapex.ranked_shape<[?,7,10]>
      %1 = shapex.make_ranked_shape %arg2 : (index) -> !shapex.ranked_shape<[7,?,10]>
      %2 = shapex.tie_shape %arg0, %0 : tensor<?x7x10xf32>, !shapex.ranked_shape<[?,7,10]>
      %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<?x7x10xf32>) -> tensor<7x?x10xf32>
      %4 = shapex.tie_shape %3, %1 : tensor<7x?x10xf32>, !shapex.ranked_shape<[7,?,10]>
      return %4 : tensor<7x?x10xf32>
    }
  }
}
