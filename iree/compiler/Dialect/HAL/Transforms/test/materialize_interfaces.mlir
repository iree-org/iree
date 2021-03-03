// RUN: iree-opt -allow-unregistered-dialect -split-input-file -iree-hal-materialize-interfaces -iree-hal-target-backends=vmla %s | IreeFileCheck %s

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
//   CHECK-DAG: hal.interface @legacy_io {
//  CHECK-NEXT:   hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
//   CHECK-DAG: hal.executable.target @vmla, filter="vmla" {
//   CHECK-DAG:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {
//  CHECK-SAME:     interface = @legacy_io,
//  CHECK-SAME:     ordinal = 0 : i32,
//  CHECK-SAME:     signature = (tensor<4xf32>) -> tensor<4xf32>
//  CHECK-SAME:   }
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
    // CHECK-NEXT: func private @simpleMath_rgn_dispatch_0_impl
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    // CHECK: hal.interface @legacy_io attributes {sym_visibility = "private"}
  }
}

// -----

// CHECK-LABEL: hal.executable @bools_ex_dispatch_0
//   CHECK-DAG: hal.interface @legacy_io {
//  CHECK-NEXT:   hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
//   CHECK-DAG: hal.executable.target @vmla, filter="vmla" {
//   CHECK-DAG:   hal.executable.entry_point @bools_rgn_dispatch_0 attributes {
//  CHECK-SAME:     interface = @legacy_io,
//  CHECK-SAME:     ordinal = 0 : i32,
//  CHECK-SAME:     signature = (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
//  CHECK-SAME:   }
flow.executable @bools_ex_dispatch_0 {
  flow.dispatch.entry @bools_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  // CHECK: module {
  module {
    // CHECK-NEXT: func @bools_rgn_dispatch_0()
    //  CHECK-DAG:   %[[ZERO:.+]] = constant 0 : index
    //  CHECK-DAG:   %[[ARG0_I8:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %[[ZERO]] : tensor<4xi8>
    //  CHECK-DAG:   %[[ARG0_I1:.+]] = "mhlo.convert"(%[[ARG0_I8]]) : (tensor<4xi8>) -> tensor<4xi1>
    //  CHECK-DAG:   %[[ARG1_I8:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %[[ZERO]] : tensor<4xi8>
    //  CHECK-DAG:   %[[ARG1_I1:.+]] = "mhlo.convert"(%[[ARG1_I8]]) : (tensor<4xi8>) -> tensor<4xi1>
    // CHECK-NEXT:   %[[RET0_I1:.+]] = call @bools_rgn_dispatch_0_impl(%[[ARG0_I1]], %[[ARG1_I1]]) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    // CHECK-NEXT:   %[[RET0_I8:.+]] = "mhlo.convert"(%[[RET0_I1]]) : (tensor<4xi1>) -> tensor<4xi8>
    // CHECK-NEXT:   hal.interface.store.tensor %[[RET0_I8]], @legacy_io::@ret0, offset = %[[ZERO]] : tensor<4xi8>
    // CHECK-NEXT:   return
    // CHECK-NEXT: }
    // CHECK-NEXT: func private @bools_rgn_dispatch_0_impl(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1>
    func @bools_rgn_dispatch_0(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
      %0 = mhlo.and %arg0, %arg1 : tensor<4xi1>
      %c = mhlo.constant dense<[false, false, true, false]> : tensor<4xi1>
      %1 = mhlo.and %0, %c : tensor<4xi1>
      return %1 : tensor<4xi1>
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
    //  CHECK-NEXT: func private @entry_impl
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

// -----

// CHECK-LABEL: hal.executable @static_tiled_dispatch
//  CHECK-NEXT: hal.interface @legacy_io {
//  CHECK-NEXT:   hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @wo1, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
flow.executable @static_tiled_dispatch {
  // CHECK-NEXT: hal.executable.target @vmla, filter="vmla" {
  // CHECK-NEXT:   hal.executable.entry_point @entry attributes {
  // CHECK-SAME:     interface = @legacy_io,
  // CHECK-SAME:     ordinal = 0 : i32,
  // CHECK-SAME:     signature = (!flow.dispatch.tensor<readonly:8x4xf32>, !flow.dispatch.tensor<writeonly:4x8xf32>) -> ()
  // CHECK-SAME:   }
  flow.dispatch.entry @entry attributes {
    signature = (tensor<8x4xf32>) -> tensor<4x8xf32>,
    workgroup_rank = 2 : index
  }
  // CHECK-NEXT: module  {
  module  {
    // CHECK-NEXT: func @entry() {
    func @entry(%arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>) {
      // CHECK-NEXT: %c0 = constant 0 : index
      // CHECK-NEXT: %[[ARG:.+]] = hal.interface.binding.subspan @legacy_io::@ro0[%c0] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-NEXT: %[[RET:.+]] = hal.interface.binding.subspan @legacy_io::@wo1[%c0] : !flow.dispatch.tensor<writeonly:4x8xf32>

      // CHECK-NEXT: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG]]
      %arg_tile = flow.dispatch.tensor.load %arg : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[RET_TILE:.+]] = "test.sink"(%[[ARG_TILE]])
      %ret_tile = "test.sink"(%arg_tile) : (tensor<8x4xf32>) -> tensor<4x8xf32>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET]]
      flow.dispatch.tensor.store %ret_tile, %ret : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
      return
    }
  }
}

// -----

// CHECK-LABEL: hal.executable @dynamic_tiled_dispatch
//  CHECK-NEXT: hal.interface @legacy_io attributes {push_constants = 4 : i32} {
//  CHECK-NEXT:   hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @wo1, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
flow.executable @dynamic_tiled_dispatch {
  // CHECK-NEXT: hal.executable.target @vmla, filter="vmla" {
  // CHECK-NEXT:   hal.executable.entry_point @entry attributes {
  // CHECK-SAME:     interface = @legacy_io,
  // CHECK-SAME:     ordinal = 0 : i32,
  // CHECK-SAME:     signature = (!flow.dispatch.tensor<readonly:7x?x24x?xf32>, !flow.dispatch.tensor<writeonly:?x?x1024xf32>, index, index, index, index) -> ()
  // CHECK-SAME:   }
  flow.dispatch.entry @entry attributes {
    signature = (tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32>,
    workgroup_rank = 2 : index
  }
  // CHECK-NEXT: module  {
  module  {
    // CHECK-NEXT: func @entry() {
    func @entry(
        // CHECK-NEXT: %c0 = constant 0 : index
        // CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan @legacy_io::@ro0[%c0] : !flow.dispatch.tensor<readonly:7x?x24x?xf32>
        %arg: !flow.dispatch.tensor<readonly:7x?x24x?xf32>,
        // CHECK-DAG: %[[RET:.+]] = hal.interface.binding.subspan @legacy_io::@wo1[%c0] : !flow.dispatch.tensor<writeonly:?x?x1024xf32>
        %ret: !flow.dispatch.tensor<writeonly:?x?x1024xf32>,
        // CHECK-DAG: %[[ARG_DIM1:.+]] = hal.interface.load.constant offset = 0 : index
        %arg_dim1: index,
        // CHECK-DAG: %[[ARG_DIM3:.+]] = hal.interface.load.constant offset = 1 : index
        %arg_dim3: index,
        // CHECK-DAG: %[[RET_DIM0:.+]] = hal.interface.load.constant offset = 2 : index
        %ret_dim0: index,
        // CHECK-DAG: %[[RET_DIM1:.+]] = hal.interface.load.constant offset = 3 : index
        %ret_dim1: index
      ) {
      // CHECK-NEXT: %[[ARG_SHAPE:.+]] = shapex.make_ranked_shape %[[ARG_DIM1]], %[[ARG_DIM3]]
      %arg_shape = shapex.make_ranked_shape %arg_dim1, %arg_dim3 : (index, index) -> !shapex.ranked_shape<[7,?,24,?]>
      // CHECK-NEXT: %[[ARG_SHAPED:.+]] = flow.dispatch.tie_shape %[[ARG]], %[[ARG_SHAPE]]
      %arg_shaped = flow.dispatch.tie_shape %arg, %arg_shape : (!flow.dispatch.tensor<readonly:7x?x24x?xf32>, !shapex.ranked_shape<[7,?,24,?]>) -> !flow.dispatch.tensor<readonly:7x?x24x?xf32>
      // CHECK-NEXT: %[[RET_SHAPE:.+]] = shapex.make_ranked_shape %[[RET_DIM0]], %[[RET_DIM1]]
      %ret_shape = shapex.make_ranked_shape %ret_dim0, %ret_dim1 : (index, index) -> !shapex.ranked_shape<[?,?,1024]>
      // CHECK-NEXT: %[[RET_SHAPED:.+]] = flow.dispatch.tie_shape %[[RET]], %[[RET_SHAPE]]
      %ret_shaped = flow.dispatch.tie_shape %ret, %ret_shape : (!flow.dispatch.tensor<writeonly:?x?x1024xf32>, !shapex.ranked_shape<[?,?,1024]>) -> !flow.dispatch.tensor<writeonly:?x?x1024xf32>
      // CHECK-NEXT: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG_SHAPED]]
      %arg_tile = flow.dispatch.tensor.load %arg_shaped : !flow.dispatch.tensor<readonly:7x?x24x?xf32> -> tensor<7x?x24x?xf32>
      // CHECK-NEXT: %[[RET_TILE:.+]] = "test.tile_math"(%[[ARG_TILE]])
      %ret_tile = "test.tile_math"(%arg_tile) : (tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET_SHAPED]]
      flow.dispatch.tensor.store %ret_tile, %ret_shaped : tensor<?x?x1024xf32> -> !flow.dispatch.tensor<writeonly:?x?x1024xf32>
      return
    }
  }
}

// -----

// CHECK-LABEL: hal.executable @workgroup_infos
//  CHECK-NEXT: hal.interface @legacy_io {
//  CHECK-NEXT:   hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @wo1, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
flow.executable @workgroup_infos {
  // CHECK-NEXT: hal.executable.target @vmla, filter="vmla" {
  // CHECK-NEXT:   hal.executable.entry_point @entry attributes {
  // CHECK-SAME:     interface = @legacy_io,
  // CHECK-SAME:     ordinal = 0 : i32,
  // CHECK-SAME:     signature = (!flow.dispatch.tensor<readonly:8x4xf32>, !flow.dispatch.tensor<writeonly:4x8xf32>) -> ()
  // CHECK-SAME:   }
  flow.dispatch.entry @entry attributes {
    signature = (tensor<8x4xf32>) -> tensor<4x8xf32>,
    workgroup_rank = 2 : index
  }
  // CHECK-NEXT: module  {
  module  {
    // CHECK-NEXT: func @entry() {
    func @entry(%arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>) {
      // CHECK-DAG: %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
      %id_x = flow.dispatch.workgroup.id[0] : index
      // CHECK-DAG: %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
      %id_y = flow.dispatch.workgroup.id[1] : index
      // CHECK-DAG: %[[WORKGROUP_COUNT_X:.+]] = hal.interface.workgroup.count[0] : index
      %count_x = flow.dispatch.workgroup.count[0] : index
      // CHECK-DAG: %[[WORKGROUP_COUNT_Y:.+]] = hal.interface.workgroup.count[1] : index
      %count_y = flow.dispatch.workgroup.count[1] : index
      // CHECK-DAG: %[[WORKGROUP_SIZE_X:.+]] = hal.interface.workgroup.size[0] : index
      %size_x = flow.dispatch.workgroup.size[0] : index
      // CHECK-DAG: %[[WORKGROUP_SIZE_Y:.+]] = hal.interface.workgroup.size[1] : index
      %size_y = flow.dispatch.workgroup.size[1] : index
      // CHECK-NEXT: "test.sink"(%[[WORKGROUP_ID_X]], %[[WORKGROUP_ID_Y]], %[[WORKGROUP_COUNT_X]], %[[WORKGROUP_COUNT_Y]], %[[WORKGROUP_SIZE_X]], %[[WORKGROUP_SIZE_Y]])
      "test.sink"(%id_x, %id_y, %count_x, %count_y, %size_x, %size_y) : (index, index, index, index, index, index) -> ()
      return
    }
  }
}
