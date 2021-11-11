// RUN: iree-opt -allow-unregistered-dialect -split-input-file -iree-flow-outline-dispatch-regions %s | IreeFileCheck %s

//      CHECK: flow.executable private @staticShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.dispatch.entry public @staticShapeDispatch_dispatch_0 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @staticShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG:.+]]: !flow.dispatch.tensor<readonly:8x4xf32>,
// CHECK-SAME:     %[[RET:.+]]: !flow.dispatch.tensor<writeonly:4x8xf32>) {
//  CHECK-DAG:   %[[ARG_VALUE:.+]] = flow.dispatch.tensor.load %[[ARG]], {{.*}} : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
//  CHECK-DAG:   %[[ARG_SHAPE:.+]] = flow.dispatch.shape %[[ARG]] : !flow.dispatch.tensor<readonly:8x4xf32> -> !shapex.ranked_shape<[8,4]>
//  CHECK-DAG:   %[[RET_SHAPE:.+]] = flow.dispatch.shape %[[RET]] : !flow.dispatch.tensor<writeonly:4x8xf32> -> !shapex.ranked_shape<[4,8]>
// CHECK-NEXT:   %[[RET_VALUE:.+]] = "test.sink"(%[[ARG_VALUE]], %[[ARG_SHAPE]], %[[RET_SHAPE]]) : (tensor<8x4xf32>, !shapex.ranked_shape<[8,4]>, !shapex.ranked_shape<[4,8]>) -> tensor<4x8xf32>
// CHECK-NEXT:   flow.dispatch.tensor.store %[[RET_VALUE]], %[[RET]], {{.*}} : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK-LABEL: func @staticShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func @staticShapeDispatch(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET:.+]] = flow.dispatch @staticShapeDispatch_dispatch_0::@staticShapeDispatch_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> tensor<4x8xf32> = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
    %arg_shape = flow.dispatch.shape %arg : !flow.dispatch.tensor<readonly:8x4xf32> -> !shapex.ranked_shape<[8,4]>
    %ret_shape = flow.dispatch.shape %ret : !flow.dispatch.tensor<writeonly:4x8xf32> -> !shapex.ranked_shape<[4,8]>
    %ret_value = "test.sink"(%arg_value, %arg_shape, %ret_shape) : (tensor<8x4xf32>, !shapex.ranked_shape<[8,4]>, !shapex.ranked_shape<[4,8]>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret,  offsets=[], sizes=[], strides=[] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
    flow.return
  }
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_0
// CHECK-NEXT:   flow.dispatch.entry public @dispatchFnMuli_dispatch_0 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @dispatchFnMuli_dispatch_0(

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_1
// CHECK-NEXT:   flow.dispatch.entry public @dispatchFnMuli_dispatch_1 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @dispatchFnMuli_dispatch_1(

// CHECK-LABEL: func @dispatchFnMuli(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func @dispatchFnMuli(%arg0 : tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET0:.+]] = flow.dispatch @dispatchFnMuli_dispatch_0::@dispatchFnMuli_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
    %arg_shape = flow.dispatch.shape %arg : !flow.dispatch.tensor<readonly:8x4xf32> -> !shapex.ranked_shape<[8,4]>
    %ret_shape = flow.dispatch.shape %ret : !flow.dispatch.tensor<writeonly:4x8xf32> -> !shapex.ranked_shape<[4,8]>
    %ret_value = "test.sink1"(%arg_value, %arg_shape, %ret_shape) : (tensor<8x4xf32>, !shapex.ranked_shape<[8,4]>, !shapex.ranked_shape<[4,8]>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[], sizes=[], strides=[] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
    flow.return
  }
  // CHECK: %[[RET1:.+]] = flow.dispatch @dispatchFnMuli_dispatch_1::@dispatchFnMuli_dispatch_1[
  // CHECK-SAME: %[[Y]], %[[X]]
  // CHECK-SAME: ](%[[RET0]]) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = flow.dispatch.workgroups[%y, %x](%0) : (tensor<4x8xf32>) -> (tensor<8x4xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:4x8xf32>, %ret: !flow.dispatch.tensor<writeonly:8x4xf32>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:4x8xf32> -> tensor<8x4xf32>
    %arg_shape = flow.dispatch.shape %arg : !flow.dispatch.tensor<readonly:4x8xf32> -> !shapex.ranked_shape<[4,8]>
    %ret_shape = flow.dispatch.shape %ret : !flow.dispatch.tensor<writeonly:8x4xf32> -> !shapex.ranked_shape<[8,4]>
    %ret_value = "test.sink2"(%arg_value, %arg_shape, %ret_shape) : (tensor<8x4xf32>, !shapex.ranked_shape<[4,8]>, !shapex.ranked_shape<[8,4]>) -> (tensor<8x4xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[], sizes=[], strides=[] : tensor<8x4xf32> -> !flow.dispatch.tensor<writeonly:8x4xf32>
    flow.return
  }
  // CHECK-NEXT: return %[[RET1]]
  return %1 : tensor<8x4xf32>
}

// -----

// CHECK: flow.executable private @dispatchFn1_dispatch_0

// CHECK-LABEL: func @dispatchFn1
func @dispatchFn1(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn1_dispatch_0::@dispatchFn1_dispatch_0
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// CHECK: flow.executable private @dispatchFn2_dispatch_0

// CHECK-LABEL: func @dispatchFn2
func @dispatchFn2(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn2_dispatch_0::@dispatchFn2_dispatch_0
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dynamicShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.dispatch.entry public @dynamicShapeDispatch_dispatch_0 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @dynamicShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG:.+]]: !flow.dispatch.tensor<readonly:7x?x24x?xf32>,
// CHECK-SAME:     %[[RET:.+]]: !flow.dispatch.tensor<writeonly:?x?x1024xf32>,
// CHECK-SAME:     %[[IN_ARG_DIM1:.+]]: index, %[[IN_ARG_DIM3:.+]]: index, %[[IN_RET_DIM0:.+]]: index, %[[IN_RET_DIM1:.+]]: index) {

//      CHECK: %[[IN_ARG_SHAPE:.+]] = shapex.make_ranked_shape %[[IN_ARG_DIM1]], %[[IN_ARG_DIM3]]
// CHECK-NEXT: %[[ARG_SHAPED:.+]] = flow.dispatch.tie_shape %[[ARG]], %[[IN_ARG_SHAPE]]
//      CHECK: %[[IN_RET_SHAPE:.+]] = shapex.make_ranked_shape %[[IN_RET_DIM0]], %[[IN_RET_DIM1]]
// CHECK-NEXT: %[[RET_SHAPED:.+]] = flow.dispatch.tie_shape %[[RET]], %[[IN_RET_SHAPE]]

//      CHECK: %[[ARG_SHAPE:.+]] = flow.dispatch.shape %[[ARG_SHAPED]]
//  CHECK-DAG: %[[ARG_DIM1:.+]] = shapex.ranked_dim %[[ARG_SHAPE]][1]
//  CHECK-DAG: %[[ARG_DIM3:.+]] = shapex.ranked_dim %[[ARG_SHAPE]][3]
// CHECK-NEXT: "test.sink_shape_arg"(%[[ARG_DIM1]], %[[ARG_DIM3]])
//      CHECK: %[[RET_SHAPE:.+]] = flow.dispatch.shape %[[RET_SHAPED]]
//  CHECK-DAG: %[[RET_DIM0:.+]] = shapex.ranked_dim %[[RET_SHAPE]][0]
//  CHECK-DAG: %[[RET_DIM1:.+]] = shapex.ranked_dim %[[RET_SHAPE]][1]
// CHECK-NEXT: "test.sink_shape_ret"(%[[RET_DIM0]], %[[RET_DIM1]])

//      CHECK: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG_SHAPED]], {{.*}}
// CHECK-NEXT: %[[ARG_TILE_SHAPE:.+]] = shapex.get_ranked_shape %[[ARG_TILE]]
// CHECK-NEXT: %[[RET_TILE:.+]] = "test.tile_math"(%[[ARG_TILE]], %[[ARG_TILE_SHAPE]], %[[RET_SHAPE]])
// CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET_SHAPED]], {{.*}}

// CHECK:   return
// CHECK-NEXT: }

// CHECK-LABEL: func @dynamicShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<7x?x24x?xf32>)
func @dynamicShapeDispatch(%arg0 : tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // CHECK-DAG: %[[ARG0_DIM1:.+]] = tensor.dim %[[ARG0]], %c1
  %dim1 = tensor.dim %arg0, %c1 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[ARG0_DIM3:.+]] = tensor.dim %[[ARG0]], %c3
  %dim3 = tensor.dim %arg0, %c3 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[X:.+]] = arith.constant 1024
  %x = arith.constant 1024 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 512
  %y = arith.constant 512 : index
  // CHECK-NEXT: %[[RET0:.+]] = flow.dispatch @dynamicShapeDispatch_dispatch_0::@dynamicShapeDispatch_dispatch_0[
  // CHECK-SAME:   %[[X]], %[[Y]]
  // CHECK-SAME: ](%arg0, %[[ARG0_DIM1]], %[[ARG0_DIM3]], %[[ARG0_DIM3]], %[[ARG0_DIM1]])
  // CHECK-SAME: : (tensor<7x?x24x?xf32>{%[[ARG0_DIM1]], %[[ARG0_DIM3]]}, index, index, index, index) -> tensor<?x?x1024xf32>{%[[ARG0_DIM3]], %[[ARG0_DIM1]]}
  %ret0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim3, %dim1} = (
    %arg: !flow.dispatch.tensor<readonly:7x?x24x?xf32>, %ret: !flow.dispatch.tensor<writeonly:?x?x1024xf32>
  ) {
    %workgroup_rank = flow.dispatch.workgroup.rank : index

    %arg_shape = flow.dispatch.shape %arg : !flow.dispatch.tensor<readonly:7x?x24x?xf32> -> !shapex.ranked_shape<[7,?,24,?]>
    %arg_dim1 = shapex.ranked_dim %arg_shape[1] : !shapex.ranked_shape<[7,?,24,?]> -> index
    %arg_dim3 = shapex.ranked_dim %arg_shape[3] : !shapex.ranked_shape<[7,?,24,?]> -> index
    "test.sink_shape_arg"(%arg_dim1, %arg_dim3) : (index, index) -> ()

    %ret_shape = flow.dispatch.shape %ret : !flow.dispatch.tensor<writeonly:?x?x1024xf32> -> !shapex.ranked_shape<[?,?,1024]>
    %ret_dim0 = shapex.ranked_dim %ret_shape[0] : !shapex.ranked_shape<[?,?,1024]> -> index
    %ret_dim1 = shapex.ranked_dim %ret_shape[1] : !shapex.ranked_shape<[?,?,1024]> -> index
    "test.sink_shape_ret"(%ret_dim0, %ret_dim1) : (index, index) -> ()

    %arg_tile = flow.dispatch.tensor.load %arg, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:7x?x24x?xf32> -> tensor<7x?x24x?xf32>
    %arg_tile_shape = shapex.get_ranked_shape %arg_tile : tensor<7x?x24x?xf32> -> !shapex.ranked_shape<[7,?,24,?]>

    %ret_tile = "test.tile_math"(%arg_tile, %arg_tile_shape, %ret_shape) :
        (tensor<7x?x24x?xf32>, !shapex.ranked_shape<[7,?,24,?]>, !shapex.ranked_shape<[?,?,1024]>) -> (tensor<?x?x1024xf32>)

    flow.dispatch.tensor.store %ret_tile, %ret, offsets=[], sizes=[], strides=[] : tensor<?x?x1024xf32> -> !flow.dispatch.tensor<writeonly:?x?x1024xf32>

    flow.return
  }
  // CHECK-NEXT: return %[[RET0]]
  return %ret0 : tensor<?x?x1024xf32>
}
