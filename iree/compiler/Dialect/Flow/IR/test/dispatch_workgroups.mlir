// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @complexWorkgroupsUsage
func @complexWorkgroupsUsage(
    // CHECK-SAME: %[[ARG0:.+]]: tensor<?x4xf32>
    %arg0 : tensor<?x4xf32>,
    // CHECK-SAME: %[[ARG1:.+]]: index
    %arg1 : index) -> tensor<4x?xf32> {
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[WORKGROUP_COUNT_X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[WORKGROUP_COUNT_Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[OUTER_RET0:.+]] = flow.dispatch.workgroups[
  // CHECK-SAME: %[[WORKGROUP_COUNT_X]], %[[WORKGROUP_COUNT_Y]]
  // CHECK-SAME: ](%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: : (tensor<?x4xf32>{%c128}, index) -> tensor<4x?xf32>{%c128} =
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %arg1) : (tensor<?x4xf32>{%c128}, index) -> tensor<4x?xf32>{%c128} =
  // CHECK-NEXT: (%[[INNER_ARG0:.+]]: !flow.dispatch.tensor<readonly:?x4xf32>
  // CHECK-SAME:  %[[INNER_ARG1:.+]]: index
  // CHECK-SAME:  %[[INNER_RET0:.+]]: !flow.dispatch.tensor<writeonly:4x?xf32>) {
  (%arg0_capture: !flow.dispatch.tensor<readonly:?x4xf32>, %arg1_capture: index, %ret0: !flow.dispatch.tensor<writeonly:4x?xf32>) {

    // Query symbolic workgroup info:

    // CHECK: flow.dispatch.workgroup.rank : index
    %workgroup_rank = flow.dispatch.workgroup.rank : index

    // CHECK-DAG: flow.dispatch.workgroup.id[0] : index
    // CHECK-DAG: flow.dispatch.workgroup.id[1] : index
    // CHECK-DAG: flow.dispatch.workgroup.count[0] : index
    // CHECK-DAG: flow.dispatch.workgroup.count[1] : index
    // CHECK-DAG: flow.dispatch.workgroup.size[0] : index
    // CHECK-DAG: flow.dispatch.workgroup.size[1] : index
    %id_x = flow.dispatch.workgroup.id[0] : index
    %id_y = flow.dispatch.workgroup.id[1] : index
    %count_x = flow.dispatch.workgroup.count[0] : index
    %count_y = flow.dispatch.workgroup.count[1] : index
    %size_x = flow.dispatch.workgroup.size[0] : index
    %size_y = flow.dispatch.workgroup.size[1] : index

    // Query shapes directly from IO (static dims will fold):

    //      CHECK: %[[ARG0_SHAPE:.+]] = flow.dispatch.shape %[[INNER_ARG0]] : !flow.dispatch.tensor<readonly:?x4xf32> -> !shapex.ranked_shape<[?,4]>
    %arg0_shape = flow.dispatch.shape %arg0_capture : !flow.dispatch.tensor<readonly:?x4xf32> -> !shapex.ranked_shape<[?,4]>
    //  CHECK-DAG: %[[ARG0_DIM0:.+]] = shapex.ranked_dim %[[ARG0_SHAPE]][0] : !shapex.ranked_shape<[?,4]> -> index
    %arg0_dim0 = shapex.ranked_dim %arg0_shape[0] : !shapex.ranked_shape<[?,4]> -> index
    //  CHECK-DAG: %[[ARG0_DIM1:.+]] = shapex.ranked_dim %[[ARG0_SHAPE]][1] : !shapex.ranked_shape<[?,4]> -> index
    %arg0_dim1 = shapex.ranked_dim %arg0_shape[1] : !shapex.ranked_shape<[?,4]> -> index
    // CHECK-NEXT: "test.sink"(%[[ARG0_DIM0]], %[[ARG0_DIM1]])
    "test.sink"(%arg0_dim0, %arg0_dim1) : (index, index) -> ()

    //      CHECK: %[[RET0_SHAPE:.+]] = flow.dispatch.shape %[[INNER_RET0]] : !flow.dispatch.tensor<writeonly:4x?xf32> -> !shapex.ranked_shape<[4,?]>
    %ret0_shape = flow.dispatch.shape %ret0 : !flow.dispatch.tensor<writeonly:4x?xf32> -> !shapex.ranked_shape<[4,?]>
    //  CHECK-DAG: %[[RET0_DIM0:.+]] = shapex.ranked_dim %[[RET0_SHAPE]][0] : !shapex.ranked_shape<[4,?]> -> index
    %ret0_dim0 = shapex.ranked_dim %ret0_shape[0] : !shapex.ranked_shape<[4,?]> -> index
    //  CHECK-DAG: %[[RET0_DIM1:.+]] = shapex.ranked_dim %[[RET0_SHAPE]][1] : !shapex.ranked_shape<[4,?]> -> index
    %ret0_dim1 = shapex.ranked_dim %ret0_shape[1] : !shapex.ranked_shape<[4,?]> -> index
    // CHECK-NEXT: "test.sink"(%[[RET0_DIM0]], %[[RET0_DIM1]])
    "test.sink"(%ret0_dim0, %ret0_dim1) : (index, index) -> ()

    // Load tensors (optional offsets/sizes/strides):

    // CHECK: %[[ARG0_VALUE:.+]] = flow.dispatch.tensor.load %[[INNER_ARG0]], {{.*}} : !flow.dispatch.tensor<readonly:?x4xf32> -> tensor<?x4xf32>
    %arg0_value = flow.dispatch.tensor.load %arg0_capture, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:?x4xf32> -> tensor<?x4xf32>

    // Operate on tensors with full IO shapes:

    // CHECK: %[[RET0_VALUE:.+]] = "test.math"(%[[ARG0_VALUE]], %[[ARG0_SHAPE]], %[[RET0_SHAPE]])
    %ret0_value = "test.math"(%arg0_value, %arg0_shape, %ret0_shape) : (tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>, !shapex.ranked_shape<[4,?]>) -> (tensor<4x?xf32>)

    // Store tensors (optional offsets/sizes/strides):

    // CHECK: flow.dispatch.tensor.store %[[RET0_VALUE]], %[[INNER_RET0]], {{.*}} : tensor<4x?xf32> -> !flow.dispatch.tensor<writeonly:4x?xf32>
    flow.dispatch.tensor.store %ret0_value, %ret0, offsets=[], sizes=[], strides=[] : tensor<4x?xf32> -> !flow.dispatch.tensor<writeonly:4x?xf32>

    // CHECK-NEXT: flow.return
    flow.return
  }
  // CHECK: return %[[OUTER_RET0]] : tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @inplaceDispatch
func @inplaceDispatch(
    // CHECK-SAME: %[[ARG0:.+]]: tensor<?x4xf32>
    %arg0: tensor<?x4xf32>,
    // CHECK-SAME: %[[ARG1:.+]]: index
    %arg1: index) -> tensor<?x4xf32> {
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[WORKGROUP_COUNT_X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[WORKGROUP_COUNT_Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[OUTER_RET0:.+]] = flow.dispatch.workgroups[
  // CHECK-SAME: %[[WORKGROUP_COUNT_X]], %[[WORKGROUP_COUNT_Y]]
  // CHECK-SAME: ](%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: : (tensor<?x4xf32>{%c128}, index) -> %arg0{%c128} =
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %arg1) : (tensor<?x4xf32>{%c128}, index) -> %arg0{%c128} =
  // CHECK-NEXT: (%[[INNER_ARG0:.+]]: !flow.dispatch.tensor<readwrite:?x4xf32>
  // CHECK-SAME:  %[[INNER_ARG1:.+]]: index) {
  (%arg0_capture: !flow.dispatch.tensor<readwrite:?x4xf32>, %arg1_capture: index) {
    // CHECK: %[[VALUE:.+]] = flow.dispatch.tensor.load %[[INNER_ARG0]], {{.*}} : !flow.dispatch.tensor<readwrite:?x4xf32> -> tensor<?x4xf32>
    %t = flow.dispatch.tensor.load %arg0_capture, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readwrite:?x4xf32> -> tensor<?x4xf32>
    // CHECK: flow.dispatch.tensor.store %[[VALUE]], %[[INNER_ARG0]], {{.*}}: tensor<?x4xf32> -> !flow.dispatch.tensor<readwrite:?x4xf32>
    flow.dispatch.tensor.store %t, %arg0_capture, offsets=[], sizes=[], strides=[] : tensor<?x4xf32> -> !flow.dispatch.tensor<readwrite:?x4xf32>
    // CHECK-NEXT: flow.return
    flow.return
  }
  // CHECK: return %[[OUTER_RET0]] : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}
