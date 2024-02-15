// RUN: iree-opt --allow-unregistered-dialect --split-input-file %s | iree-opt --allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: @complexWorkgroupsUsage
util.func public @complexWorkgroupsUsage(
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
  // CHECK-SAME: ](%[[ARG0]], %[[ARG1]], %c128, %c128)
  // CHECK-SAME: : (tensor<?x4xf32>{%c128}, index, index, index) -> tensor<4x?xf32>{%c128} =
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %arg1, %c128, %c128) : (tensor<?x4xf32>{%c128}, index, index, index) -> tensor<4x?xf32>{%c128} =
  // CHECK-NEXT: (%[[INNER_ARG0:.+]]: !flow.dispatch.tensor<readonly:tensor<?x4xf32>>
  // CHECK-SAME:  %[[INNER_ARG1:.+]]: index, %[[INNER_ARG0_DIM0:.+]]: index, %[[INNER_RET0_DIM1:.+]]: index,
  // CHECK-SAME:  %[[INNER_RET0:.+]]: !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>) {
  (%arg0_capture: !flow.dispatch.tensor<readonly:tensor<?x4xf32>>,
   %arg1_capture: index, %arg0_dim0: index, %ret0_dim1: index,
   %ret0: !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>) {

    // Query symbolic workgroup info:

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

    // Load tensors (optional offsets/sizes/strides):

    // CHECK: %[[ARG0_VALUE:.+]] = flow.dispatch.tensor.load %[[INNER_ARG0]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x4xf32>>{%[[INNER_ARG0_DIM0]]} -> tensor<?x4xf32>
    %arg0_value = flow.dispatch.tensor.load %arg0_capture, offsets=[0, 0], sizes=[%arg0_dim0, 4], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<?x4xf32>>{%arg0_dim0} -> tensor<?x4xf32>

    // Operate on tensors:

    // CHECK: %[[RET0_VALUE:.+]] = "test.math"(%[[ARG0_VALUE]])
    %ret0_value = "test.math"(%arg0_value) : (tensor<?x4xf32>) -> (tensor<4x?xf32>)

    // Store tensors (optional offsets/sizes/strides):

    // CHECK: flow.dispatch.tensor.store %[[RET0_VALUE]], %[[INNER_RET0]], {{.*}} : tensor<4x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>{%[[INNER_RET0_DIM1]]}
    flow.dispatch.tensor.store %ret0_value, %ret0, offsets=[0, 0], sizes=[4, %ret0_dim1], strides=[1, 1] : tensor<4x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>{%ret0_dim1}

    // CHECK-NEXT: flow.return
    flow.return
  }
  // CHECK: util.return %[[OUTER_RET0]] : tensor<4x?xf32>
  util.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @inplaceDispatch
util.func public @inplaceDispatch(
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
  // CHECK-NEXT: (%[[INNER_ARG0:.+]]: !flow.dispatch.tensor<readwrite:tensor<?x4xf32>>
  // CHECK-SAME:  %[[INNER_ARG1:.+]]: index) {
  (%arg0_capture: !flow.dispatch.tensor<readwrite:tensor<?x4xf32>>, %arg1_capture: index) {
    // CHECK: %[[VALUE:.+]] = flow.dispatch.tensor.load %[[INNER_ARG0]], {{.*}} : !flow.dispatch.tensor<readwrite:tensor<?x4xf32>>{%[[INNER_ARG1]]} -> tensor<?x4xf32>
    %t = flow.dispatch.tensor.load %arg0_capture, offsets=[0, 0], sizes=[%arg1_capture, 4], strides=[1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x4xf32>>{%arg1_capture} -> tensor<?x4xf32>
    // CHECK: flow.dispatch.tensor.store %[[VALUE]], %[[INNER_ARG0]], {{.*}}: tensor<?x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x4xf32>>{%[[INNER_ARG1]]}
    flow.dispatch.tensor.store %t, %arg0_capture, offsets=[0, 0], sizes=[%arg1_capture, 4], strides=[1, 1] : tensor<?x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x4xf32>>{%arg1_capture}
    // CHECK-NEXT: flow.return
    flow.return
  }
  // CHECK: util.return %[[OUTER_RET0]] : tensor<?x4xf32>
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @dispatchWithCountRegion
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xi32>)
util.func public @dispatchWithCountRegion(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-DAG: %[[WORKGROUP_COUNT_X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[WORKGROUP_COUNT_Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[OUTER_RET0:.+]] = flow.dispatch.workgroups[
  // CHECK-SAME: %[[WORKGROUP_COUNT_X]], %[[WORKGROUP_COUNT_Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<4xi32>) -> %[[ARG0]] =
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<4xi32>) -> %arg0 =
  // CHECK-NEXT: (%{{.+}}: !flow.dispatch.tensor<readwrite:tensor<4xi32>>) {
  (%arg0_capture: !flow.dispatch.tensor<readwrite:tensor<4xi32>>) {
    // CHECK-NEXT: flow.return
    flow.return
  // CHECK-NEXT: count(%[[X_CAPTURE:.+]]: index, %[[Y_CAPTURE:.+]]: index)
  // CHECK-SAME:   -> (index, index, index)
  } count(%x_capture: index, %y_capture: index) -> (index, index, index) {
    // CHECK-NEXT: %[[Z:.+]] = arith.constant 1
    %z = arith.constant 1 : index
    // CHECK-NEXT: flow.return %[[X_CAPTURE]], %[[Y_CAPTURE]], %[[Z]]
    flow.return %x_capture, %y_capture, %z : index, index, index
  }
  // CHECK: util.return %[[OUTER_RET0]] : tensor<4xi32>
  util.return %0 : tensor<4xi32>
}
