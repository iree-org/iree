// RUN: iree-opt --split-input-file --allow-unregistered-dialect %s --verify-diagnostics | FileCheck %s

flow.executable @ex0 {
  flow.executable.export @dispatch_fn
  builtin.module {
    util.func public @dispatch_fn(%cst : index, %arg0 : tensor<4xf32>) -> tensor<4xf32> {
      util.return %arg0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: @dispatch
util.func public @dispatch(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CST:.+]] = arith.constant
  %cst = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

flow.executable private @ex0 {
  flow.executable.export public @dispatch_a
  flow.executable.export public @dispatch_b
}

// CHECK-LABEL: @dispatchWithMultipleRefs
util.func public @dispatchWithMultipleRefs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: = flow.dispatch {@ex0::@dispatch_a, @ex0::@dispatch_b}(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch {@ex0::@dispatch_a, @ex0::@dispatch_b}(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}


// -----

flow.executable private @ex0 {
  flow.executable.export public @dispatch workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    flow.return %arg0, %arg1, %arg0 : index, index, index
  }
}

// CHECK-LABEL: @dispatchWithWorkgroupCount
util.func public @dispatchWithWorkgroupCount(%arg0: tensor<4xf32>, %arg1: index) -> tensor<4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: = flow.dispatch @ex0::@dispatch[%c1, %c2](%arg0, %arg1) : (tensor<4xf32>, index) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch[%c1, %c2](%arg0, %arg1) : (tensor<4xf32>, index) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

flow.executable private @ex0 {
  flow.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    flow.return %arg0, %arg0, %arg0 : index, index, index
  }
}

util.func public @dispatchWithInvalidWorkload(%arg0: tensor<4xf32>, %arg1: index) -> tensor<4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // expected-error @+1 {{op workload mismatch; entry point expects 1 arguments but dispatch provides 2}}
  %0 = flow.dispatch @ex0::@dispatch[%c1, %c2](%arg0, %arg1) : (tensor<4xf32>, index) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @dispatchNoWorkload
util.func public @dispatchNoWorkload(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CST:.+]] = arith.constant
  %cst = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn(%[[CST]], %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch_fn(%cst, %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @inplaceDispatch
util.func public @inplaceDispatch(%arg0 : tensor<4xf32>, %arg1 : tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
  // CHECK: %[[CST:.+]] = arith.constant
  %cst = arith.constant 4 : index
  // CHECK: %0:2 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0, %arg1) : (index, tensor<4xf32>, tensor<8xf32>) -> (%arg0, %arg1)
  %0, %1 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0, %arg1) : (index, tensor<4xf32>, tensor<8xf32>) -> (%arg0, %arg1)
  util.return %0, %1 : tensor<4xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: @inplaceDynamicDispatch
util.func public @inplaceDynamicDispatch(%arg0 : tensor<4x?xf32>, %arg1 : tensor<8x?xf32>) -> (tensor<4x?xf32>, tensor<8x?xf32>) {
  // CHECK-DAG: %[[CST:.+]] = arith.constant 4
  %cst = arith.constant 4 : index
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 100
  %dim0 = arith.constant 100 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 200
  %dim1 = arith.constant 200 : index
  // CHECK: %0:2 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0, %arg1) : (index, tensor<4x?xf32>{%[[DIM0]]}, tensor<8x?xf32>{%[[DIM1]]}) -> (%arg0{%[[DIM1]]}, %arg1{%[[DIM0]]})
  %0, %1 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0, %arg1) : (index, tensor<4x?xf32>{%dim0}, tensor<8x?xf32>{%dim1}) -> (%arg0{%dim1}, %arg1{%dim0})
  util.return %0, %1 : tensor<4x?xf32>, tensor<8x?xf32>
}

// -----

// CHECK-LABEL: @inplaceTypeChange
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?xf32>)
util.func public @inplaceTypeChange(%arg0: tensor<4x?xf32>) -> tensor<?x4xf32> {
  // CHECK-DAG: %[[CST:.+]] = arith.constant 4
  %cst = arith.constant 4 : index
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 100
  %dim0 = arith.constant 100 : index
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[ARG0]]) : (tensor<4x?xf32>{%[[DIM0]]}) -> %arg0 as tensor<?x4xf32>{%[[DIM0]]}
  %0 = flow.dispatch @ex0::@dispatch_fn[%cst](%arg0) : (tensor<4x?xf32>{%dim0}) -> %arg0 as tensor<?x4xf32>{%dim0}
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @region
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>)
util.func public @region(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[R:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%{{.*}}, %{{.*}}}) {
  // CHECK:   flow.return %[[ARG0]] : tensor<?x?xf32>
  // CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = flow.dispatch.region -> (tensor<?x?xf32>{%d0, %d1}) {
    flow.return %arg0 : tensor<?x?xf32>
  }
  // CHECK: util.return %[[R]]
  util.return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @regionStaticShape
// CHECK-SAME: (%[[ARG0:.+]]: tensor<5x10xf32>)
util.func public @regionStaticShape(%arg0: tensor<5x10xf32>) -> tensor<5x10xf32> {
  // CHECK: %[[R:.*]] = flow.dispatch.region -> (tensor<5x10xf32>) {
  // CHECK:   flow.return %[[ARG0]] : tensor<5x10xf32>
  // CHECK: }
  %r = flow.dispatch.region -> (tensor<5x10xf32>) {
    flow.return %arg0 : tensor<5x10xf32>
  }
  // CHECK: util.return %[[R]]
  util.return %r : tensor<5x10xf32>
}

// -----

// CHECK-LABEL: util.func public @regionDynamicShape
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?x16xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index, %[[DIM3:.+]]: index)
util.func public @regionDynamicShape(%arg0: tensor<?x?x16xf32>, %dim0: index, %dim1: index, %dim2: index, %dim3: index) -> tensor<?x?x16xf32> {
  // CHECK: %[[C16:.+]] = arith.constant 16 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[R:.+]] = flow.dispatch.region[%[[DIM0]], %[[DIM1]], %[[C16]]] -> (tensor<?x?x16xf32>{%[[DIM2]], %[[DIM3]]}) {
  // CHECK:   flow.return %[[ARG0]] : tensor<?x?x16xf32>
  // CHECK: }
  %region = flow.dispatch.region[%dim0, %dim1, %c16] -> (tensor<?x?x16xf32>{%dim2, %dim3}) {
    flow.return %arg0 : tensor<?x?x16xf32>
  }
  // CHECK: util.return %[[R]]
  util.return %region: tensor<?x?x16xf32>
}

// -----

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
  // CHECK-NEXT: (%[[INNER_ARG0:.+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4xf32>>
  // CHECK-SAME:  %[[INNER_ARG1:.+]]: index, %[[INNER_ARG0_DIM0:.+]]: index, %[[INNER_RET0_DIM1:.+]]: index,
  // CHECK-SAME:  %[[INNER_RET0:.+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32>>) {
  (%arg0_capture: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4xf32>>,
   %arg1_capture: index, %arg0_dim0: index, %ret0_dim1: index,
   %ret0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32>>) {

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

    // CHECK: %[[ARG0_VALUE:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INNER_ARG0]], {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4xf32>>{%[[INNER_ARG0_DIM0]]} -> tensor<?x4xf32>
    %arg0_value = iree_tensor_ext.dispatch.tensor.load %arg0_capture, offsets=[0, 0], sizes=[%arg0_dim0, 4], strides=[1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4xf32>>{%arg0_dim0} -> tensor<?x4xf32>

    // Operate on tensors:

    // CHECK: %[[RET0_VALUE:.+]] = "test.math"(%[[ARG0_VALUE]])
    %ret0_value = "test.math"(%arg0_value) : (tensor<?x4xf32>) -> (tensor<4x?xf32>)

    // Store tensors (optional offsets/sizes/strides):

    // CHECK: iree_tensor_ext.dispatch.tensor.store %[[RET0_VALUE]], %[[INNER_RET0]], {{.*}} : tensor<4x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32>>{%[[INNER_RET0_DIM1]]}
    iree_tensor_ext.dispatch.tensor.store %ret0_value, %ret0, offsets=[0, 0], sizes=[4, %ret0_dim1], strides=[1, 1] : tensor<4x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32>>{%ret0_dim1}

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
  // CHECK-NEXT: (%[[INNER_ARG0:.+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4xf32>>
  // CHECK-SAME:  %[[INNER_ARG1:.+]]: index) {
  (%arg0_capture: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4xf32>>, %arg1_capture: index) {
    // CHECK: %[[VALUE:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INNER_ARG0]], {{.*}} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4xf32>>{%[[INNER_ARG1]]} -> tensor<?x4xf32>
    %t = iree_tensor_ext.dispatch.tensor.load %arg0_capture, offsets=[0, 0], sizes=[%arg1_capture, 4], strides=[1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4xf32>>{%arg1_capture} -> tensor<?x4xf32>
    // CHECK: iree_tensor_ext.dispatch.tensor.store %[[VALUE]], %[[INNER_ARG0]], {{.*}}: tensor<?x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4xf32>>{%[[INNER_ARG1]]}
    iree_tensor_ext.dispatch.tensor.store %t, %arg0_capture, offsets=[0, 0], sizes=[%arg1_capture, 4], strides=[1, 1] : tensor<?x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4xf32>>{%arg1_capture}
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
  // CHECK-NEXT: (%{{.+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>>) {
  (%arg0_capture: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>>) {
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
