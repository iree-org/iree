// RUN: iree-opt --allow-unregistered-dialect --split-input-file --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: util.func public @bubble_up_ordinal_ops(
util.func public @bubble_up_ordinal_ops(%arg0 : index, %arg1 : index) -> tensor<?x?xf32> {
  %result = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1) : (index, index) -> (tensor<?x?xf32>{%arg0, %arg1}) =
      (%b0 : index, %b1 : index, %b2 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    //      CHECK: flow.dispatch.workgroups
    // CHECK-NEXT:     %[[B0:[a-zA-Z0-9]+]]: index,
    // CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: index,
    // CHECK-SAME:     %[[B2:[a-zA-Z0-9]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>
    //  CHECK-DAG:   %[[WL0:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B0]], 0 : index
    //  CHECK-DAG:   %[[WL1:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B1]], 1 : index
    //      CHECK:   %[[BINDING:.+]] = flow.dispatch.tie_shape %[[B2]]
    // CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[WL0]], %[[WL1]]}
    //      CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[WL0]], %[[WL1]])
    //      CHECK:   iree_tensor_ext.dispatch.tensor.store %[[EMPTY]], %[[BINDING]]
    // CHECK-SAME:       sizes = [%[[WL0]], %[[WL1]]]
    // CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[WL0]], %[[WL1]]}
    %binding = flow.dispatch.tie_shape %b2 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%b0, %b1}
    %wl0 = iree_tensor_ext.dispatch.workload.ordinal %b0, 0 : index
    %wl1 = iree_tensor_ext.dispatch.workload.ordinal %b1, 1 : index
    %empty = tensor.empty(%wl0, %wl1) : tensor<?x?xf32>
    iree_tensor_ext.dispatch.tensor.store %empty, %binding, offsets = [0, 0], sizes = [%wl0, %wl1], strides = [1, 1]
        : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%wl0, %wl1}
    flow.return
  }
  util.return %result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: util.func public @dedup_workgroup_count_from_slice_operands(
util.func public @dedup_workgroup_count_from_slice_operands(
  %arg0 : index, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?x?xf32> {
  %result = flow.dispatch.workgroups [%arg0, %arg1, %arg2](%arg0, %arg1, %arg2)
      : (index, index, index) -> tensor<?x?x?x?x?xf32>{%arg0, %arg1, %arg2, %arg2, %arg0} =
      (%b0 : index, %b1 : index, %b2 : index, %b3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>) {
    //      CHECK: flow.dispatch.workgroups
    // CHECK-NEXT:   (%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index
    //  CHECK-DAG:   %[[WL0:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B0]], 0
    //  CHECK-DAG:   %[[WL1:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B1]], 1
    //  CHECK-DAG:   %[[WL2:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B2]], 2
    //      CHECK:   tensor.empty(%[[WL0]], %[[WL1]], %[[WL2]], %[[WL2]], %[[WL0]])
    %wl0 = iree_tensor_ext.dispatch.workload.ordinal %b0, 0 : index
    %wl1 = iree_tensor_ext.dispatch.workload.ordinal %b1, 1 : index
    %wl2 = iree_tensor_ext.dispatch.workload.ordinal %b2, 2 : index
    %wl3 = iree_tensor_ext.dispatch.workload.ordinal %b2, 3 : index
    %wl4 = iree_tensor_ext.dispatch.workload.ordinal %b0, 4 : index
    %out_binding = flow.dispatch.tie_shape %b3
        : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    %tensor = tensor.empty(%wl0, %wl1, %wl2, %wl3, %wl4) : tensor<?x?x?x?x?xf32>
    iree_tensor_ext.dispatch.tensor.store %tensor, %out_binding,
        offsets = [0, 0, 0, 0, 0], sizes = [%wl0, %wl1, %wl2, %wl3, %wl4], strides = [1, 1, 1, 1, 1]
        : tensor<?x?x?x?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    flow.return
  } count(%b0 : index, %b1 : index, %b2 : index) -> (index, index, index) {
    //     CHECK: count(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index)
    //     CHECK: iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[B0]], %[[B1]], %[[B2]])
    // CHECK-NOT: %[[B2]]
    // CHECK-NOT: %[[B0]]
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%b0, %b1, %b2, %b2, %b0)
    flow.return %x, %y, %z : index, index, index
  }
  util.return %result :tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: util.func public @dedup_workload(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index)
util.func public @dedup_workload(
  %arg0 : index, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?x?xf32> {
  %result = flow.dispatch.workgroups [%arg0, %arg1, %arg2, %arg2, %arg0](%arg0, %arg1, %arg2)
      : (index, index, index) -> tensor<?x?x?x?x?xf32>{%arg0, %arg1, %arg2, %arg2, %arg0} =
      (%b0 : index, %b1 : index, %b2 : index, %b3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>) {
    //      CHECK: flow.dispatch.workgroups[%[[ARG0]], %[[ARG1]], %[[ARG2]]]
    // CHECK-NEXT:   (%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index
    //  CHECK-DAG:   %[[WL0:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B0]], 0
    //  CHECK-DAG:   %[[WL1:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B1]], 1
    //  CHECK-DAG:   %[[WL2:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[B2]], 2
    //      CHECK:   tensor.empty(%[[WL0]], %[[WL1]], %[[WL2]], %[[WL2]], %[[WL0]])
    %wl0 = iree_tensor_ext.dispatch.workload.ordinal %b0, 0 : index
    %wl1 = iree_tensor_ext.dispatch.workload.ordinal %b1, 1 : index
    %wl2 = iree_tensor_ext.dispatch.workload.ordinal %b2, 2 : index
    %wl3 = iree_tensor_ext.dispatch.workload.ordinal %b2, 3 : index
    %wl4 = iree_tensor_ext.dispatch.workload.ordinal %b0, 4 : index
    %out_binding = flow.dispatch.tie_shape %b3
        : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    %tensor = tensor.empty(%wl0, %wl1, %wl2, %wl3, %wl4) : tensor<?x?x?x?x?xf32>
    iree_tensor_ext.dispatch.tensor.store %tensor, %out_binding,
        offsets = [0, 0, 0, 0, 0], sizes = [%wl0, %wl1, %wl2, %wl3, %wl4], strides = [1, 1, 1, 1, 1]
        : tensor<?x?x?x?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    flow.return
  } count(%b0 : index, %b1 : index, %b2 : index, %b3 : index, %b4 : index) -> (index, index, index) {
    //     CHECK: count(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index)
    //     CHECK: iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[B0]], %[[B1]], %[[B2]])
    // CHECK-NOT: %[[B2]]
    // CHECK-NOT: %[[B0]]
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%b0, %b1, %b2, %b3, %b4)
    flow.return %x, %y, %z : index, index, index
  }
  util.return %result :tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: util.func public @constant_fold_workload_ordinal()
util.func public @constant_fold_workload_ordinal() -> (index) {
  // CHECK: %[[C2:.+]] = arith.constant 2 : index
  %c2 = arith.constant 2: index
  // CHECK-NOT: iree_tensor_ext.dispatch.workload.ordinal
  %0 = iree_tensor_ext.dispatch.workload.ordinal %c2, 0 : index
  // CHECK: util.return %[[C2]]
  util.return %0 : index
}
