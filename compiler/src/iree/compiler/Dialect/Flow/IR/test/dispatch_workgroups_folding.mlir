// RUN: iree-opt --allow-unregistered-dialect --split-input-file --canonicalize --cse %s | iree-opt --allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @dontInlineReadWrite
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
func.func @dontInlineReadWrite(%arg0: tensor<1x4xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[CST:.+]] = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
  %cst = arith.constant dense<0.0> : tensor<4x8xf32>
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  //      CHECK: flow.dispatch.workgroups[{{.+}}](%[[ARG0]], %[[CST]]) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<readonly:tensor<1x4xf32>>, %{{.+}}: !flow.dispatch.tensor<readwrite:tensor<4x8xf32>>)
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %cst) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst = (
    %arg0_capture: !flow.dispatch.tensor<readonly:tensor<1x4xf32>>,
    %arg1_capture: !flow.dispatch.tensor<readwrite:tensor<4x8xf32>>
  ) {
    "test.sink"(%arg0_capture) : (!flow.dispatch.tensor<readonly:tensor<1x4xf32>>) -> ()
    %load = flow.dispatch.tensor.load %arg1_capture, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : !flow.dispatch.tensor<readwrite:tensor<4x8xf32>> -> tensor<4x8xf32>
    %0 = "test.do_work"(%load) : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %0, %arg1_capture, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<readwrite:tensor<4x8xf32>>
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @remove_unused_result
func.func @remove_unused_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:tensor<i32>>)
  //      CHECK: flow.dispatch.tensor.store
  //  CHECK-NOT: flow.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !flow.dispatch.tensor<readonly:tensor<9xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<9xi32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<i32>>, %arg3: !flow.dispatch.tensor<writeonly:tensor<i32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %1 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    flow.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:tensor<i32>>
    flow.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:tensor<i32>>
    flow.return
  }
  return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @remove_unused_dynamic_result
func.func @remove_unused_dynamic_result(%dim: index) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:tensor<i32>>)
  //  CHECK-NOT: flow.dispatch.tie_shape
  //      CHECK: flow.dispatch.tensor.store
  //  CHECK-NOT: flow.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%dim) : (index) -> (tensor<i32>, tensor<?xi32>{%dim}) =
      (%dim: index, %ret0: !flow.dispatch.tensor<writeonly:tensor<i32>>, %ret1: !flow.dispatch.tensor<writeonly:tensor<?xi32>>) {
    // Used as a result; should remain after canonicalization.
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %ret0_init = tensor.empty() : tensor<i32>
    %ret0_value = linalg.fill ins(%c-2147483648_i32 : i32) outs(%ret0_init : tensor<i32>) -> tensor<i32>
    flow.dispatch.tensor.store %ret0_value, %ret0, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:tensor<i32>>

    // Unused as a result; should be stripped entirely.
    %c0_i32 = arith.constant 0 : i32
    %ret1_shaped = flow.dispatch.tie_shape %ret1 : !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%dim}
    %ret1_init = tensor.empty(%dim) : tensor<?xi32>
    %ret1_value = linalg.fill ins(%c0_i32 : i32) outs(%ret1_init : tensor<?xi32>) -> tensor<?xi32>
    flow.dispatch.tensor.store %ret1_value, %ret1_shaped, offsets = [0], sizes = [%dim], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%dim}
    flow.return
  }
  return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @remove_unused_read_write_result
func.func @remove_unused_read_write_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:tensor<i32>>)
  //      CHECK: flow.dispatch.tensor.store %{{.+}},
  //  CHECK-NOT: flow.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !flow.dispatch.tensor<readonly:tensor<9xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<9xi32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<i32>>, %arg3: !flow.dispatch.tensor<readwrite:tensor<i32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %1 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    flow.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:tensor<i32>>
    flow.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<readwrite:tensor<i32>>
    flow.return
  }
  return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @keep_used_read_write_result
func.func @keep_used_read_write_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> (tensor<i32>, tensor<i32>) =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:tensor<i32>>, %{{.+}}: !flow.dispatch.tensor<readwrite:tensor<i32>>)
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !flow.dispatch.tensor<readonly:tensor<9xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<9xi32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<i32>>, %arg3: !flow.dispatch.tensor<readwrite:tensor<i32>>) {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = flow.dispatch.tensor.load %arg3, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:tensor<i32>> -> tensor<i32>
    %val = tensor.extract %0[] : tensor<i32>
    %1 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %4 = linalg.fill ins(%val : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    flow.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:tensor<i32>>
    flow.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<readwrite:tensor<i32>>
    flow.return
  }
  return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @drop_unused_dispatch_region_result
func.func @drop_unused_dispatch_region_result(
    %arg0: tensor<?x?xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<7x11xf32>)
  -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  // CHECK: %[[r:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%{{.*}}, %{{.*}}}) {
  // CHECK:   %[[slice:.*]] = tensor.insert_slice
  // CHECK:   flow.return %[[slice]] : tensor<?x?xf32>
  // CHECK: }
  %r:2 = flow.dispatch.region -> (tensor<?x?xf32>{%d0, %d1}, tensor<?x?xf32>{%d0, %d1}) {
    %0 = tensor.insert_slice %arg1 into %arg0[6, 7][5, 10][1, 1] : tensor<5x10xf32> into tensor<?x?xf32>
    %1 = tensor.insert_slice %arg2 into %0[9, 10][7, 11][1, 1] : tensor<7x11xf32> into tensor<?x?xf32>
    flow.return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
  }
  // CHECK: return %[[r]]
  return %r#0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @bubble_up_ordinal_ops(
func.func @bubble_up_ordinal_ops(%arg0 : index, %arg1 : index) -> tensor<?x?xf32> {
  %result = flow.dispatch.workgroups[%arg0, %arg1](%arg0, %arg1) : (index, index) -> (tensor<?x?xf32>{%arg0, %arg1}) =
      (%b0 : index, %b1 : index, %b2 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    //      CHECK: flow.dispatch.workgroups
    // CHECK-NEXT:     %[[B0:[a-zA-Z0-9]+]]: index,
    // CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: index,
    // CHECK-SAME:     %[[B2:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
    //  CHECK-DAG:   %[[WL0:.+]] = flow.dispatch.workload.ordinal %[[B0]], 0 : index
    //  CHECK-DAG:   %[[WL1:.+]] = flow.dispatch.workload.ordinal %[[B1]], 1 : index
    //      CHECK:   %[[BINDING:.+]] = flow.dispatch.tie_shape %[[B2]]
    // CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[WL0]], %[[WL1]]}
    //      CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[WL0]], %[[WL1]])
    //      CHECK:   flow.dispatch.tensor.store %[[EMPTY]], %[[BINDING]]
    // CHECK-SAME:       sizes = [%[[WL0]], %[[WL1]]]
    // CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[WL0]], %[[WL1]]}
    %binding = flow.dispatch.tie_shape %b2 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%b0, %b1}
    %wl0 = flow.dispatch.workload.ordinal %b0, 0 : index
    %wl1 = flow.dispatch.workload.ordinal %b1, 1 : index
    %empty = tensor.empty(%wl0, %wl1) : tensor<?x?xf32>
    flow.dispatch.tensor.store %empty, %binding, offsets = [0, 0], sizes = [%wl0, %wl1], strides = [1, 1]
        : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%wl0, %wl1}
    flow.return
  }
  return %result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dedup_workgroup_count_from_slice_operands(
func.func @dedup_workgroup_count_from_slice_operands(
  %arg0 : index, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?x?xf32> {
  %result = flow.dispatch.workgroups [%arg0, %arg1, %arg2](%arg0, %arg1, %arg2)
      : (index, index, index) -> tensor<?x?x?x?x?xf32>{%arg0, %arg1, %arg2, %arg2, %arg0} =
      (%b0 : index, %b1 : index, %b2 : index, %b3 : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>) {
    //      CHECK: flow.dispatch.workgroups
    // CHECK-NEXT:   (%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index
    //  CHECK-DAG:   %[[WL0:.+]] = flow.dispatch.workload.ordinal %[[B0]], 0
    //  CHECK-DAG:   %[[WL1:.+]] = flow.dispatch.workload.ordinal %[[B1]], 1
    //  CHECK-DAG:   %[[WL2:.+]] = flow.dispatch.workload.ordinal %[[B2]], 2
    //      CHECK:   tensor.empty(%[[WL0]], %[[WL1]], %[[WL2]], %[[WL2]], %[[WL0]])
    %wl0 = flow.dispatch.workload.ordinal %b0, 0 : index
    %wl1 = flow.dispatch.workload.ordinal %b1, 1 : index
    %wl2 = flow.dispatch.workload.ordinal %b2, 2 : index
    %wl3 = flow.dispatch.workload.ordinal %b2, 3 : index
    %wl4 = flow.dispatch.workload.ordinal %b0, 4 : index
    %out_binding = flow.dispatch.tie_shape %b3
        : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    %tensor = tensor.empty(%wl0, %wl1, %wl2, %wl3, %wl4) : tensor<?x?x?x?x?xf32>
    flow.dispatch.tensor.store %tensor, %out_binding,
        offsets = [0, 0, 0, 0, 0], sizes = [%wl0, %wl1, %wl2, %wl3, %wl4], strides = [1, 1, 1, 1, 1]
        : tensor<?x?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    flow.return
  } count(%b0 : index, %b1 : index, %b2 : index) -> (index, index, index) {
    //     CHECK: count(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index)
    //     CHECK: flow.dispatch.workgroup_count_from_slice %[[B0]], %[[B1]], %[[B2]]
    // CHECK-NOT: %[[B2]]
    // CHECK-NOT: %[[B0]]
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice %b0, %b1, %b2, %b2, %b0
    flow.return %x, %y, %z : index, index, index
  }
  return %result :tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @dedup_workload(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index)
func.func @dedup_workload(
  %arg0 : index, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?x?xf32> {
  %result = flow.dispatch.workgroups [%arg0, %arg1, %arg2, %arg2, %arg0](%arg0, %arg1, %arg2)
      : (index, index, index) -> tensor<?x?x?x?x?xf32>{%arg0, %arg1, %arg2, %arg2, %arg0} =
      (%b0 : index, %b1 : index, %b2 : index, %b3 : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>) {
    //      CHECK: flow.dispatch.workgroups[%[[ARG0]], %[[ARG1]], %[[ARG2]]]
    // CHECK-NEXT:   (%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index
    //  CHECK-DAG:   %[[WL0:.+]] = flow.dispatch.workload.ordinal %[[B0]], 0
    //  CHECK-DAG:   %[[WL1:.+]] = flow.dispatch.workload.ordinal %[[B1]], 1
    //  CHECK-DAG:   %[[WL2:.+]] = flow.dispatch.workload.ordinal %[[B2]], 2
    //      CHECK:   tensor.empty(%[[WL0]], %[[WL1]], %[[WL2]], %[[WL2]], %[[WL0]])
    %wl0 = flow.dispatch.workload.ordinal %b0, 0 : index
    %wl1 = flow.dispatch.workload.ordinal %b1, 1 : index
    %wl2 = flow.dispatch.workload.ordinal %b2, 2 : index
    %wl3 = flow.dispatch.workload.ordinal %b2, 3 : index
    %wl4 = flow.dispatch.workload.ordinal %b0, 4 : index
    %out_binding = flow.dispatch.tie_shape %b3
        : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    %tensor = tensor.empty(%wl0, %wl1, %wl2, %wl3, %wl4) : tensor<?x?x?x?x?xf32>
    flow.dispatch.tensor.store %tensor, %out_binding,
        offsets = [0, 0, 0, 0, 0], sizes = [%wl0, %wl1, %wl2, %wl3, %wl4], strides = [1, 1, 1, 1, 1]
        : tensor<?x?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?x?xf32>>{%wl0, %wl1, %wl2, %wl3, %wl4}
    flow.return
  } count(%b0 : index, %b1 : index, %b2 : index, %b3 : index, %b4 : index) -> (index, index, index) {
    //     CHECK: count(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index)
    //     CHECK: flow.dispatch.workgroup_count_from_slice %[[B0]], %[[B1]], %[[B2]]
    // CHECK-NOT: %[[B2]]
    // CHECK-NOT: %[[B0]]
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice %b0, %b1, %b2, %b3, %b4
    flow.return %x, %y, %z : index, index, index
  }
  return %result :tensor<?x?x?x?x?xf32>
}
