// RUN: iree-opt -allow-unregistered-dialect -split-input-file -canonicalize %s | iree-opt -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: @workgroupRankFolding
func @workgroupRankFolding(%arg0 : tensor<?x4xf32>) -> tensor<4x?xf32> {
  %c128 = arith.constant 128 : index
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch.workgroups
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<?x4xf32>{%c128}) -> (tensor<4x?xf32>{%c128}) = (
    %arg0_capture: !flow.dispatch.tensor<readonly:?x4xf32>,
    %ret0: !flow.dispatch.tensor<writeonly:4x?xf32>
  ) {
    // CHECK: %[[RANK:.+]] = arith.constant 2 : index
    %workgroup_rank = flow.dispatch.workgroup.rank : index
    // CHECK-NEXT: "test.sink"(%[[RANK]])
    "test.sink"(%workgroup_rank) : (index) -> ()
    flow.return
  }
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @inlineWithTiedResults1
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
func @inlineWithTiedResults1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK-NOT: constant 128
  %cst = arith.constant 128 : index
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  //      CHECK: flow.dispatch.workgroups[%[[X]], %[[Y]]](%[[ARG0]]) : (tensor<1x4xf32>) -> %[[ARG0]] =
  // CHECK-NEXT:   (%[[ARG0_INNER:.+]]: !flow.dispatch.tensor<readwrite:1x4xf32>)
  %0 = flow.dispatch.workgroups[%x, %y](%cst, %arg0) : (index, tensor<1x4xf32>) -> %arg0 = (
    %cst_capture: index,
    %arg0_capture: !flow.dispatch.tensor<readwrite:1x4xf32>
  ) {
    //      CHECK: %[[INLINED_CST:.+]] = arith.constant 128 : index
    // CHECK-NEXT: "test.sink"(%[[INLINED_CST]])
    "test.sink"(%cst_capture) : (index) -> ()
    // CHECK-NEXT: "test.sink"(%[[ARG0_INNER]])
    "test.sink"(%arg0_capture) : (!flow.dispatch.tensor<readwrite:1x4xf32>) -> ()
    flow.return
  }
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @inlineWithTiedResults2
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
func @inlineWithTiedResults2(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK-NOT: constant 128
  %cst = arith.constant 128 : index
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  //      CHECK: flow.dispatch.workgroups[%[[X]], %[[Y]]](%[[ARG0]]) : (tensor<1x4xf32>) -> %[[ARG0]] =
  // CHECK-NEXT:   (%[[ARG0_INNER:.+]]: !flow.dispatch.tensor<readwrite:1x4xf32>)
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %cst) : (tensor<1x4xf32>, index) -> %arg0 = (
    %arg0_capture: !flow.dispatch.tensor<readwrite:1x4xf32>,
    %cst_capture: index
  ) {
    //      CHECK: %[[INLINED_CST:.+]] = arith.constant 128 : index
    // CHECK-NEXT: "test.sink"(%[[INLINED_CST]])
    "test.sink"(%cst_capture) : (index) -> ()
    // CHECK-NEXT: "test.sink"(%[[ARG0_INNER]])
    "test.sink"(%arg0_capture) : (!flow.dispatch.tensor<readwrite:1x4xf32>) -> ()
    flow.return
  }
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dontInlineReadWrite
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
func @dontInlineReadWrite(%arg0: tensor<1x4xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[CST:.+]] = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
  %cst = arith.constant dense<0.0> : tensor<4x8xf32>
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  //      CHECK: flow.dispatch.workgroups[{{.+}}](%[[ARG0]], %[[CST]]) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<readonly:1x4xf32>, %{{.+}}: !flow.dispatch.tensor<readwrite:4x8xf32>)
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %cst) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst = (
    %arg0_capture: !flow.dispatch.tensor<readonly:1x4xf32>,
    %arg1_capture: !flow.dispatch.tensor<readwrite:4x8xf32>
  ) {
    "test.sink"(%arg0_capture) : (!flow.dispatch.tensor<readonly:1x4xf32>) -> ()
    %load = flow.dispatch.tensor.load %arg1_capture, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : !flow.dispatch.tensor<readwrite:4x8xf32> -> tensor<4x8xf32>
    %0 = "test.do_work"(%load) : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %0, %arg1_capture, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<readwrite:4x8xf32>
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @remove_unused_result
func @remove_unused_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:i32>)
  //      CHECK: flow.dispatch.tensor.store
  //  CHECK-NOT: flow.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !flow.dispatch.tensor<readonly:9xi32>, %arg1: !flow.dispatch.tensor<readonly:9xi32>, %arg2: !flow.dispatch.tensor<writeonly:i32>, %arg3: !flow.dispatch.tensor<writeonly:i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:9xi32> -> tensor<9xi32>
    %1 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:9xi32> -> tensor<9xi32>
    %2 = linalg.init_tensor [] : tensor<i32>
    %3 = linalg.fill(%c-2147483648_i32, %2) : i32, tensor<i32> -> tensor<i32>
    %4 = linalg.fill(%c0_i32, %2) : i32, tensor<i32> -> tensor<i32>
    flow.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:i32>
    flow.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:i32>
    flow.return
  }
  return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @remove_unused_dynamic_result
func @remove_unused_dynamic_result(%dim: index) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:i32>)
  //  CHECK-NOT: flow.dispatch.tie_shape
  //      CHECK: flow.dispatch.tensor.store
  //  CHECK-NOT: flow.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%dim) : (index) -> (tensor<i32>, tensor<?xi32>{%dim}) =
      (%dim: index, %ret0: !flow.dispatch.tensor<writeonly:i32>, %ret1: !flow.dispatch.tensor<writeonly:?xi32>) {
    // Used as a result; should remain after canonicalization.
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %ret0_init = linalg.init_tensor [] : tensor<i32>
    %ret0_value = linalg.fill(%c-2147483648_i32, %ret0_init) : i32, tensor<i32> -> tensor<i32>
    flow.dispatch.tensor.store %ret0_value, %ret0, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:i32>

    // Unused as a result; should be stripped entirely.
    %c0_i32 = arith.constant 0 : i32
    %ret1_shaped = flow.dispatch.tie_shape %ret1 : !flow.dispatch.tensor<writeonly:?xi32>{%dim}
    %ret1_init = linalg.init_tensor [%dim] : tensor<?xi32>
    %ret1_value = linalg.fill(%c0_i32, %ret1_init) : i32, tensor<?xi32> -> tensor<?xi32>
    flow.dispatch.tensor.store %ret1_value, %ret1_shaped, offsets = [0], sizes = [%dim], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:?xi32>{%dim}
    flow.return
  }
  return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @remove_unused_read_write_result
func @remove_unused_read_write_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:i32>)
  //      CHECK: flow.dispatch.tensor.store %{{.+}},
  //  CHECK-NOT: flow.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !flow.dispatch.tensor<readonly:9xi32>, %arg1: !flow.dispatch.tensor<readonly:9xi32>, %arg2: !flow.dispatch.tensor<writeonly:i32>, %arg3: !flow.dispatch.tensor<readwrite:i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:9xi32> -> tensor<9xi32>
    %1 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:9xi32> -> tensor<9xi32>
    %2 = linalg.init_tensor [] : tensor<i32>
    %3 = linalg.fill(%c-2147483648_i32, %2) : i32, tensor<i32> -> tensor<i32>
    %4 = linalg.fill(%c0_i32, %2) : i32, tensor<i32> -> tensor<i32>
    flow.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:i32>
    flow.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<readwrite:i32>
    flow.return
  }
  return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @keep_used_read_write_result
func @keep_used_read_write_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> (tensor<i32>, tensor<i32>) =
  // CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<writeonly:i32>, %{{.+}}: !flow.dispatch.tensor<readwrite:i32>)
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !flow.dispatch.tensor<readonly:9xi32>, %arg1: !flow.dispatch.tensor<readonly:9xi32>, %arg2: !flow.dispatch.tensor<writeonly:i32>, %arg3: !flow.dispatch.tensor<readwrite:i32>) {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = flow.dispatch.tensor.load %arg3, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:i32> -> tensor<i32>
    %val = tensor.extract %0[] : tensor<i32>
    %1 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !flow.dispatch.tensor<readonly:9xi32> -> tensor<9xi32>
    %2 = linalg.init_tensor [] : tensor<i32>
    %3 = linalg.fill(%c-2147483648_i32, %2) : i32, tensor<i32> -> tensor<i32>
    %4 = linalg.fill(%val, %2) : i32, tensor<i32> -> tensor<i32>
    flow.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:i32>
    flow.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<readwrite:i32>
    flow.return
  }
  return %0#0 : tensor<i32>
}
