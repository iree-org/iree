// RUN: iree-opt --allow-unregistered-dialect --split-input-file --canonicalize %s | iree-opt --allow-unregistered-dialect --split-input-file | FileCheck %s

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
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> tensor<i32> =
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
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> tensor<i32> =
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
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> tensor<i32> =
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
  //      CHECK: flow.dispatch.workgroups[%c1, %c1, %c1]() : () -> (tensor<i32>, tensor<i32>) =
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
