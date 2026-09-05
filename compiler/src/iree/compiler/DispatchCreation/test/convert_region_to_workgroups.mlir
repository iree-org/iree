// RUN: iree-opt %s --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups, iree-flow-canonicalize, cse))" -split-input-file | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: util.func public @foo(
//       CHECK:   %[[argA:.*]]: tensor<?x?xf32>, %[[argB:.*]]: tensor<5x10xf32>, %[[argC:.*]]: tensor<10x11xf32>
util.func public @foo(%argA: tensor<?x?xf32>, %argB: tensor<5x10xf32>, %argC: tensor<10x11xf32>) -> (tensor<?x?xf32>, tensor<5x11xf32>) {
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  //  CHECK-DAG: %[[dim_argA_0:.*]] = tensor.dim %[[argA]], %[[c0]]
  //  CHECK-DAG: %[[dim_argA_1:.*]] = tensor.dim %[[argA]], %[[c1]]
  //      CHECK: %[[r0:.*]] = flow.dispatch.workgroups(%[[argA]], %[[dim_argA_0]], %[[dim_argA_1]]) : (tensor<?x?xf32>{%[[dim_argA_0]], %[[dim_argA_1]]}, index, index) -> %[[argA]]{%[[dim_argA_0]], %[[dim_argA_1]]} =
  // CHECK-NEXT: (%[[arg1:.*]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>, %[[arg2:.*]]: index, %[[arg3:.*]]: index) {
  //      CHECK:   %[[load:.*]] = iree_tensor_ext.dispatch.tensor.load %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[arg2]], %[[arg3]]} -> tensor<?x?xf32>
  //      CHECK:   iree_tensor_ext.dispatch.tensor.store %[[load]], %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[arg2]], %[[arg3]]}
  //      CHECK:   flow.return
  //      CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dimA0 = tensor.dim %argA, %c0 : tensor<?x?xf32>
  %dimA1 = tensor.dim %argA, %c1 : tensor<?x?xf32>
  %r0 = flow.dispatch.region -> (tensor<?x?xf32>{%dimA0, %dimA1}) {
    flow.return %argA : tensor<?x?xf32>
  }
  //      CHECK: %[[r1:.*]] = flow.dispatch.workgroups(%[[argB]], %[[argC]]) : (tensor<5x10xf32>, tensor<10x11xf32>) -> tensor<5x11xf32>
  // CHECK-SAME:   stream.affinity = #hal.device.affinity<@device>
  // CHECK-NEXT: (%[[arg3:.*]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x10xf32>>, %[[arg4:.*]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x11xf32>>, %[[arg5:.*]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x11xf32>>)
  //  CHECK-DAG:   %[[loadB:.*]] = iree_tensor_ext.dispatch.tensor.load %[[arg3]], offsets = [0, 0], sizes = [5, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x10xf32>> -> tensor<5x10xf32>
  //  CHECK-DAG:   %[[loadC:.*]] = iree_tensor_ext.dispatch.tensor.load %[[arg4]], offsets = [0, 0], sizes = [10, 11], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x11xf32>> -> tensor<10x11xf32>
  //      CHECK:   %[[empty:.*]] = tensor.empty() : tensor<5x11xf32>
  //      CHECK:   %[[fill:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[empty]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   %[[matmul:.*]] = linalg.matmul ins(%[[loadB]], %[[loadC]] : tensor<5x10xf32>, tensor<10x11xf32>) outs(%[[fill]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   iree_tensor_ext.dispatch.tensor.store %[[matmul]], %[[arg5]], offsets = [0, 0], sizes = [5, 11], strides = [1, 1] : tensor<5x11xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x11xf32>>
  //      CHECK:   flow.return
  //      CHECK: }
  %r1 = flow.dispatch.region -> (tensor<5x11xf32>) attributes {
    stream.affinity = #hal.device.affinity<@device>
  } {
    %zero = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<5x11xf32>
    %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<5x11xf32>) -> tensor<5x11xf32>
    %2 = linalg.matmul ins(%argB, %argC : tensor<5x10xf32>, tensor<10x11xf32>)
        outs(%1 : tensor<5x11xf32>) -> tensor<5x11xf32>
    flow.return %2 : tensor<5x11xf32>
  }

  //      CHECK: util.return %[[r0]], %[[r1]]
  util.return %r0, %r1 : tensor<?x?xf32>, tensor<5x11xf32>
}

// -----

// CHECK-LABEL: util.func public @sort_with_internal_key_use
util.func public @sort_with_internal_key_use(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: %[[RESULT:[a-zA-Z0-9_]+]]:2 = flow.dispatch.workgroups
  // CHECK-NEXT: (%[[KEYS:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>,
  // CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>)
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    // CHECK: %[[SORTED:[a-zA-Z0-9_]+]]:2 = iree_linalg_ext.sort
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %combined = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%sorted#0 : tensor<4xi64>)
        outs(%sorted#1 : tensor<4xi64>) {
    ^bb0(%key: i64, %index: i64):
      %sum = arith.addi %key, %index : i64
      linalg.yield %sum : i64
    } -> tensor<4xi64>
    // CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORTED]]#0, %[[KEYS]]
    flow.return %combined : tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]]#0 : tensor<4xi64>
  util.return %result : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @sort_with_escaping_key_alias
util.func public @sort_with_escaping_key_alias(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<2x2xi64>) {
  // CHECK: %[[RESULT:[a-zA-Z0-9_]+]]:2 = flow.dispatch.workgroups
  // CHECK-NEXT: (%{{[a-zA-Z0-9_]+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi64>>,
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<2x2xi64>) {
    %key_view = tensor.expand_shape %keys [[0, 1]] output_shape [2, 2]
        : tensor<4xi64> into tensor<2x2xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#1, %key_view : tensor<4xi64>, tensor<2x2xi64>
  }
  // CHECK: util.return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<4xi64>, tensor<2x2xi64>
  util.return %result#0, %result#1 : tensor<4xi64>, tensor<2x2xi64>
}

// -----

// CHECK-LABEL: util.func public @sort_with_tensor_view_input
util.func public @sort_with_tensor_view_input(
    %keys: tensor<2x2xi64>, %indices: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<2x2xi64>) {
  %key_view = tensor.collapse_shape %keys [[0, 1]]
      : tensor<2x2xi64> into tensor<4xi64>
  // CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = flow.dispatch.workgroups
  // CHECK-NEXT: (%{{[a-zA-Z0-9_]+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi64>>,
  // CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>)
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%key_view, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#1 : tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]], %{{.+}} : tensor<4xi64>, tensor<2x2xi64>
  util.return %result, %keys : tensor<4xi64>, tensor<2x2xi64>
}

// -----

// CHECK-LABEL: util.func public @sort_without_live_results
// CHECK-NOT: iree_linalg_ext.sort
util.func public @sort_without_live_results(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %indices : tensor<4xi64>
  }
  // CHECK: util.return
  util.return %result : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @generic_with_unused_result
util.func public @generic_with_unused_result(
    %first: tensor<4xi64>, %second: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = flow.dispatch.workgroups
  // CHECK-NEXT: (%{{[a-zA-Z0-9_]+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi64>>,
  // CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>)
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %generic:2 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        outs(%first, %second : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64):
      %sum = arith.addi %lhs, %rhs : i64
      %difference = arith.subi %sum, %rhs : i64
      linalg.yield %sum, %difference : i64, i64
    } -> (tensor<4xi64>, tensor<4xi64>)
    flow.return %generic#1 : tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]] : tensor<4xi64>
  util.return %result : tensor<4xi64>
}
