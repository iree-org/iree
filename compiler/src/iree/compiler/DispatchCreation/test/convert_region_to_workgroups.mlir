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

util.global private @device : !hal.device

// The unused comparator result needs storage, but Sort initializes it itself.
// CHECK-LABEL: util.func public @sort_workspace
// CHECK: %[[WORKSPACE:.+]] = flow.tensor.alloca : tensor<?xi64>
// CHECK-NOT: flow.tensor.alloca
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK-SAME: -> (tensor<?xi64>{{.*}}, %[[WORKSPACE]]
// CHECK-SAME: stream.affinity = #hal.device.affinity<@device>
// CHECK: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>
// CHECK: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort dimension(0) ins(
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0
// CHECK: count(%[[N:.+]]: index)
// CHECK: flow.return %[[N]], %{{.+}}, %{{.+}} : index, index, index
// CHECK-NOT: flow.dispatch.workgroups
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_workspace(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region[%n] -> (tensor<?xi64>{%n}) attributes {
    stream.affinity = #hal.device.affinity<@device>
  } {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1 : tensor<?xi64>
  } count(%workload: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    flow.return %workload, %c1, %c1 : index, index, index
  }
  util.return %result : tensor<?xi64>
}

// -----

// Live dispatch results already provide destinations for both Sort outputs.
// CHECK-LABEL: util.func public @sort_returned_keys
// CHECK-NOT: flow.tensor.alloca
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1
// CHECK-NOT: flow.tensor.alloca
// CHECK: util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1
util.func public @sort_returned_keys(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#0, %sorted#1 : tensor<?xi64>, tensor<?xi64>
  }
  util.return %result#0, %result#1 : tensor<?xi64>, tensor<?xi64>
}

// -----

// Equal inputs do not give two independently mutated outputs the same storage.
// CHECK-LABEL: util.func public @sort_shared_keys
// CHECK: %[[WORK0:.+]] = flow.tensor.alloca
// CHECK: %[[WORK1:.+]] = flow.tensor.alloca
// CHECK-NOT: flow.tensor.alloca
// CHECK: %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups
// CHECK-SAME: -> (tensor<?xi64>{%{{.+}}}, %[[WORK0]]{%{{.+}}}, %[[WORK1]]{%{{.+}}})
// CHECK: %[[SORT:.+]]:3 = iree_linalg_ext.sort
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#2
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_shared_keys(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %sorted:3 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %keys, %labels : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty, %empty : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs1: i64, %rhs1: i64,
         %lhs_label: i64, %rhs_label: i64):
      %cmp0 = arith.cmpi sle, %lhs, %rhs : i64
      %cmp1 = arith.cmpi sle, %lhs1, %rhs1 : i64
      %cmp = arith.andi %cmp0, %cmp1 : i1
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#2 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----

// The hidden store must follow a unique update instead of pinning the earlier
// Sort result. A shape observation does not require preserving its data.
// CHECK-LABEL: util.func public @sort_updated_workspace
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort
// CHECK: %[[UPDATED:.+]] = linalg.generic
// CHECK: %[[COMBINED:.+]] = linalg.generic
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[COMBINED]]
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[UPDATED]]
// CHECK-NOT: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_updated_workspace(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %size = tensor.dim %sorted#0, %c0 : tensor<?xi64>
    %size_i64 = arith.index_cast %size : index to i64
    %updated = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        outs(%sorted#0 : tensor<?xi64>) {
    ^bb0(%key: i64):
      %one = arith.constant 1 : i64
      %sum = arith.addi %key, %one : i64
      linalg.yield %sum : i64
    } -> tensor<?xi64>
    // Keep the updated tensor live without returning its data.
    %observed = tensor.extract %updated[%c0] : tensor<?xi64>
    %combined = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        outs(%sorted#1 : tensor<?xi64>) {
    ^bb0(%label: i64):
      %value = arith.addi %label, %observed : i64
      %with_size = arith.addi %value, %size_i64 : i64
      linalg.yield %with_size : i64
    } -> tensor<?xi64>
    flow.return %combined : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----

// An unused lane that does not participate in the comparator needs no storage.
// CHECK-LABEL: util.func public @sort_unused_payload
// CHECK-NOT: flow.tensor.alloca
// CHECK: iree_linalg_ext.sort dimension(0) ins(%{{.+}} : tensor<?xi64>)
// CHECK-NOT: flow.tensor.alloca
// CHECK: util.return
util.func public @sort_unused_payload(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#0 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----

// Replacing a later Sort destination must not disconnect the earlier workspace.
// CHECK-LABEL: util.func public @chained_output_inits
// CHECK: %[[DISPATCH:.+]]:4 = flow.dispatch.workgroups
// CHECK: %[[FIRST:.+]]:2 = iree_linalg_ext.sort
// CHECK: %[[SECOND:.+]]:2 = iree_linalg_ext.sort
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SECOND]]#1
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SECOND]]#0
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[FIRST]]#0
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[FIRST]]#1
// CHECK: util.return %[[DISPATCH]]#0
util.func public @chained_output_inits(%keys: tensor<?xi64>, %labels: tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %first:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %ll: i64, %rl: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %second:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %first#1 : tensor<?xi64>, tensor<?xi64>)
        outs(%first#0, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %ll: i64, %rl: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %second#1 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----

// Bounded static workspaces keep private storage instead of adding bindings.
// CHECK-LABEL: util.func public @sort_static_workspace
// CHECK-NOT: flow.tensor.alloca
// CHECK: %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1
// CHECK-NOT: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0
// CHECK-NOT: flow.tensor.alloca
// CHECK: util.return %[[DISPATCH]]
util.func public @sort_static_workspace(%keys: tensor<4xi64>, %labels: tensor<4xi64>)
    -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %empty = tensor.empty() : tensor<4xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<4xi64>, tensor<4xi64>)
        outs(%empty, %empty : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#1 : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----

// A dynamic batch does not require workspace bindings for a fixed sorting axis.
// CHECK-LABEL: util.func public @sort_dynamic_batch_static_axis
// CHECK-NOT: flow.tensor.alloca
// CHECK: %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1
// CHECK-NOT: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0
// CHECK-NOT: flow.tensor.alloca
// CHECK: util.return %[[DISPATCH]]
util.func public @sort_dynamic_batch_static_axis(%keys: tensor<?x4xi64>, %labels: tensor<?x4xi64>)
    -> tensor<?x4xi64> {
  %c0 = arith.constant 0 : index
  %batch = tensor.dim %keys, %c0 : tensor<?x4xi64>
  %result = flow.dispatch.region -> (tensor<?x4xi64>{%batch}) {
    %empty = tensor.empty(%batch) : tensor<?x4xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(1)
        ins(%keys, %labels : tensor<?x4xi64>, tensor<?x4xi64>)
        outs(%empty, %empty : tensor<?x4xi64>, tensor<?x4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?x4xi64>, tensor<?x4xi64>
    flow.return %sorted#1 : tensor<?x4xi64>
  }
  util.return %result : tensor<?x4xi64>
}
