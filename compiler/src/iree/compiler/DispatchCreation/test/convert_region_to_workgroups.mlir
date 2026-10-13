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

// CHECK-LABEL: util.func public @sort_unused_comparator_workspace
// CHECK: %[[KEY_STORAGE:.+]] = flow.tensor.alloca
// CHECK: %[[KEYS:.+]] = flow.dispatch.workgroups
// CHECK-SAME: -> %[[KEY_STORAGE]]
// CHECK-SAME: stream.affinity = #hal.device.affinity<@device>
// CHECK: linalg.copy
// CHECK: %[[LABELS:.+]] = flow.dispatch.workgroups
// CHECK: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[SORT_DISPATCH:.+]]:2 = flow.dispatch.workgroups[%{{.+}}](%[[KEYS]], %[[LABELS]],
// CHECK-SAME: -> (%[[LABELS]]{{.*}}, %[[KEYS]]{{.*}})
// CHECK-SAME: stream.affinity = #hal.device.affinity<@device>
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK-NOT: linalg.copy
// CHECK: count(%[[WORKLOAD:.+]]: index)
// CHECK: flow.return %[[WORKLOAD]], %{{.+}}, %{{.+}} : index, index, index
// CHECK: util.return %[[SORT_DISPATCH]]#0
util.func public @sort_unused_comparator_workspace(
    %keys: tensor<?xi64>, %indices: tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region[%n] -> (tensor<?xi64>{%n}) attributes {
    stream.affinity = #hal.device.affinity<@device>
  } {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
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
// CHECK-LABEL: util.func public @sort_preserves_internal_key_view
// CHECK-COUNT-2: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi64>>
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-COUNT-2: iree_tensor_ext.dispatch.tensor.store
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_preserves_internal_key_view(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %view = tensor.expand_shape %keys [[0, 1]] output_shape [1, 4]
        : tensor<4xi64> into tensor<1x4xi64>
    %combined = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (0, d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%view : tensor<1x4xi64>) outs(%sorted#1 : tensor<4xi64>) {
    ^bb0(%key: i64, %index: i64):
      %sum = arith.addi %key, %index : i64
      linalg.yield %sum : i64
    } -> tensor<4xi64>
    flow.return %combined : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----
// Equal sources have independent storage for their different working roles.
// CHECK-LABEL: util.func public @sort_shared_comparator_sources
// CHECK: %[[STORAGE0:.+]] = flow.tensor.alloca
// CHECK: %[[KEY0:.+]] = flow.dispatch.workgroups
// CHECK-SAME: -> %[[STORAGE0]]
// CHECK: linalg.copy
// CHECK: %[[STORAGE1:.+]] = flow.tensor.alloca
// CHECK: %[[KEY1:.+]] = flow.dispatch.workgroups
// CHECK-SAME: -> %[[STORAGE1]]
// CHECK: linalg.copy
// CHECK: %[[LABELS:.+]] = flow.dispatch.workgroups
// CHECK: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups(%[[KEY0]], %[[KEY1]], %[[LABELS]])
// CHECK-SAME: -> (%[[LABELS]], %[[KEY0]], %[[KEY1]])
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK-COUNT-3: iree_tensor_ext.dispatch.tensor.store
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_shared_comparator_sources(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:3 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %keys, %indices
             : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs0: i64, %rhs0: i64, %lhs1: i64, %rhs1: i64,
         %lhs_index: i64, %rhs_index: i64):
      %cmp0 = arith.cmpi sle, %lhs0, %rhs0 : i64
      %cmp1 = arith.cmpi sle, %lhs1, %rhs1 : i64
      %cmp = arith.andi %cmp0, %cmp1 : i1
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#2 : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----
// CHECK-LABEL: util.func public @sort_trailing_rhs_comparator
// CHECK: flow.tensor.alloca : tensor<?xf32>
// CHECK: %[[KEYS:.+]] = flow.dispatch.workgroups
// CHECK: linalg.copy ins(%{{.+}} : tensor<?xf32>)
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK-SAME: -> ({{.*}}, %[[KEYS]]{{.*}})
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_trailing_rhs_comparator(
    %indices: tensor<?xi64>, %keys: tensor<?xf32>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xf32>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%indices, %keys : tensor<?xi64>, tensor<?xf32>) {
    ^bb0(%lhs_index: i64, %rhs_index: i64, %lhs: f32, %rhs: f32):
      %zero = arith.constant 0.0 : f32
      %cmp = arith.cmpf oge, %rhs, %zero : f32
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xf32>
    flow.return %sorted#0 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----
// Returning both lanes must not add duplicate hidden results.
// CHECK-LABEL: util.func public @sort_returned_keys
// CHECK-COUNT-2: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK-COUNT-2: iree_tensor_ext.dispatch.tensor.store
// CHECK: util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1
util.func public @sort_returned_keys(%keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<4xi64>) {
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#0, %sorted#1 : tensor<4xi64>, tensor<4xi64>
  }
  util.return %result#0, %result#1 : tensor<4xi64>, tensor<4xi64>
}

// -----
// An unused noncomparator lane needs no storage or initialization.
// CHECK-LABEL: util.func public @sort_unused_noncomparator_lane
// CHECK: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: iree_linalg_ext.sort dimension(0) outs(%{{.+}} : tensor<4xi64>)
// CHECK-NOT: linalg.copy
// CHECK: util.return
util.func public @sort_unused_noncomparator_lane(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#0 : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----
// Move the pure initializer once; its untied result already owns storage.
// CHECK-LABEL: util.func public @sort_local_keys
// CHECK: %[[KEYS:.+]] = flow.dispatch.workgroups
// CHECK: linalg.generic
// CHECK: iree_tensor_ext.dispatch.tensor.store
// CHECK: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups(%[[KEYS]],
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.copy
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_local_keys(%indices: tensor<?xi64>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %indices, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %keys = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]} outs(%empty : tensor<?xi64>) {
    ^bb0(%out: i64):
      %index = linalg.index 0 : index
      %value = arith.index_cast %index : index to i64
      linalg.yield %value : i64
    } -> tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----
// CHECK-LABEL: util.func public @dead_sort
// CHECK-NOT: flow.tensor.alloca
// CHECK-NOT: linalg.copy
// CHECK-NOT: iree_linalg_ext.sort
// CHECK: util.return
util.func public @dead_sort(%keys: tensor<4xi64>, %indices: tensor<4xi64>) {
  flow.dispatch.region {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return
  }
  util.return
}

// -----
// CHECK-LABEL: util.func public @shape_only_sort
// CHECK-NOT: flow.tensor.alloca
// CHECK-NOT: linalg.copy
// CHECK-NOT: iree_linalg_ext.sort
// CHECK: util.return
util.func public @shape_only_sort(%keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %c0 = arith.constant 0 : index
    %n = tensor.dim %sorted#1, %c0 : tensor<4xi64>
    %empty = tensor.empty(%n) : tensor<?xi64>
    %one = arith.constant 1 : i64
    %filled = linalg.fill ins(%one : i64) outs(%empty : tensor<?xi64>)
        -> tensor<?xi64>
    %cast = tensor.cast %filled : tensor<?xi64> to tensor<4xi64>
    flow.return %cast : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----
// Shape queries must not break the workspace tie.
// CHECK-LABEL: util.func public @sort_key_shape_use
// CHECK-COUNT-2: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK: linalg.fill
// CHECK-COUNT-3: iree_tensor_ext.dispatch.tensor.store
// CHECK: util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1
util.func public @sort_key_shape_use(%keys: tensor<?xi64>, %indices: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %c0_inner = arith.constant 0 : index
    %size = tensor.dim %sorted#0, %c0_inner : tensor<?xi64>
    %empty = tensor.empty(%size) : tensor<?xi64>
    %one = arith.constant 1 : i64
    %filled = linalg.fill ins(%one : i64) outs(%empty : tensor<?xi64>)
        -> tensor<?xi64>
    flow.return %sorted#1, %filled : tensor<?xi64>, tensor<?xi64>
  }
  util.return %result#0, %result#1 : tensor<?xi64>, tensor<?xi64>
}

// -----
// The second Sort must start from the original tensor version.
// CHECK-LABEL: util.func public @two_sorts_shared_source
// CHECK-COUNT-4: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:4 = flow.dispatch.workgroups
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK-COUNT-4: iree_tensor_ext.dispatch.tensor.store
// CHECK: util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1
util.func public @two_sorts_shared_source(
    %keys: tensor<?xi64>, %indices0: tensor<?xi64>, %indices1: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices0 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %ascending#1, %descending#1 : tensor<?xi64>, tensor<?xi64>
  }
  util.return %result#0, %result#1 : tensor<?xi64>, tensor<?xi64>
}

// -----
util.func private @observe()

// Removing pure dead computation must preserve required evaluation.
// CHECK-LABEL: util.func public @sort_dead_internal_chain
// CHECK-NOT: iree_linalg_ext.sort
// CHECK-NOT: linalg.copy
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = flow.dispatch.workgroups
// CHECK-NOT: iree_linalg_ext.sort
// CHECK-NOT: linalg.copy
// CHECK: util.call @observe()
// CHECK: util.return %[[RESULT]]
util.func public @sort_dead_internal_chain(
    %keys: tensor<?xi64>, %indices: tensor<?xi64>, %other: tensor<4xi64>)
    -> tensor<4xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %indices, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %cast = tensor.cast %sorted#1 : tensor<?xi64> to tensor<?xi64>
    %dead = tensor.extract_slice %cast[0] [%n] [1]
        : tensor<?xi64> to tensor<?xi64>
    util.call @observe() : () -> ()
    flow.return %other : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----
// Dead comparator uses must not keep an unused lane alive.
// CHECK-LABEL: util.func public @sort_dead_comparator_user
// CHECK-COUNT-2: linalg.copy
// CHECK-NOT: linalg.copy
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK-NOT: arith.addi
// CHECK-COUNT-2: iree_tensor_ext.dispatch.tensor.store
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_dead_comparator_user(
    %keys: tensor<?xi64>, %extra: tensor<?xi64>, %indices: tensor<?xi64>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %sorted:3 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %extra, %indices
             : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_extra: i64, %rhs_extra: i64,
         %lhs_index: i64, %rhs_index: i64):
      %dead = arith.addi %lhs_extra, %rhs_extra : i64
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#2 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----

util.func private @produce_keys() -> tensor<4xi64>

// An effectful initializer must remain in its original dispatch.
// CHECK-LABEL: util.func public @sort_effectful_initializer
// CHECK: linalg.copy
// CHECK-NOT: flow.tensor.alloca
// CHECK: flow.dispatch.workgroups
// CHECK: util.call @produce_keys
// CHECK: iree_linalg_ext.sort
// CHECK-NOT: flow.dispatch.workgroups
// CHECK: util.return
util.func public @sort_effectful_initializer(%indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %keys = util.call @produce_keys() : () -> tensor<4xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#1 : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----

// Dispatch results cannot export a scalar shared by the initializer/consumer.
// CHECK-LABEL: util.func public @sort_scalar_initializer_use
// CHECK: linalg.copy
// CHECK-NOT: flow.tensor.alloca
// CHECK: flow.dispatch.workgroups
// CHECK: linalg.generic
// CHECK: iree_linalg_ext.sort
// CHECK: linalg.generic
// CHECK-NOT: flow.dispatch.workgroups
// CHECK: util.return
util.func public @sort_scalar_initializer_use(%indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %offset = arith.constant 100 : i64
    %empty = tensor.empty() : tensor<4xi64>
    %keys = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]} outs(%empty : tensor<4xi64>) {
    ^bb0(%out: i64):
      %index = linalg.index 0 : index
      %value = arith.index_cast %index : index to i64
      %key = arith.subi %offset, %value : i64
      linalg.yield %key : i64
    } -> tensor<4xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %combined = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]} outs(%sorted#1 : tensor<4xi64>) {
    ^bb0(%index: i64):
      %value = arith.addi %offset, %index : i64
      linalg.yield %value : i64
    } -> tensor<4xi64>
    flow.return %combined : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----

// A dependent Sort updates the hidden role; only its final version is stored.
// CHECK-LABEL: util.func public @sort_dependent_workspace
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: %[[ASCENDING:.+]]:2 = iree_linalg_ext.sort
// CHECK: %[[DESCENDING:.+]]:2 = iree_linalg_ext.sort
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[DESCENDING]]#1
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[DESCENDING]]#0
// CHECK-NOT: iree_tensor_ext.dispatch.tensor.store %[[ASCENDING]]#0
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_dependent_workspace(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%ascending#0, %ascending#1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %descending#1 : tensor<?xi64>
  }
  util.return %result, %keys, %labels : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}

// -----

// Shape reads permit forwarding through an intervening destination update.
// CHECK-LABEL: util.func public @sort_local_workspace_update
// CHECK: %[[KEYS:.+]] = flow.dispatch.workgroups
// CHECK: %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups
// CHECK: %[[ASCENDING:.+]]:2 = iree_linalg_ext.sort
// CHECK: %[[NEGATED:.+]] = linalg.generic
// CHECK: %[[DESCENDING:.+]]:2 = iree_linalg_ext.sort
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[DESCENDING]]#1
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[DESCENDING]]#0
// CHECK-NOT: iree_tensor_ext.dispatch.tensor.store %[[ASCENDING]]#0
// CHECK: util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1
#identity = affine_map<(d0) -> (d0)>
util.func public @sort_local_workspace_update(%labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %labels, %c0 : tensor<?xi64>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %keys = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]}
        outs(%empty : tensor<?xi64>) {
    ^bb0(%out: i64):
      %zero = arith.constant 0 : i64
      %i = linalg.index 0 : index
      %value = arith.index_cast %i : index to i64
      %key = arith.subi %zero, %value : i64
      linalg.yield %key : i64
    } -> tensor<?xi64>
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %size = tensor.dim %ascending#0, %c0 : tensor<?xi64>
    %shape_empty = tensor.empty(%size) : tensor<?xi64>
    %seven = arith.constant 7 : i64
    %filled = linalg.fill ins(%seven : i64) outs(%shape_empty : tensor<?xi64>) -> tensor<?xi64>
    %negated = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]}
        outs(%ascending#0 : tensor<?xi64>) {
    ^bb0(%key: i64):
      %zero = arith.constant 0 : i64
      %negative = arith.subi %zero, %key : i64
      linalg.yield %negative : i64
    } -> tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%negated, %ascending#1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %descending#1, %filled : tensor<?xi64>, tensor<?xi64>
  }
  util.return %results#0, %results#1, %labels : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}

// -----

// Another data user observes the earlier version; keep that workspace export.
// CHECK-LABEL: util.func public @sort_workspace_earlier_data_use
// CHECK: %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups
// CHECK: %[[ASCENDING:.+]]:2 = iree_linalg_ext.sort
// CHECK: %[[DESCENDING:.+]]:2 = iree_linalg_ext.sort
// CHECK: linalg.generic
// CHECK-NOT: iree_tensor_ext.dispatch.tensor.store %[[DESCENDING]]#0
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[ASCENDING]]#0
// CHECK: util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1
#identity = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
util.func public @sort_workspace_earlier_data_use(%keys: tensor<4xi64>, %labels: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>) {
  %zero = arith.constant 0 : i64
  %sum_empty = tensor.empty() : tensor<i64>
  %sum_init = linalg.fill ins(%zero : i64) outs(%sum_empty : tensor<i64>) -> tensor<i64>
  %results:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<i64>) {
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%ascending#0, %ascending#1 : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %sum = linalg.generic {indexing_maps = [#identity, #scalar], iterator_types = ["reduction"]}
        ins(%ascending#0 : tensor<4xi64>) outs(%sum_init : tensor<i64>) {
    ^bb0(%key: i64, %acc: i64):
      %position = linalg.index 0 : index
      %position_i64 = arith.index_cast %position : index to i64
      %one = arith.constant 1 : i64
      %weight = arith.addi %position_i64, %one : i64
      %weighted = arith.muli %weight, %key : i64
      %next = arith.addi %weighted, %acc : i64
      linalg.yield %next : i64
    } -> tensor<i64>
    flow.return %descending#1, %sum : tensor<4xi64>, tensor<i64>
  }
  util.return %results#0, %results#1, %keys, %labels
      : tensor<4xi64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>
}
