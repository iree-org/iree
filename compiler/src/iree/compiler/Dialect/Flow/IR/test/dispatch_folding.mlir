// RUN: iree-opt --allow-unregistered-dialect --split-input-file --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: util.func public @dontInlineReadWrite
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x4xf32>)
util.func public @dontInlineReadWrite(%arg0: tensor<1x4xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[CST:.+]] = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
  %cst = arith.constant dense<0.0> : tensor<4x8xf32>
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  //      CHECK: flow.dispatch.workgroups[{{.+}}](%[[ARG0]], %[[CST]]) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst
  // CHECK-NEXT:   (%{{.+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4xf32>>, %{{.+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x8xf32>>)
  %0 = flow.dispatch.workgroups[%x, %y](%arg0, %cst) : (tensor<1x4xf32>, tensor<4x8xf32>) -> %cst = (
    %arg0_capture: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4xf32>>,
    %arg1_capture: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x8xf32>>
  ) {
    "test.sink"(%arg0_capture) : (!iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4xf32>>) -> ()
    %load = iree_tensor_ext.dispatch.tensor.load %arg1_capture, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x8xf32>> -> tensor<4x8xf32>
    %0 = "test.do_work"(%load) : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
    iree_tensor_ext.dispatch.tensor.store %0, %arg1_capture, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x8xf32>>
    flow.return
  }
  util.return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: util.func public @remove_unused_result
util.func public @remove_unused_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>)
  //      CHECK: iree_tensor_ext.dispatch.tensor.store
  //  CHECK-NOT: iree_tensor_ext.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[9], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
    iree_tensor_ext.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
    flow.return
  }
  util.return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: util.func public @remove_unused_dynamic_result
util.func public @remove_unused_dynamic_result(%dim: index) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>)
  //  CHECK-NOT: flow.dispatch.tie_shape
  //      CHECK: iree_tensor_ext.dispatch.tensor.store
  //  CHECK-NOT: iree_tensor_ext.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%dim) : (index) -> (tensor<i32>, tensor<?xi32>{%dim}) =
      (%dim: index, %ret0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>, %ret1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi32>>) {
    // Used as a result; should remain after canonicalization.
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %ret0_init = tensor.empty() : tensor<i32>
    %ret0_value = linalg.fill ins(%c-2147483648_i32 : i32) outs(%ret0_init : tensor<i32>) -> tensor<i32>
    iree_tensor_ext.dispatch.tensor.store %ret0_value, %ret0, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>

    // Unused as a result; should be stripped entirely.
    %c0_i32 = arith.constant 0 : i32
    %ret1_shaped = flow.dispatch.tie_shape %ret1 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi32>>{%dim}
    %ret1_init = tensor.empty(%dim) : tensor<?xi32>
    %ret1_value = linalg.fill ins(%c0_i32 : i32) outs(%ret1_init : tensor<?xi32>) -> tensor<?xi32>
    iree_tensor_ext.dispatch.tensor.store %ret1_value, %ret1_shaped, offsets = [0], sizes = [%dim], strides = [1] : tensor<?xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi32>>{%dim}
    flow.return
  }
  util.return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: util.func public @remove_unused_read_write_result
util.func public @remove_unused_read_write_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> tensor<i32> =
  // CHECK-NEXT:   (%{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>)
  //      CHECK: iree_tensor_ext.dispatch.tensor.store %{{.+}},
  //  CHECK-NOT: iree_tensor_ext.dispatch.tensor.store
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<i32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[9], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
    iree_tensor_ext.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<i32>>
    flow.return
  }
  util.return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: util.func public @keep_used_read_write_result
util.func public @keep_used_read_write_result(%arg0 : tensor<9xi32>, %arg1 : tensor<9xi32>) -> (tensor<i32>) {
  %c1 = arith.constant 1 : index
  //      CHECK: flow.dispatch.workgroups[%c1]() : () -> (tensor<i32>, tensor<i32>) =
  // CHECK-NEXT:   (%{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>, %{{.+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<i32>>)
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1) : (tensor<9xi32>, tensor<9xi32>) -> (tensor<i32>, tensor<i32>) =
      (%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<i32>>) {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<i32>> -> tensor<i32>
    %val = tensor.extract %0[] : tensor<i32>
    %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets=[0], sizes=[9], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9xi32>> -> tensor<9xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %4 = linalg.fill ins(%val : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
    iree_tensor_ext.dispatch.tensor.store %4, %arg3, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<i32>>
    flow.return
  }
  util.return %0#0 : tensor<i32>
}

// -----

// CHECK-LABEL: util.func public @multiple_results_tied_to_same_input
util.func public @multiple_results_tied_to_same_input(%arg0: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %c1 = arith.constant 1 : index
  // CHECK: flow.dispatch.workgroups[%c1](%arg0) : (tensor<4x4xf32>) -> (%arg0, %arg0) =
  // CHECK-NEXT: (%{{.+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>>)
  %0:2 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0) :
      (tensor<4x4xf32>) -> (%arg0, %arg0) =
      (%arg0_capture: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>>) {
    %1 = iree_tensor_ext.dispatch.tensor.load %arg0_capture,
        offsets = [0, 0], sizes = [4, 4], strides = [1, 1]
        : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>> -> tensor<4x4xf32>
    iree_tensor_ext.dispatch.tensor.store %1, %arg0_capture,
        offsets = [0, 0], sizes = [4, 4], strides = [1, 1]
        : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>>
    flow.return
  }
  util.return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: util.func public @drop_unused_dispatch_region_result
util.func public @drop_unused_dispatch_region_result(
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
  // CHECK: util.return %[[r]]
  util.return %r#0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: util.func public @keep_required_sort_carrier
util.func public @keep_required_sort_carrier(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: %[[RESULT:.+]]:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
    // CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    // CHECK: flow.return %[[SORT]]#1, %[[SORT]]#0 : tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#1, %sorted#0 : tensor<4xi64>, tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]]#0 : tensor<4xi64>
  util.return %result#0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @drop_required_sort_carrier_when_base_live
//  CHECK-SAME: (%[[KEYS:.+]]: tensor<4xi64>, %[[INDICES:.+]]: tensor<4xi64>)
util.func public @drop_required_sort_carrier_when_base_live(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<4xi64>) {
  // CHECK: %[[RESULT:.+]] = flow.dispatch.region -> (tensor<4xi64>) {
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    // CHECK: flow.return %{{.+}}#1 : tensor<4xi64>
    flow.return %sorted#0, %sorted#1 : tensor<4xi64>, tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]], %[[KEYS]]
  util.return %result#1, %keys : tensor<4xi64>, tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @drop_required_sort_carrier_when_alias_live
//  CHECK-SAME: (%[[KEYS:.+]]: tensor<4xi64>, %[[INDICES:.+]]: tensor<4xi64>)
util.func public @drop_required_sort_carrier_when_alias_live(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<2x2xi64>) {
  %key_view = flow.tensor.reshape %keys : tensor<4xi64> -> tensor<2x2xi64>
  // CHECK: %[[RESULT:.+]] = flow.dispatch.region -> (tensor<4xi64>) {
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    // CHECK: flow.return %{{.+}}#1 : tensor<4xi64>
    flow.return %sorted#0, %sorted#1 : tensor<4xi64>, tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]], %[[KEY_VIEW:.+]]
  util.return %result#1, %key_view : tensor<4xi64>, tensor<2x2xi64>
}

// -----

// CHECK-LABEL: util.func public @drop_required_carriers_with_dead_owner
util.func public @drop_required_carriers_with_dead_owner(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: %[[RESULT:.+]] = flow.dispatch.region -> (tensor<4xi64>) {
  // CHECK-NOT: iree_linalg_ext.sort
  %result:3 = flow.dispatch.region ->
      (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    // CHECK: flow.return %{{.+}} : tensor<4xi64>
    flow.return %sorted#0, %sorted#1, %indices
        : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]] : tensor<4xi64>
  util.return %result#2 : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @drop_generic_dps_sibling
util.func public @drop_generic_dps_sibling(
    %first: tensor<4xi64>, %second: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: %[[RESULT:.+]] = flow.dispatch.region -> (tensor<4xi64>) {
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
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
    // CHECK: flow.return %{{.+}} : tensor<4xi64>
    flow.return %generic#0, %generic#1 : tensor<4xi64>, tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]] : tensor<4xi64>
  util.return %result#1 : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @keep_required_sort_carrier_with_live_same_base
util.func public @keep_required_sort_carrier_with_live_same_base(
    %storage: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: %[[RESULT:.+]]:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%storage, %storage : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_value: i64, %rhs_value: i64, %lhs_key: i64, %rhs_key: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    // CHECK: flow.return %[[SORTED:.+]]#0, %[[SORTED]]#1 : tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#0, %sorted#1 : tensor<4xi64>, tensor<4xi64>
  }
  // CHECK: util.return %[[RESULT]]#0 : tensor<4xi64>
  util.return %result#0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @drop_external_sort_carrier
util.func public @drop_external_sort_carrier(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<4xi64>) {
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
  ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
    %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
    iree_linalg_ext.yield %take_lhs : i1
  } -> tensor<4xi64>, tensor<4xi64>
  // CHECK: %[[FORWARDED:.+]] = flow.dispatch.region -> (tensor<4xi64>) {
  %forwarded:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
    // CHECK: flow.return %{{.+}} : tensor<4xi64>
    flow.return %sorted#0, %indices : tensor<4xi64>, tensor<4xi64>
  }
  // CHECK: util.return %{{.+}}, %[[FORWARDED]] : tensor<4xi64>, tensor<4xi64>
  util.return %sorted#0, %forwarded#1 : tensor<4xi64>, tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @remove_redundant_results
//  CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>)
util.func @remove_redundant_results(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]]
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  // CHECK: %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<?xf32>{%[[DIM]]}
  %0:3 = flow.dispatch.region -> (tensor<?xf32>{%d0}, tensor<?xf32>{%d0}, tensor<?xf32>{%d0}) {
    // CHECK-NEXT: flow.return %[[ARG0]] : tensor<?xf32>
    flow.return %arg0, %arg0, %arg0 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
  }
  // CHECK: util.return %[[DISPATCH]], %[[DISPATCH]]
  util.return %0#0, %0#2 : tensor<?xf32>, tensor<?xf32>
}
