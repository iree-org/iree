// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-hoist-uniform-scalar-compute))" --split-input-file %s | FileCheck %s

util.func public @no_hoist_constants(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: index, %n: index) -> tensor<?x?xf32> {
  %0 = flow.dispatch.region -> (tensor<?x?xf32>{%m, %n}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.matmul
        ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %2 : tensor<?x?xf32>
  }
  util.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @no_hoist_constants
// CHECK:       flow.dispatch.region
// CHECK-DAG:     arith.constant 0.0{{0+}}e+00 : f32
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[D0:.+]] = tensor.dim %{{.*}}, %[[C0]]
// CHECK-DAG:     %[[D1:.+]] = tensor.dim %{{.*}}, %[[C1]]
// CHECK:         tensor.empty(%[[D0]], %[[D1]])
// CHECK:         flow.return

// -----

util.func public @hoist_arithmetic(%m: index, %n: index) -> tensor<?x?xf32> {
  %0 = flow.dispatch.region -> (tensor<?x?xf32>{%m, %n}) {
    %c3 = arith.constant 3 : index
    %mul = arith.muli %m, %c3 : index
    %1 = tensor.empty(%mul, %n) : tensor<?x?xf32>
    flow.return %1 : tensor<?x?xf32>
  }
  util.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @hoist_arithmetic
// CHECK:       %[[C3:.+]] = arith.constant 3
// CHECK:       %[[MUL:.+]] = arith.muli %{{.*}}, %[[C3]]
// CHECK:       flow.dispatch.region
// CHECK:         tensor.empty(%[[MUL]]
// CHECK:         flow.return

// -----

util.func public @no_hoist_dep_inside_dispatch(%arg0: tensor<?xf32>, %m: index) -> tensor<?xf32> {
  %0 = flow.dispatch.region -> (tensor<?xf32>{%m}) {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
    %c3 = arith.constant 3 : index
    %mul = arith.muli %d0, %c3 : index
    %1 = tensor.empty(%mul) : tensor<?xf32>
    flow.return %1 : tensor<?xf32>
  }
  util.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @no_hoist_dep_inside_dispatch
// CHECK:       flow.dispatch.region
// CHECK-DAG:     arith.constant 0
// CHECK-DAG:     arith.constant 3
// CHECK:         tensor.dim
// CHECK:         arith.muli
// CHECK:         tensor.empty
// CHECK:         flow.return

// -----

util.func public @hoist_one_element_tensor_extract(%arg0: tensor<1xi64>, %m: index) -> tensor<?xf32> {
  %0 = flow.dispatch.region -> (tensor<?xf32>{%m}) {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<1xi64>
    %idx = arith.index_cast %extracted : i64 to index
    %sum = arith.addi %idx, %m : index
    %1 = tensor.empty(%sum) : tensor<?xf32>
    flow.return %1 : tensor<?xf32>
  }
  util.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @hoist_one_element_tensor_extract
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1xi64>, %[[M:.+]]: index
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[C0]]] : tensor<1xi64>
// CHECK:       %[[IDX:.+]] = arith.index_cast %[[EXTRACT]] : i64 to index
// CHECK:       %[[SUM:.+]] = arith.addi %[[IDX]], %[[M]] : index
// CHECK:       flow.dispatch.region
// CHECK:         tensor.empty(%[[SUM]])
// CHECK:         flow.return

// -----

util.func public @dont_hoist_extract_from_multi_element_tensor(%arg0: tensor<4xi64>, %i: index, %m: index) -> tensor<?xf32> {
  %0 = flow.dispatch.region -> (tensor<?xf32>{%m}) {
    %extracted = tensor.extract %arg0[%i] : tensor<4xi64>
    %idx = arith.index_cast %extracted : i64 to index
    %sum = arith.addi %idx, %m : index
    %1 = tensor.empty(%sum) : tensor<?xf32>
    flow.return %1 : tensor<?xf32>
  }
  util.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @dont_hoist_extract_from_multi_element_tensor
// CHECK:       flow.dispatch.region
// CHECK:         %[[EXTRACT:.+]] = tensor.extract
// CHECK:         %[[IDX:.+]] = arith.index_cast %[[EXTRACT]]
// CHECK:         %[[SUM:.+]] = arith.addi %[[IDX]]
// CHECK:         tensor.empty(%[[SUM]])
// CHECK:         flow.return
