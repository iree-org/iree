// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-stream-conversion %s | FileCheck %s

// CHECK: stream.executable private @executable
flow.executable private @executable {
  // CHECK: stream.executable.export public @dispatch
  flow.executable.export public @dispatch
  builtin.module {
    // CHECK: func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %[[ARG0_DIM0:.+]]: index, %[[ARG1_DIM1:.+]]: index)
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readonly:tensor<?x4xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>,
                   %arg0_dim0: index, %arg1_dim1: index) {
      // CHECK: %[[ARG0_TENSOR:.+]] = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x4xf32>>{%[[ARG0_DIM0]]}
      %arg0_tied = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x4xf32>>{%arg0_dim0}
      // CHECK: %[[ARG1_TENSOR:.+]] = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>{%[[ARG1_DIM1]]}
      %arg1_tied = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>{%arg1_dim1}

      // CHECK: %[[TILE:.+]] = flow.dispatch.tensor.load %[[ARG0_TENSOR]], offsets = [0, 0], sizes = [%[[ARG0_DIM0]], 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x4xf32>>{%[[ARG0_DIM0]]} -> tensor<?x4xf32>
      %0 = flow.dispatch.tensor.load %arg0_tied, offsets = [0, 0], sizes = [%arg0_dim0, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x4xf32>>{%arg0_dim0} -> tensor<?x4xf32>
      // CHECK: flow.dispatch.tensor.store %[[TILE]], %[[ARG1_TENSOR]], offsets = [0, 0], sizes = [%[[ARG0_DIM0]], 4], strides = [1, 1] : tensor<?x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>{%[[ARG1_DIM1]]}
      flow.dispatch.tensor.store %0, %arg1_tied, offsets = [0, 0], sizes = [%arg0_dim0, 4], strides = [1, 1] : tensor<?x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x?xf32>>{%arg1_dim1}

      return
    }
  }
}

// CHECK-LABEL: @simple_mul
func.func @simple_mul(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: %[[DIM0:.+]] = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %dim0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  // CHECK: hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("tensor") shape([%0, %c4]) type(%c553648160_i32) encoding(%c1_i32)
  // CHECK: %[[ARG0_SIZE:.+]] = stream.tensor.sizeof tensor<?x4xf32>{%[[DIM0]]} : index
  // CHECK: %[[ARG0_IMPORT:.+]] = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x4xf32>{%[[DIM0]]} in !stream.resource<external>{%[[ARG0_SIZE]]}
  // CHECK: %[[ARG0_T:.+]] = stream.async.transfer %[[ARG0_IMPORT]] : !stream.resource<external>{%[[ARG0_SIZE]]} -> !stream.resource<*>{%[[ARG0_SIZE]]}
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x4xf32>{%dim0}

  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[RET0_SIZE:.+]] = stream.tensor.sizeof tensor<?xf32>{%[[DIM0]]} : index
  // CHECK: %[[RET0:.+]] = stream.async.dispatch @executable::@dispatch[%c2, %c1, %c1](%[[ARG0_T]][%c0 to %[[ARG0_SIZE]] for %[[ARG0_SIZE]]]) : (!stream.resource<*>{%[[ARG0_SIZE]]}) -> !stream.resource<*>{%[[RET0_SIZE]]}
  %1 = flow.dispatch @executable::@dispatch[%c2, %c1, %c1](%0) : (tensor<?x4xf32>{%dim0}) -> tensor<?xf32>{%dim0}

  // CHECK: %[[RET0_T:.+]] = stream.async.transfer %[[RET0]] : !stream.resource<*>{%[[RET0_SIZE]]} -> !stream.resource<external>{%[[RET0_SIZE]]}
  // CHECK: %[[RET0_EXPORT:.+]] = stream.tensor.export %[[RET0_T]] : tensor<?xf32>{%[[DIM0]]} in !stream.resource<external>{%[[RET0_SIZE]]} -> !hal.buffer_view
  %2 = hal.tensor.export %1 : tensor<?xf32>{%dim0} -> !hal.buffer_view
  // CHECK: return %[[RET0_EXPORT]] : !hal.buffer_view
  return %2 : !hal.buffer_view
}

// -----

// Tests that ops consuming/producing tensors in other dialects pass through ok.

// CHECK-LABEL: @custom_ops
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<*>, %[[ARG_SIZE:.+]]: index) -> (!stream.resource<*>, index)
func.func @custom_ops(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // CHECK: %[[ARG_EXTERNAL:.+]] = stream.async.transfer %[[ARG]]
  // CHECK: %[[ARG_TENSOR:.+]] = stream.tensor.export %[[ARG_EXTERNAL]]
  // CHECK: %[[RET_TENSOR:.+]] = "some.op"(%[[ARG_TENSOR]]) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %0 = "some.op"(%arg0) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  // CHECK: %[[RET_SIZE:.+]] = stream.tensor.sizeof tensor<8x4xf32>
  // CHECK: %[[RET_EXTERNAL:.+]] = stream.tensor.import %[[RET_TENSOR]]
  // CHECK: %[[RET:.+]] = stream.async.transfer %[[RET_EXTERNAL]]
  // CHECK: return %[[RET]], %[[RET_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<8x4xf32>
}

// -----

// This is the while test, which exercises flow control and readbacks but is
// still simple enough to reason about: tests/e2e/tosa_ops/while.mlir
// This test is also nice because it contains a check test op, which requires
// stream/tensor interop.

// CHECK: stream.executable private @while_test_dispatch_0
flow.executable private @while_test_dispatch_0 {
  // CHECK: stream.executable.export public @dispatch
  flow.executable.export public @dispatch
  // CHECK: builtin.module
  builtin.module {
    // CHECK: func.func @dispatch(%[[BINDING0:.+]]: !stream.binding, %[[BINDING1:.+]]: !stream.binding)
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readonly:tensor<i32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<i1>>) {
      %c3_i32 = arith.constant 3 : i32
      // CHECK: %[[ARG0:.+]] = stream.binding.subspan %[[BINDING0]][%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<i32>>
      // CHECK: %[[ARG1:.+]] = stream.binding.subspan %[[BINDING1]][%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<i1>>
      // CHECK: = flow.dispatch.tensor.load %[[ARG0]], offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
      %0 = flow.dispatch.tensor.load %arg0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
      %1 = tensor.empty() : tensor<i1>
      // CHECK: linalg.generic
      %2 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%0 : tensor<i32>) outs(%1 : tensor<i1>) {
      ^bb0(%arg2: i32, %arg3: i1):
        %3 = arith.cmpi sge, %c3_i32, %arg2 : i32
        linalg.yield %3 : i1
      } -> tensor<i1>
      // CHECK: flow.dispatch.tensor.store %{{.+}}, %[[ARG1]], offsets = [], sizes = [], strides = [] : tensor<i1> -> !flow.dispatch.tensor<writeonly:tensor<i1>>
      flow.dispatch.tensor.store %2, %arg1, offsets = [], sizes = [], strides = [] : tensor<i1> -> !flow.dispatch.tensor<writeonly:tensor<i1>>
      return
    }
  }
}

// CHECK: stream.executable private @while_test_dispatch_1
flow.executable private @while_test_dispatch_1 {
  flow.executable.export public @dispatch
  builtin.module  {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readonly:tensor<i32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<i32>>) {
      %c2_i32 = arith.constant 2 : i32
      %0 = flow.dispatch.tensor.load %arg0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
      %1 = tensor.empty() : tensor<i32>
      %2 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%0 : tensor<i32>) outs(%1 : tensor<i32>) {
      ^bb0(%arg2: i32, %arg3: i32):
        %3 = arith.addi %arg2, %c2_i32 : i32
        linalg.yield %3 : i32
      } -> tensor<i32>
      flow.dispatch.tensor.store %2, %arg1, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:tensor<i32>>
      return
    }
  }
}

// CHECK-LABEL: func.func @while_test
func.func @while_test() {
  %c1 = arith.constant 1 : index

  // CHECK: %[[CONSTANT:.+]] = stream.tensor.constant : tensor<i32> in !stream.resource<constant> = dense<4> : tensor<i32>
  // CHECK: %[[CONSTANT_SIZE:.+]] = stream.resource.size %[[CONSTANT]] : !stream.resource<constant>
  // CHECK: %[[INITIAL:.+]] = stream.async.transfer %[[CONSTANT]] : !stream.resource<constant>{%[[CONSTANT_SIZE]]} -> !stream.resource<*>{%[[CONSTANT_SIZE]]}
  %cst = arith.constant dense<4> : tensor<i32>
  // CHECK: %[[INITIAL_DNO:.+]] = util.optimization_barrier %[[INITIAL]] : !stream.resource<*>
  %0 = util.optimization_barrier %cst : tensor<i32>

  // CHECK: %[[VAR_SIZE:.+]] = stream.resource.size %[[INITIAL_DNO]] : !stream.resource<*>
  // CHECK: cf.br ^bb1(%[[INITIAL_DNO]], %[[VAR_SIZE]] : !stream.resource<*>, index)
  cf.br ^bb1(%0 : tensor<i32>)

// CHECK: ^bb1(%[[BB1_ARG:.+]]: !stream.resource<*>, %[[BB1_ARG_SIZE:.+]]: index):
^bb1(%1: tensor<i32>):
  // CHECK: %[[COND_SIZE:.+]] = stream.tensor.sizeof tensor<i1> : index
  // CHECK: %[[COND_RESOURCE:.+]] = stream.async.dispatch @while_test_dispatch_0::@dispatch[%c1, %c1, %c1](%[[BB1_ARG]][%c0{{[_0-9]*}} to %[[BB1_ARG_SIZE]] for %[[BB1_ARG_SIZE]]]) : (!stream.resource<*>{%[[BB1_ARG_SIZE]]}) -> !stream.resource<*>{%[[COND_SIZE]]}
  %2 = flow.dispatch @while_test_dispatch_0::@dispatch[%c1, %c1, %c1](%1) : (tensor<i32>) -> tensor<i1>

  // CHECK: %[[READBACK:.+]] = stream.async.transfer %[[COND_RESOURCE]] : !stream.resource<*>{%[[COND_SIZE]]} -> !stream.resource<staging>{%[[COND_SIZE]]}
  // CHECK: %[[COND:.+]] = stream.tensor.load %[[READBACK]] : tensor<i1> in !stream.resource<staging>{%[[COND_SIZE]]} -> i1
  %3 = flow.tensor.load %2 : tensor<i1>

  // CHECK: cf.cond_br %[[COND]], ^bb2, ^bb3
  cf.cond_br %3, ^bb2, ^bb3

// CHECK: ^bb2:
^bb2:
  // CHECK: %[[BB2_VAR_SIZE:.+]] = stream.tensor.sizeof tensor<i32> : index
  // CHECK: %[[BB2_VAR:.+]] = stream.async.dispatch @while_test_dispatch_1::@dispatch[%c1, %c1, %c1](%[[BB1_ARG]][%c0{{[_0-9]*}} to %[[BB1_ARG_SIZE]] for %[[BB1_ARG_SIZE]]]) : (!stream.resource<*>{%[[BB1_ARG_SIZE]]}) -> !stream.resource<*>{%[[BB2_VAR_SIZE]]}
  %4 = flow.dispatch @while_test_dispatch_1::@dispatch[%c1, %c1, %c1](%1) : (tensor<i32>) -> tensor<i32>

  // CHECK: cf.br ^bb1(%[[BB2_VAR]], %[[BB2_VAR_SIZE]] : !stream.resource<*>, index)
  cf.br ^bb1(%4 : tensor<i32>)

// CHECK: ^bb3:
^bb3:
  // CHECK: %[[EXTERNAL_RESULT:.+]] = stream.async.transfer %[[BB1_ARG]] : !stream.resource<*>{%[[BB1_ARG_SIZE]]} -> !stream.resource<external>{%[[BB1_ARG_SIZE]]}
  // CHECK: %[[TENSOR_RESULT:.+]] = stream.tensor.export %[[EXTERNAL_RESULT]] : tensor<i32> in !stream.resource<external>{%[[BB1_ARG_SIZE]]} -> tensor<i32>
  // CHECK: %[[EXTERNAL_CONSTANT:.+]] = stream.async.transfer %[[INITIAL]] : !stream.resource<*>{%[[CONSTANT_SIZE]]} -> !stream.resource<external>{%[[CONSTANT_SIZE]]}
  // CHECK: %[[TENSOR_CONSTANT:.+]] = stream.tensor.export %[[EXTERNAL_CONSTANT]] : tensor<i32> in !stream.resource<external>{%[[CONSTANT_SIZE]]} -> tensor<i32>
  // CHECK: check.expect_eq(%[[TENSOR_RESULT]], %[[TENSOR_CONSTANT]]) : tensor<i32>
  check.expect_eq(%1, %cst) : tensor<i32>
  return
}

// -----

// Tests that generic ops that may rely on unrealized_conversion_cast reroute
// correctly to the original values.

// CHECK-LABEL: unrealizedCastCleanup
// CHECK-SAME: (%[[COND:.+]]: i1, %[[LHS:.+]]: !stream.resource<*>, %[[LHS_SIZE:.+]]: index, %[[RHS:.+]]: !stream.resource<*>, %[[RHS_SIZE:.+]]: index) -> (!stream.resource<*>, index)
func.func @unrealizedCastCleanup(%cond: i1, %lhs: tensor<1024xf32>, %rhs: tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK-DAG: %[[RET:.+]] = arith.select %[[COND]], %[[LHS]], %[[RHS]] : !stream.resource<*>
  // CHECK-DAG: %[[RET_SIZE:.+]] = arith.select %[[COND]], %[[LHS_SIZE]], %[[RHS_SIZE]] : index
  %0 = arith.select %cond, %lhs, %rhs : tensor<1024xf32>
  // CHECK: return %[[RET]], %[[RET_SIZE]]
  return %0 : tensor<1024xf32>
}
