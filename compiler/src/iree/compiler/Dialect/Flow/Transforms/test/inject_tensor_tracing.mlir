// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-flow-inject-tensor-tracing))' --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: util.func public @traceTensorOp
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[ARG1:.+]]: tensor<4xf32>)
util.func public @traceTensorOp(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  //      CHECK: flow.tensor.trace "arith.addf inputs" = [%[[ARG0]] : tensor<4xf32>, %[[ARG1]] : tensor<4xf32>]
  // CHECK-NEXT: %[[RESULT:.+]] = arith.addf
  //  CHECK-NOT: iree.tensor.trace
  %result = arith.addf %arg0, %arg1 {iree.tensor.trace} : tensor<4xf32>
  // CHECK-NEXT: flow.tensor.trace "arith.addf outputs" = [%[[RESULT]] : tensor<4xf32>]
  util.return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: util.func public @traceDispatchRegion
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[ARG1:.+]]: tensor<4xi32>)
util.func public @traceDispatchRegion(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> tensor<4xf32> {
  //      CHECK: flow.tensor.trace "flow.dispatch.region inputs" = [%[[ARG0]] : tensor<4xf32>, %[[ARG1]] : tensor<4xi32>]
  // CHECK-NEXT: %[[RESULT:.+]] = flow.dispatch.region
  //  CHECK-NOT: iree.tensor.trace
  %result = flow.dispatch.region -> (tensor<4xf32>) attributes {iree.tensor.trace} {
    %0 = "some.op"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
    flow.return %0 : tensor<4xf32>
  }
  //      CHECK: flow.tensor.trace "flow.dispatch.region outputs" = [%[[RESULT]] : tensor<4xf32>]
  util.return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: util.func public @traceDispatch
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>, %[[ARG1:.+]]: tensor<?xi32>)
util.func public @traceDispatch(%arg0: tensor<?xf32>, %arg1: tensor<?xi32>) -> (tensor<?xf32>, tensor<?xi16>) {
  %c0 = arith.constant 0 : index
  //  CHECK-DAG: %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %c0
  %arg0_d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  //  CHECK-DAG: %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %c0
  %arg1_d0 = tensor.dim %arg1, %c0 : tensor<?xi32>
  //      CHECK: flow.tensor.trace "ex::entry0 inputs" = [%[[ARG0]] : tensor<?xf32>{%[[ARG0_D0]]}, %[[ARG1]] : tensor<?xi32>{%[[ARG1_D0]]}]
  // CHECK-NEXT: %[[RESULT:.+]]:2 = flow.dispatch @ex::@entry0
  //  CHECK-NOT: iree.tensor.trace
  %result:2 = flow.dispatch @ex::@entry0(%arg0, %arg1) {iree.tensor.trace} : (tensor<?xf32>{%arg0_d0}, tensor<?xi32>{%arg1_d0}) -> (%arg0 as tensor<?xf32>{%arg0_d0}, tensor<?xi16>{%arg1_d0})
  // CHECK-NEXT: flow.tensor.trace "ex::entry0 outputs" = [%[[RESULT]]#0 : tensor<?xf32>{%[[ARG0_D0]]}, %[[RESULT]]#1 : tensor<?xi16>{%[[ARG1_D0]]}]
  util.return %result#0, %result#1 : tensor<?xf32>, tensor<?xi16>
}

// -----

util.func private @callee(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> tensor<4xf32>

// CHECK-LABEL: util.func public @traceCall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[ARG1:.+]]: tensor<4xi32>)
util.func public @traceCall(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xf32>) {
  //      CHECK: flow.tensor.trace "callee inputs" = [%[[ARG0]] : tensor<4xf32>, %[[ARG1]] : tensor<4xi32>]
  // CHECK-NEXT: %[[RESULT0:.+]] = util.call @callee
  //  CHECK-NOT: iree.tensor.trace
  %result0 = util.call @callee(%arg0, %arg1) {iree.tensor.trace} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  // CHECK-NEXT: flow.tensor.trace "callee outputs" = [%[[RESULT0]] : tensor<4xf32>]
  //      CHECK: flow.tensor.trace "a key inputs" = [%[[ARG0]] : tensor<4xf32>, %[[ARG1]] : tensor<4xi32>]
  // CHECK-NEXT: %[[RESULT1:.+]] = util.call @callee
  //  CHECK-NOT: iree.tensor.trace
  %result1 = util.call @callee(%arg0, %arg1) {iree.tensor.trace = "a key"} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  // CHECK-NEXT: flow.tensor.trace "a key outputs" = [%[[RESULT1]] : tensor<4xf32>]
  util.return %result0, %result1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// CHECK-LABEL: util.func public @traceNested
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[ARG1:.+]]: tensor<4xf32>
util.func public @traceNested(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %cond: i1) -> tensor<4xf32> {
  // CHECK: scf.if
  %result = scf.if %cond -> tensor<4xf32> {
    // CHECK-NEXT: flow.tensor.trace "arith.addf inputs" = [%[[ARG0]] : tensor<4xf32>, %[[ARG1]] : tensor<4xf32>]
    // CHECK-NEXT: %[[RESULT:.+]] = arith.addf
    //  CHECK-NOT: iree.tensor.trace
    %0 = arith.addf %arg0, %arg1 {iree.tensor.trace} : tensor<4xf32>
    // CHECK-NEXT: flow.tensor.trace "arith.addf outputs" = [%[[RESULT]] : tensor<4xf32>]
    scf.yield %0 : tensor<4xf32>
  } else {
    scf.yield %arg0 : tensor<4xf32>
  }
  util.return %result : tensor<4xf32>
}
