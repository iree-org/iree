// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-flow-inject-dispatch-tracing))' %s | FileCheck %s

// CHECK-LABEL: util.func public @singleDispatch
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>)
util.func public @singleDispatch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = arith.constant 4 : index
  //      CHECK: flow.tensor.trace "ex::entry0 inputs" = [%[[ARG0]] : tensor<4xf32>]
  // CHECK-NEXT: %[[RET0:.+]] = flow.dispatch @ex::@entry0[%c4](%[[ARG0]]) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex::@entry0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: flow.tensor.trace "ex::entry0 outputs" = [%[[RET0]] : tensor<4xf32>]
  // CHECK-NEXT: util.return %[[RET0]]
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: util.func public @multiDispatch
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>)
util.func public @multiDispatch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = arith.constant 4 : index

  //      CHECK: flow.tensor.trace "ex::entry0 inputs" = [%[[ARG0]] : tensor<4xf32>]
  // CHECK-NEXT: %[[RET0:.+]] = flow.dispatch @ex::@entry0[%c4](%[[ARG0]]) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex::@entry0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: flow.tensor.trace "ex::entry0 outputs" = [%[[RET0]] : tensor<4xf32>]

  //      CHECK: flow.tensor.trace "ex::entry1 inputs" = [%[[RET0]] : tensor<4xf32>]
  // CHECK-NEXT: %[[RET1:.+]] = flow.dispatch @ex::@entry1[%c4](%[[RET0]]) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @ex::@entry1[%c4](%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: flow.tensor.trace "ex::entry1 outputs" = [%[[RET1]] : tensor<4xf32>]

  // CHECK: util.return %[[RET1]]
  util.return %1 : tensor<4xf32>
}

// -----

// CHECK: #[[$ENCODING:.+]] = #iree_encoding.testing<>
#encoding = #iree_encoding.testing<>
// CHECK-LABEL: util.func public @encodedDispatch
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[DIM:[a-zA-Z0-9_]+]]
util.func public @encodedDispatch(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %dim : index) -> tensor<?xf32, #encoding>, tensor<?xf32> {
  %encoded = flow.tensor.encode %arg0 : tensor<?xf32>{%dim} -> tensor<?xf32, #encoding>{%dim}
  // CHECK:      %[[ARG0_ENCODED:.+]] = flow.tensor.encode %[[ARG0]]{{.*}} -> tensor<?xf32, #[[$ENCODING]]>{%[[DIM]]}
  // CHECK:      %[[ARG0_ROW_MAJOR:.+]] = flow.tensor.encode %[[ARG0_ENCODED]]{{.*}} -> tensor<?xf32>{%[[DIM]]}
  // CHECK:      flow.tensor.trace "ex::entry0 inputs with {0} decoded to row major layout" =
  // CHECK-SAME:   [%[[ARG0_ROW_MAJOR]] : tensor<?xf32>{%[[DIM]]}, %[[ARG1]] : tensor<?xf32>{%[[DIM]]}]
  // CHECK:      %[[RET:.+]]:2 = flow.dispatch @ex::@entry0[%[[DIM]], %[[DIM]]](%[[ARG0_ENCODED]], %[[ARG1]])
  %0:2 = flow.dispatch @ex::@entry0[%dim, %dim](%encoded, %arg1) : (tensor<?xf32, #encoding>{%dim}, tensor<?xf32>{%dim}) -> (tensor<?xf32, #encoding>{%dim}, tensor<?xf32>{%dim})
  // CHECK:      %[[RET0_ROW_MAJOR:.+]] = flow.tensor.encode %[[RET]]#0{{.*}} -> tensor<?xf32>{%[[DIM]]}
  // CHECK:      flow.tensor.trace "ex::entry0 outputs with {0} decoded to row major layout" =
  // CHECK-SAME:   [%[[RET0_ROW_MAJOR]] : tensor<?xf32>{%[[DIM]]}, %[[RET]]#1 : tensor<?xf32>{%[[DIM]]}]
  // CHECK:      util.return %[[RET]]#0, %[[RET]]#1
  util.return %0#0, %0#1 : tensor<?xf32, #encoding>, tensor<?xf32>
}
