// Tests folding and canonicalization of stream ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

flow.executable @dispatch_0 {
  flow.dispatch.entry @rgn_dispatch_0
  module {
    func @rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: func @fragmentDedupCaptures
// CHECK-SAME: %[[A0:.+]]: tensor<4xf32>
func @fragmentDedupCaptures(%arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK: %[[CST:.+]] = constant 4
  %cst = constant 4 : index
  // Should dedup %cst in arg list.
  // CHECK: flow.ex.stream.fragment(%arg1 = %[[CST]] : index, %arg2 = %[[A0]] : tensor<4xf32>)
  %0:2 = flow.ex.stream.fragment(%arg1 = %cst : index, %arg2 = %cst : index, %arg3 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // Both referreants of the constant should use the deduped arg.
    // CHECK: flow.dispatch @dispatch_0::@rgn_dispatch_0[%arg1 : index]
    // CHECK: flow.dispatch @dispatch_0::@rgn_dispatch_0[%arg1 : index]
    %1 = flow.dispatch @dispatch_0::@rgn_dispatch_0[%arg1 : index](%arg3) : (tensor<4xf32>) -> tensor<4xf32>
    %2 = flow.dispatch @dispatch_0::@rgn_dispatch_0[%arg2 : index](%1) : (tensor<4xf32>) -> tensor<4xf32>
    flow.return %2, %2 : tensor<4xf32>, tensor<4xf32>
  }
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}

// -----
// CHECK-LABEL: func @removeUnusedCapture
func @removeUnusedCapture() -> (index) {
  // CHECK: %[[CST:.+]] = constant 4
  %cst = constant 4 : index
  %unused = constant 5 : index
  // CHECK: flow.ex.stream.fragment(%arg0 = %[[CST]] : index)
  %0 = flow.ex.stream.fragment(%arg0 = %cst : index, %arg1 = %unused : index) -> (index) {
    flow.return %arg0 : index
  }
  return %0 : index
}

// -----
// CHECK-LABEL: func @removeUnusedDupCapture
func @removeUnusedDupCapture() -> (index) {
  // CHECK: %[[CST:.+]] = constant 4
  %cst = constant 4 : index
  // CHECK: flow.ex.stream.fragment(%arg0 = %[[CST]] : index)
  %0 = flow.ex.stream.fragment(%arg0 = %cst : index, %arg1 = %cst : index) -> (index) {
    flow.return %arg1 : index
  }
  return %0 : index
}

// -----
// CHECK-LABEL: func @removeUnusedResult
func @removeUnusedResult() -> (index) {
  // CHECK: %[[CST:.+]] = constant 4
  %cst = constant 4 : index
  // Note that the unused result should also cascade to elide the newly
  // unused operand.
  // CHECK: flow.ex.stream.fragment(%arg0 = %[[CST]] : index)
  // CHECK-SAME: -> index
  %0:2 = flow.ex.stream.fragment(%arg0 = %cst : index, %arg1 = %cst : index) -> (index, index) {
    flow.return %arg1, %arg0 : index, index
  }
  return %0 : index
}
