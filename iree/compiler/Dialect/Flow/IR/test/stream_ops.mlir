// Tests printing and parsing of stream ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

flow.executable @dispatch_0 {
  flow.dispatch.entry @rgn_dispatch_0
  module {
    func @rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: func @fragment
func @fragment(%arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK: %[[WORKLOAD:.+]] = constant
  %cst = constant 4 : index
  // CHECK: %0:2 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = flow.ex.stream.fragment(%arg1 = %cst : index, %arg2 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // CHECK-NEXT: flow.dispatch
    %1 = flow.dispatch @dispatch_0::@rgn_dispatch_0[%arg1 : index](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: flow.return
    flow.return %1, %1 : tensor<4xf32>, tensor<4xf32>
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: return
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
