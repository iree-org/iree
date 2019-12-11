// Tests printing and parsing of dispatch ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

flow.executable @ex0 {
  module {
    func @dispatch_fn(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
      return %arg0 : tensor<4xf32>
    }
  }
  flow.dispatch.entry @dispatch_fn
}

// CHECK-LABEL: @dispatch
func @dispatch(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant dense<1> : vector<3xi32>
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn[%cst : vector<3xi32>](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch_fn[%cst : vector<3xi32>](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
