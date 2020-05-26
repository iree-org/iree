// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @dceOperandsAndResults
func @dceOperandsAndResults(%arg0 : tensor<?xf32>) -> (tensor<?xf32>) {
  // CHECK: %[[WORKLOAD:.+]] = constant 5
  %workload = constant 5 : index
  // CHECK: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD]] : index]
  // CHECK-SAME: (%[[CA1:.+]] = %arg0 : tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[DR0:.+]] = addf %[[CA1]], %[[CA1]]
  // CHECK: flow.return %[[DR0]] : tensor<?xf32>
  %ret0, %ret1 = flow.dispatch.region[%workload : index](
      %i0 = %arg0 : tensor<?xf32>, %i1 = %arg0 : tensor<?xf32>, %i2 = %arg0 : tensor<?xf32>) 
      -> (tensor<?xf32>, tensor<?xf32>) {
    %1 = addf %i0, %i1 : tensor<?xf32>
    flow.return %1, %i2 : tensor<?xf32>, tensor<?xf32>
  }
  // CHECK: return %[[R0]] : tensor<?xf32>
  return %ret0 : tensor<?xf32>
}
