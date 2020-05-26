// RUN: iree-opt -split-input-file -iree-flow-dispatchability-analysis -iree-flow-identify-dispatch-regions2 %s | IreeFileCheck %s

// CHECK-LABEL: @empty
func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @stdElementwiseOps
func @stdElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4
  // CHECK-NEXT: %[[R1:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = addf %arg1, %arg1 : tensor<4xf32>
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = subf %1, %arg1 : tensor<4xf32>
  %1 = subf %0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %3 = mulf %2, %arg1 : tensor<4xf32>
  %2 = mulf %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R1]] : tensor<4xf32>
  return %2 : tensor<4xf32>
}
