// RUN: iree-opt -split-input-file -iree-flow-rematerialize-dispatch-constants %s | IreeFileCheck %s

// CHECK-LABEL: func @rematerializeSmall
func @rematerializeSmall(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: %[[WORKLOAD0:.+]] = constant 16 : index
  %cst = constant 16 : index
  %small = constant dense<1.23> : tensor<4x4xf32>
  // CHECK: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = flow.dispatch.region[%cst : index](%arg1 = %arg0 : tensor<4x4xf32>, %arg2 = %small : tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-NEXT: %[[REMAT_SMALL:.+]] = constant dense<1.230000e+00> : tensor<4x4xf32>
    // CHECK-NEXT: %1 = xla_hlo.add %arg1, %[[REMAT_SMALL]] : tensor<4x4xf32>
    %3 = xla_hlo.add %arg1, %arg2 : tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @noRematerializeLarge
func @noRematerializeLarge(%arg0 : tensor<4096x4xf32>) -> tensor<4096x4xf32> {
  // CHECK-DAG: %[[WORKLOAD0:.+]] = constant 16 : index
  // CHECK-DAG: %[[CST:.+]] = constant dense<1.230000e+00> : tensor<4096x4xf32>
  %cst = constant 16 : index
  %large = constant dense<1.23> : tensor<4096x4xf32>
  // CHECK-NEXT: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4096x4xf32>, %arg2 = %[[CST]] : tensor<4096x4xf32>) -> tensor<4096x4xf32> {
  %0 = flow.dispatch.region[%cst : index](%arg1 = %arg0 : tensor<4096x4xf32>, %arg2 = %large : tensor<4096x4xf32>) -> tensor<4096x4xf32> {
    // CHECK-NEXT: %1 = xla_hlo.add %arg1, %arg2 : tensor<4096x4xf32>
    %3 = xla_hlo.add %arg1, %arg2 : tensor<4096x4xf32>
    flow.return %3 : tensor<4096x4xf32>
  }
  return %0 : tensor<4096x4xf32>
}

// -----

// CHECK-LABEL: func @noRematerializeIntoDot
func @noRematerializeIntoDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-DAG: %[[WORKLOAD0:.+]] = constant 16 : index
  // CHECK-DAG: %[[SMALL:.+]] = constant dense<1.230000e+00> : tensor<4x4xf32>
  %cst = constant 16 : index
  %small = constant dense<1.23> : tensor<4x4xf32>
  // CHECK-NEXT: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4x4xf32>, %arg2 = %[[SMALL]] : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = flow.dispatch.region[%cst : index](%arg1 = %arg0 : tensor<4x4xf32>, %arg2 = %small : tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-NEXT: %1 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}
