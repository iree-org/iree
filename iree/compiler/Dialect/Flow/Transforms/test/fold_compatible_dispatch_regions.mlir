// RUN: iree-opt -split-input-file -iree-flow-fold-compatible-dispatch-regions %s | IreeFileCheck %s

func @noFolding(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant 4 : index
  %0 = flow.dispatch.region[%cst : index](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
    %1 = mhlo.add %arg1, %arg1 : tensor<4xf32>
    flow.return %1 : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @noFolding
// CHECK-NEXT: %[[WORKLOAD0:.+]] = constant 4 : index
// CHECK-NEXT: %0 = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %1 = mhlo.add %arg1, %arg1 : tensor<4xf32>
// CHECK-NEXT:   flow.return %1 : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %0 : tensor<4xf32>

// -----

func @elementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant 4 : index
  %0 = flow.dispatch.region[%cst : index](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
    %1 = mhlo.add %arg1, %arg1 : tensor<4xf32>
    flow.return %1 : tensor<4xf32>
  }
  %2 = flow.dispatch.region[%cst : index](%arg2 = %arg0 : tensor<4xf32>, %arg3 = %0 : tensor<4xf32>) -> tensor<4xf32> {
    %3 = mhlo.subtract %arg3, %arg2 : tensor<4xf32>
    flow.return %3 : tensor<4xf32>
  }
  %4 = flow.dispatch.region[%cst : index](%arg4 = %arg0 : tensor<4xf32>, %arg5 = %2 : tensor<4xf32>) -> tensor<4xf32> {
    %5 = mhlo.multiply %arg4, %arg5 : tensor<4xf32>
    flow.return %5 : tensor<4xf32>
  }
  return %4 : tensor<4xf32>
}

// CHECK-LABEL: func @elementwiseOps
// CHECK: %[[WORKLOAD0:.+]] = constant 4
// CHECK: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %1 = mhlo.add %arg1, %arg1 : tensor<4xf32>
// CHECK-NEXT:   %2 = mhlo.subtract %1, %arg1 : tensor<4xf32>
// CHECK-NEXT:   %3 = mhlo.multiply %arg1, %2 : tensor<4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4xf32>
// CHECK-NEXT: }
// CHECK: return %[[R0]] : tensor<4xf32>

// -----

func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = constant 16 : index
  %0 = flow.dispatch.region[%cst : index](%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %3 = mhlo.add %arg1, %arg1 : tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  %cst_0 = constant 16 : index
  %1 = flow.dispatch.region[%cst_0 : index](%arg1 = %0 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %3 = "mhlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  %cst_1 = constant 16 : index
  %2 = flow.dispatch.region[%cst_1 : index](%arg1 = %1 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %3 = mhlo.multiply %arg1, %arg2 : tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func @interleavedDot
// CHECK-NEXT: %[[WORKLOAD0:.+]] = constant 16 : index
// CHECK-NEXT: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %3 = mhlo.add %arg1, %arg1 : tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[WORKLOAD1:.+]] = constant 16 : index
// CHECK-NEXT: %[[R1:.+]] = flow.dispatch.region[%[[WORKLOAD1]] : index](%arg1 = %[[R0]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %3 = "mhlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[WORKLOAD2:.+]] = constant 16 : index
// CHECK-NEXT: %[[R2:.+]] = flow.dispatch.region[%[[WORKLOAD2]] : index](%arg1 = %[[R1]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %3 = mhlo.multiply %arg1, %arg2 : tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[R2]] : tensor<4x4xf32>
