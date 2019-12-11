// RUN: iree-opt -split-input-file -iree-flow-fold-compatible-dispatch-regions %s | IreeFileCheck %s

func @noFolding(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
    %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
    flow.return %1 : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @noFolding
// CHECK-NEXT: %cst = constant dense<[4, 1, 1]> : vector<3xi32>
// CHECK-NEXT: %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
// CHECK-NEXT:   flow.return %1 : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %0 : tensor<4xf32>

// -----

func @elementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
    %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
    flow.return %1 : tensor<4xf32>
  }
  %2 = flow.dispatch.region[%cst : vector<3xi32>](%arg2 = %arg0 : tensor<4xf32>, %arg3 = %0 : tensor<4xf32>) -> tensor<4xf32> {
    %3 = xla_hlo.sub %arg3, %arg2 : tensor<4xf32>
    flow.return %3 : tensor<4xf32>
  }
  %4 = flow.dispatch.region[%cst : vector<3xi32>](%arg4 = %arg0 : tensor<4xf32>, %arg5 = %2 : tensor<4xf32>) -> tensor<4xf32> {
    %5 = xla_hlo.mul %arg4, %arg5 : tensor<4xf32>
    flow.return %5 : tensor<4xf32>
  }
  return %4 : tensor<4xf32>
}

// CHECK-LABEL: func @elementwiseOps
// CHECK-NEXT: %cst = constant dense<[4, 1, 1]>
// CHECK-NEXT: %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
// CHECK-NEXT:   %2 = xla_hlo.sub %1, %arg1 : tensor<4xf32>
// CHECK-NEXT:   %3 = xla_hlo.mul %arg1, %2 : tensor<4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %0 : tensor<4xf32>

// -----

func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = constant dense<[4, 4, 1]> : vector<3xi32>
  %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %3 = xla_hlo.add %arg1, %arg1 : tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  %cst_0 = constant dense<[4, 4, 1]> : vector<3xi32>
  %1 = flow.dispatch.region[%cst_0 : vector<3xi32>](%arg1 = %0 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %3 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  %cst_1 = constant dense<[4, 4, 1]> : vector<3xi32>
  %2 = flow.dispatch.region[%cst_1 : vector<3xi32>](%arg1 = %1 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %3 = xla_hlo.mul %arg1, %arg2 : tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func @interleavedDot
// CHECK-NEXT: %cst = constant dense<[4, 4, 1]> : vector<3xi32>
// CHECK-NEXT: %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %3 = xla_hlo.add %arg1, %arg1 : tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %cst_0 = constant dense<[4, 4, 1]> : vector<3xi32>
// CHECK-NEXT: %1 = flow.dispatch.region[%cst_0 : vector<3xi32>](%arg1 = %0 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %3 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %cst_1 = constant dense<[4, 4, 1]> : vector<3xi32>
// CHECK-NEXT: %2 = flow.dispatch.region[%cst_1 : vector<3xi32>](%arg1 = %1 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %3 = xla_hlo.mul %arg1, %arg2 : tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %2 : tensor<4x4xf32>
