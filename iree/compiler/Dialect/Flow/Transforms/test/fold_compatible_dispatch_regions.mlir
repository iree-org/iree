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
// CHECK-NEXT: %0 = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>) {
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
// CHECK: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>) {
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
// CHECK-NEXT: %[[R0:.+]] = flow.dispatch.region[%[[WORKLOAD0]] : index](%arg1 = %arg0 : tensor<4x4xf32>) -> (tensor<4x4xf32>) {
// CHECK-NEXT:   %3 = mhlo.add %arg1, %arg1 : tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[WORKLOAD1:.+]] = constant 16 : index
// CHECK-NEXT: %[[R1:.+]] = flow.dispatch.region[%[[WORKLOAD1]] : index](%arg1 = %[[R0]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> (tensor<4x4xf32>) {
// CHECK-NEXT:   %3 = "mhlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[WORKLOAD2:.+]] = constant 16 : index
// CHECK-NEXT: %[[R2:.+]] = flow.dispatch.region[%[[WORKLOAD2]] : index](%arg1 = %[[R1]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> (tensor<4x4xf32>) {
// CHECK-NEXT:   %3 = mhlo.multiply %arg1, %arg2 : tensor<4x4xf32>
// CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[R2]] : tensor<4x4xf32>

// -----

module {
  flow.variable @var1 dense<1.000000e+00> : tensor<4xf32>
  flow.variable @var2 dense<2.000000e+00> : tensor<4xf32>
  func @notDominate() -> tensor<4xf32> {
    %c4 = constant 4 : index
    %0 = flow.variable.load @var1 : tensor<4xf32>
    %1 = flow.dispatch.region[%c4 : index](%arg0 = %0 : tensor<4xf32>) -> tensor<4xf32> {
      %4 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      flow.return %4 : tensor<4xf32>
    }
    %2 = flow.variable.load @var2 : tensor<4xf32>
    %3 = flow.dispatch.region[%c4 : index](%arg0 = %0 : tensor<4xf32>, %arg1 = %2 : tensor<4xf32>) -> tensor<4xf32> {
      %4 = mhlo.subtract %arg1, %arg0 : tensor<4xf32>
      flow.return %4 : tensor<4xf32>
    }
    return %3 : tensor<4xf32>
  }
}
// CHECK-LABEL: func @notDominate
//       CHECK: flow.dispatch.region
//       CHECK: flow.dispatch.region

// -----

module {
  flow.variable @var1 dense<1.000000e+00> : tensor<4xf32>
  flow.variable @var2 dense<2.000000e+00> : tensor<4xf32>
  func @dominate() -> tensor<4xf32> {
    %c4 = constant 4 : index
    %0 = flow.variable.load @var1 : tensor<4xf32>
    %1 = flow.variable.load @var2 : tensor<4xf32>
    %2 = flow.dispatch.region[%c4 : index](%arg0 = %0 : tensor<4xf32>) -> tensor<4xf32> {
      %4 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      flow.return %4 : tensor<4xf32>
    }
    %3 = flow.dispatch.region[%c4 : index](%arg0 = %0 : tensor<4xf32>, %arg1 = %1 : tensor<4xf32>) -> tensor<4xf32> {
      %4 = mhlo.subtract %arg1, %arg0 : tensor<4xf32>
      flow.return %4 : tensor<4xf32>
    }
    return %3 : tensor<4xf32>
  }
}
// CHECK-LABEL: func @dominate
//       CHECK: flow.dispatch.region
//   CHECK-NOT: flow.dispatch.region

// -----

module {
  func @torch_index_select_producer(%arg0: tensor<5x1x5xi32>,
                                    %arg1: tensor<2xi32>) -> tensor<2x1x5xi32> {
    %c10 = constant 0 : index
    %0 = flow.dispatch.region[%c10 : index](%arg2 = %arg0 : tensor<5x1x5xi32>,
                                            %arg3 = %arg1 : tensor<2xi32>) -> tensor<2x1x5xi32> {
      %1 = "mhlo.torch_index_select"(%arg2, %arg3) {
        dim = 0 : i64,
        batch_dims = 0 : i64
      } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
      flow.return %1 : tensor<2x1x5xi32>
    }
    %1 = flow.dispatch.region[%c10 : index](%arg2 = %0 : tensor<2x1x5xi32>) -> tensor<2x1x5xi32> {
      %2 = mhlo.add %arg2, %arg2 : tensor<2x1x5xi32>
      flow.return %2 : tensor<2x1x5xi32>
    }
    return %1 : tensor<2x1x5xi32>
  }
}
// CHECK-LABEL: func @torch_index_select_producer
//       CHECK: flow.dispatch.region
//  CHECK-NEXT:   mhlo.torch_index_select
//  CHECK-NEXT:   mhlo.add
