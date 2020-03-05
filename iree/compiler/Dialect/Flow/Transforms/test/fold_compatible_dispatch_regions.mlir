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

// -----

func @independentReductions(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4xf32> {
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %workload = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.dispatch.region[%workload : vector<3xi32>](%arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %cst : tensor<f32>) -> tensor<4xf32> {
    %1 = "xla_hlo.reduce"(%arg2, %arg3) ( {
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):	// no predecessors
      %2 = xla_hlo.add %arg4, %arg5 : tensor<f32>
      "xla_hlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    flow.return %1 : tensor<4xf32>
  }
  %3 = flow.dispatch.region[%workload : vector<3xi32>](%arg2 = %arg1 : tensor<4x8xf32>, %arg3 = %cst : tensor<f32>) -> tensor<4xf32> {
    %4 = "xla_hlo.reduce"(%arg2, %arg3) ( {
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):	// no predecessors
      %5 = xla_hlo.add %arg4, %arg5 : tensor<f32>
      "xla_hlo.return"(%5) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    flow.return %4 : tensor<4xf32>
  }
  return %3 : tensor<4xf32>
}

// CHECK-LABEL: func @independentReductions
//   CHECK-DAG:   [[INITIAL:%.+]] = constant dense<0.000000e+00> : tensor<f32>
//   CHECK-DAG:   [[WORKLOAD:%.+]] = constant dense<[4, 1, 1]> : vector<3xi32>
//  CHECK-NEXT:   [[RESULT:%.+]] = flow.dispatch.region[
//  CHECK-SAME:       [[WORKLOAD]] : vector<3xi32>
//  CHECK-SAME:     ](%arg2 = %arg0 : tensor<4x8xf32>, %arg3 = [[INITIAL]] : tensor<f32>, %arg4 = %arg1 : tensor<4x8xf32>) -> tensor<4xf32> {
//  CHECK-NEXT:     %1 = "xla_hlo.reduce"(%arg2, %arg3) ( {
//  CHECK-NEXT:     ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):	// no predecessors
//  CHECK-NEXT:       %3 = xla_hlo.add %arg5, %arg6 : tensor<f32>
//  CHECK-NEXT:       "xla_hlo.return"(%3) : (tensor<f32>) -> ()
//  CHECK-NEXT:     }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
//  CHECK-NEXT:     %2 = "xla_hlo.reduce"(%arg4, %arg3) ( {
//  CHECK-NEXT:     ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):	// no predecessors
//  CHECK-NEXT:       %3 = xla_hlo.add %arg5, %arg6 : tensor<f32>
//  CHECK-NEXT:       "xla_hlo.return"(%3) : (tensor<f32>) -> ()
//  CHECK-NEXT:     }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
//  CHECK-NEXT:     flow.return %2 : tensor<4xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return [[RESULT]] : tensor<4xf32>
//  CHECK-NEXT: }

// -----

func @interleavedReduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %workload_0 = constant dense<[4, 8, 1]> : vector<3xi32>
  %0 = flow.dispatch.region[%workload_0 : vector<3xi32>](%arg1 = %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
    %1 = xla_hlo.add %arg1, %arg1 : tensor<4x8xf32>
    flow.return %1 : tensor<4x8xf32>
  }
  %workload_1 = constant dense<[4, 1, 1]> : vector<3xi32>
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %2 = flow.dispatch.region[%workload_1 : vector<3xi32>](%arg1 = %0 : tensor<4x8xf32>, %arg2 = %cst : tensor<f32>) -> tensor<4xf32> {
    %3 = "xla_hlo.reduce"(%arg1, %arg2) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):	// no predecessors
      %4 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%4) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    flow.return %3 : tensor<4xf32>
  }
  %4 = flow.dispatch.region[%workload_1 : vector<3xi32>](%arg1 = %2 : tensor<4xf32>) -> tensor<4xf32> {
    %5 = xla_hlo.mul %arg1, %arg1 : tensor<4xf32>
    flow.return %5 : tensor<4xf32>
  }
  return %4 : tensor<4xf32>
}

// CHECK-LABEL: func @interleavedReduction
//  CHECK-NEXT:   [[WORKLOAD_0:%.+]] = constant dense<[4, 8, 1]> : vector<3xi32>
//  CHECK-NEXT:   [[UNFUSED_RESULT:%.+]] = flow.dispatch.region[
//  CHECK-SAME:       [[WORKLOAD_0]] : vector<3xi32>
//  CHECK-SAME:     ](%arg1 = %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
//  CHECK-NEXT:     %2 = xla_hlo.add %arg1, %arg1 : tensor<4x8xf32>
//  CHECK-NEXT:     flow.return %2 : tensor<4x8xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   [[WORKLOAD_1:%.+]] = constant dense<[4, 1, 1]> : vector<3xi32>
//  CHECK-NEXT:   [[INITIAL:%.+]] = constant dense<0.000000e+00> : tensor<f32>
//  CHECK-NEXT:   [[FUSED_RESULT:%.+]] = flow.dispatch.region[
//  CHECK-SAME:       [[WORKLOAD_1]] : vector<3xi32>
//  CHECK-SAME:     ](%arg1 = [[UNFUSED_RESULT]] : tensor<4x8xf32>, %arg2 = [[INITIAL]] : tensor<f32>) -> tensor<4xf32> {
//  CHECK-NEXT:     %2 = "xla_hlo.reduce"(%arg1, %arg2) ( {
//  CHECK-NEXT:     ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):	// no predecessors
//  CHECK-NEXT:       %4 = xla_hlo.add %arg3, %arg4 : tensor<f32>
//  CHECK-NEXT:       "xla_hlo.return"(%4) : (tensor<f32>) -> ()
//  CHECK-NEXT:     }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
//  CHECK-NEXT:     %3 = xla_hlo.mul %2, %2 : tensor<4xf32>
//  CHECK-NEXT:     flow.return %3 : tensor<4xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return [[FUSED_RESULT]] : tensor<4xf32>
//  CHECK-NEXT: }
