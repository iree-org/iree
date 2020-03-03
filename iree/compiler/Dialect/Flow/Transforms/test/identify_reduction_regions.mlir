// RUN: iree-opt -split-input-file -iree-flow-identify-reduction-regions %s | IreeFileCheck %s

// CHECK-LABEL: @single_reduction
func @single_reduction(%arg0 : tensor<4x8xf32>) -> tensor<4xf32> {
  // CHECK-DAG: [[INITIAL:%.+]] = constant dense<0.000000e+00>
  %0 = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: constant dense<[4, 1, 1]>
  // CHECK-NEXT: [[RESULT:%.+]] = flow.reduction.region
  // CHECK-SAME: [%cst_0 : vector<3xi32>]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4x8xf32>, %arg2 = [[INITIAL]] : tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: = "xla_hlo.reduce"(%arg1, %arg2)
  %1 = "xla_hlo.reduce"(%arg0, %0) ( {
  // CHECK: invocation((%arg1, %arg2) : tensor<f32>) -> tensor<f32> {
  ^bb0(%arg1 : tensor<f32>, %arg2 : tensor<f32>):
    // CHECK-NEXT: %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    %2 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    // CHECK-NEXT: flow.return %1 : tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  // CHECK-NEXT: } {dimensions = dense<1> : vector<1xi32>}
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: return [[RESULT]] : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @multi_reduction
func @multi_reduction(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-DAG: [[INITIALA:%.+]] = constant dense<0.000000e+00>
  %0 = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: [[INITIALB:%.+]] = constant dense<1.000000e+00>
  %1 = constant dense<1.000000e+00> : tensor<f32>
  // CHECK: constant dense<[4, 1, 1]>
  // CHECK-NEXT: [[RESULT:%.+]]:2 = flow.reduction.region
  // CHECK-SAME: [%cst_1 : vector<3xi32>]
  // CHECK-SAME: (%arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %arg1 : tensor<4x8xf32>, %arg4 = [[INITIALA]] : tensor<f32>, %arg5 = [[INITIALB]] : tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK-NEXT: = "xla_hlo.reduce"(%arg2, %arg3, %arg4, %arg5)
  %2, %3 = "xla_hlo.reduce"(%arg0, %arg1, %0, %1) ( {
  // CHECK: invocation((%arg2, %arg3) : tensor<f32>, (%arg4, %arg5) : tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  ^bb0(%arg0_lhs : tensor<f32>, %arg1_lhs : tensor<f32>, %arg0_rhs : tensor<f32>, %arg1_rhs : tensor<f32>):
    // CHECK-NEXT: %1 = xla_hlo.add %arg2, %arg4 : tensor<f32>
    %4 = xla_hlo.add %arg0_lhs, %arg0_rhs : tensor<f32>
    // CHECK-NEXT: %2 = xla_hlo.add %arg3, %arg5 : tensor<f32>
    %5 = xla_hlo.add %arg1_lhs, %arg1_rhs : tensor<f32>
    // CHECK-NEXT: flow.return %1, %2 : tensor<f32>, tensor<f32>
    "xla_hlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK-NEXT: } {dimensions = dense<1> : vector<1xi32>}
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK: return [[RESULT]]#0, [[RESULT]]#1 : tensor<4xf32>, tensor<4xf32>
  return %2, %3 : tensor<4xf32>, tensor<4xf32>
}

// -----

// TODO(benvanik): windowed reduction.
