// RUN: iree-opt -split-input-file -iree-flow-identify-reduction-regions %s | IreeFileCheck %s

// CHECK-LABEL: @single_reduction
func @single_reduction(%arg0 : tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = constant dense<0.0> : tensor<f32>
  // CHECK: constant dense<[4, 1, 1]>
  // CHECK-NEXT: %0 = flow.reduction.region
  // CHECK-SAME: [%cst_0 : vector<3xi32>]
  // CHECK-SAME: (%arg0) : (tensor<4x8xf32>) -> tensor<4xf32>
  %1 = "xla_hlo.reduce"(%arg0, %0) ( {
  // CHECK-NEXT: invocation((%arg1, %arg2) = %cst : tensor<f32>) {
  ^bb0(%arg1 : tensor<f32>, %arg2 : tensor<f32>):
    // CHECK-NEXT: %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    %2 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    // CHECK-NEXT: flow.return %1 : tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  // CHECK-NEXT: } {dimensions = dense<1> : vector<1xi32>}
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @multi_reduction
func @multi_reduction(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = constant dense<0.0> : tensor<f32>
  %1 = constant dense<1.0> : tensor<f32>
  // CHECK: constant dense<[4, 1, 1]>
  // CHECK-NEXT: %0:2 = flow.reduction.region
  // CHECK-SAME: [%cst_1 : vector<3xi32>]
  // CHECK-SAME: (%arg0, %arg1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %2, %3 = "xla_hlo.reduce"(%arg0, %arg1, %0, %1) ( {
  // CHECK-NEXT: invocation((%arg2, %arg3) = %cst : tensor<f32>, (%arg4, %arg5) = %cst_0 : tensor<f32>) {
  ^bb0(%arg0_lhs : tensor<f32>, %arg1_lhs : tensor<f32>, %arg0_rhs : tensor<f32>, %arg1_rhs : tensor<f32>):
    // CHECK-NEXT: %1 = xla_hlo.add %arg2, %arg4 : tensor<f32>
    %4 = xla_hlo.add %arg0_lhs, %arg0_rhs : tensor<f32>
    // CHECK-NEXT: %2 = xla_hlo.add %arg3, %arg5 : tensor<f32>
    %5 = xla_hlo.add %arg1_lhs, %arg1_rhs : tensor<f32>
    // CHECK-NEXT: flow.return %1, %2 : tensor<f32>, tensor<f32>
    "xla_hlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK-NEXT: } {dimensions = dense<1> : vector<1xi32>}
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK-NEXT: return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  return %2, %3 : tensor<4xf32>, tensor<4xf32>
}

// -----

// TODO(benvanik): windowed reduction.
