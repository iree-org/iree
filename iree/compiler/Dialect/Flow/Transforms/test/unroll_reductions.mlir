// RUN: iree-opt -split-input-file -iree-flow-unroll-reductions -cse %s | IreeFileCheck %s

// CHECK-LABEL: func @unrolled_reduction
func @unrolled_reduction(%arg0: tensor<4x2x8xf32>) -> tensor<4xf32> {
  // CHECK-DAG: [[INITIAL:%.+]] = constant dense<0.000000e+00> : tensor<f32>
  %cst = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: [[TEMP:%.+]] = "xla_hlo.reduce"(%arg0, [[INITIAL]]) ( {
  // CHECK-NEXT: ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %2 = xla_hlo.add %arg1, %arg2 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<4x2x8xf32>, tensor<f32>) -> tensor<4x2xf32>
  // CHECK-NEXT: [[RESULT:%.+]] = "xla_hlo.reduce"([[TEMP]], [[INITIAL]]) ( {
  // CHECK-NEXT: ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %2 = xla_hlo.add %arg1, %arg2 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4xf32>
  %0 = "xla_hlo.reduce"(%arg0, %cst) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>): // no predecessors
    %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x2x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return [[RESULT]]
  return %0 : tensor<4xf32>
}
