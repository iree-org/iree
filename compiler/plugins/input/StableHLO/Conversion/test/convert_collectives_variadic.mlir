// The variadic forms are not implemented; they must diagnose, not crash.
// RUN: not iree-opt --split-input-file \
// RUN:   --iree-stablehlo-input-transformation-pipeline -o /dev/null 2>&1 %s \
// RUN:   | FileCheck %s

// CHECK: failed to legalize operation 'stablehlo.all_reduce'
func.func @all_reduce_variadic(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = "stablehlo.all_reduce"(%arg0, %arg1) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%sum) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// CHECK: failed to legalize operation 'stablehlo.all_gather'
func.func @all_gather_variadic(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 0 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK: failed to legalize operation 'stablehlo.all_to_all'
func.func @all_to_all_variadic(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>)
    -> (tensor<2x8xf32>, tensor<2x8xf32>) {
  %0:2 = "stablehlo.all_to_all"(%arg0, %arg1) {
    split_dimension = 0 : i64,
    concat_dimension = 1 : i64,
    split_count = 2 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<2x8xf32>, tensor<2x8xf32>)
  return %0#0, %0#1 : tensor<2x8xf32>, tensor<2x8xf32>
}
