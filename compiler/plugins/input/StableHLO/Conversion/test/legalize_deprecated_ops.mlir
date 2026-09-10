// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// CHECK-LABEL: @cross_replica_sum
// CHECK: flow.collective.all_reduce sum, f32
func.func @cross_replica_sum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.cross-replica-sum"(%arg0) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @unary_einsum
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<4xf32>
func.func @unary_einsum(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.unary_einsum"(%arg0) {
    einsum_config = "ab->a"
  } : (tensor<4x8xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
