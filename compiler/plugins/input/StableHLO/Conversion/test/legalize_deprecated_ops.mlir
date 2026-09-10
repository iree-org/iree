// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// unary_einsum is rewritten by this pass to einsum, but einsum-to-dot_general
// does not yet handle the resulting degenerate contraction, so it is left out.
// CHECK-LABEL: @cross_replica_sum
// CHECK: flow.collective.all_reduce sum, f32
func.func @cross_replica_sum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.cross-replica-sum"(%arg0) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
