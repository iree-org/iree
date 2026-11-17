// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-stablehlo-legalize-deprecated-ops %s | FileCheck %s --check-prefix=NORMALIZE

// unary_einsum becomes a binary einsum with a scalar one, then a transpose.
// NORMALIZE-LABEL: @unary_einsum_transpose
// NORMALIZE: stablehlo.einsum
// NORMALIZE-SAME: config = ",ab->ba"
// CHECK-LABEL: func.func @unary_einsum_transpose
// CHECK-NOT:     einsum
// CHECK:         linalg.generic {{.*}} ins(%{{.+}} : tensor<3x4xf32>) outs(%{{.+}} : tensor<4x3xf32>)
// CHECK:         linalg.yield
func.func @unary_einsum_transpose(%arg0: tensor<3x4xf32>) -> tensor<4x3xf32> {
  %0 = "stablehlo.unary_einsum"(%arg0) {einsum_config = "ab->ba"} : (tensor<3x4xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// NORMALIZE-LABEL: @unary_einsum_reduce
// NORMALIZE: stablehlo.einsum
// NORMALIZE-SAME: config = ",ab->a"
// CHECK-LABEL: func.func @unary_einsum_reduce
// CHECK-NOT:     einsum
// CHECK:         linalg.generic {{.*}}iterator_types = ["parallel", "reduction"]{{.*}}outs(%{{.+}} : tensor<3xf32>)
func.func @unary_einsum_reduce(%arg0: tensor<3x4xf32>) -> tensor<3xf32> {
  %0 = "stablehlo.unary_einsum"(%arg0) {einsum_config = "ab->a"} : (tensor<3x4xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// cross-replica-sum becomes an all_reduce over the default channel.
// NORMALIZE-LABEL: @cross_replica_sum
// NORMALIZE: stablehlo.all_reduce
// NORMALIZE: stablehlo.add
// CHECK-LABEL: func.func @cross_replica_sum
// CHECK:         %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
// CHECK:         flow.collective.all_reduce sum, f32, %{{.+}}, %{{.+}}, %[[CHANNEL]]
func.func @cross_replica_sum(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.cross-replica-sum"(%arg0) {replica_groups = dense<[[0]]> : tensor<1x1xi64>} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// NORMALIZE-LABEL: @broadcast
// NORMALIZE: stablehlo.broadcast_in_dim
// NORMALIZE-SAME: dims = [2]
// CHECK-LABEL: @broadcast
// CHECK: return %{{.+}} : tensor<2x3x4xf32>
func.func @broadcast(%input: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %r = stablehlo.broadcast %input, sizes = [2, 3] : (tensor<4xf32>) -> tensor<2x3x4xf32>
  return %r : tensor<2x3x4xf32>
}

// -----

// NORMALIZE-LABEL: @dot
// NORMALIZE: stablehlo.dot_general
// NORMALIZE-SAME: contracting_dims = [1] x [0]
// CHECK-LABEL: @dot
// CHECK: linalg.matmul
func.func @dot(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %r = stablehlo.dot %lhs, %rhs : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// The upstream default rejects map and rng. IREE must keep both for its own
// supported conversions.
// NORMALIZE-LABEL: @map
// NORMALIZE: stablehlo.map
// CHECK-LABEL: @map
// CHECK: linalg.generic
func.func @map(%input: tensor<4xf32>) -> tensor<4xf32> {
  %r = "stablehlo.map"(%input) ({
  ^bb0(%element: tensor<f32>):
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %sum = stablehlo.add %element, %one : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// NORMALIZE-LABEL: @rng
// NORMALIZE: stablehlo.rng
// CHECK-LABEL: @rng
// CHECK: linalg.generic
func.func @rng(%lower: tensor<f32>, %upper: tensor<f32>) -> tensor<4xf32> {
  %shape = stablehlo.constant dense<4> : tensor<1xi64>
  %r = "stablehlo.rng"(%lower, %upper, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// NORMALIZE-LABEL: @unused_token
// NORMALIZE: stablehlo.after_all
// CHECK-LABEL: @unused_token
// CHECK-NOT: stablehlo
// CHECK: return
func.func @unused_token() {
  %token = stablehlo.create_token : !stablehlo.token
  return
}
