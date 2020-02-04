// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-convert-hlo %s | IreeFileCheck %s

// CHECK-LABEL: @rankBroadcastAdd
// CHECK-SAME: %[[LHS:[^:[:space:]]+]]: tensor<?x16xf32>
// CHECK-SAME: %[[RHS:[^:[:space:]]+]]: tensor<16xf32>
func @rankBroadcastAdd(%arg0 : tensor<?x16xf32>, %arg1 : tensor<16xf32>) -> (tensor<?x16xf32>) {
  // CHECK-DAG: %[[RS_LHS:.+]] = shapex.get_ranked_shape %[[LHS]]
  // CHECK-DAG: %[[RS_RHS:.+]] = shapex.get_ranked_shape %[[RHS]]
  // CHECK-DAG: %[[RS_RESULT:.+]] = "shapex.ranked_broadcast_shape"(%[[RS_LHS]], %[[RS_RHS]]) {lhs_broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, rhs_broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: %[[BCAST_LHS:.+]] = "shapex.ranked_broadcast_in_dim"(%[[LHS]], %[[RS_RESULT]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-DAG: %[[BCAST_RHS:.+]] = "shapex.ranked_broadcast_in_dim"(%[[RHS]], %[[RS_RESULT]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: %[[SUM:.+]] = xla_hlo.add %[[BCAST_LHS]], %[[BCAST_RHS]]
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?x16xf32>, tensor<16xf32>) -> tensor<?x16xf32>
  // CHECK-DAG: return %[[SUM]]
  return %0 : tensor<?x16xf32>
}

// -----
// CHECK-LABEL: @implicitBroadcastAdd
func @implicitBroadcastAdd(%arg0 : tensor<16xf32>, %arg1 : tensor<1xf32>) -> (tensor<16xf32>) {
  // expected-error @+1 {{degenerate broadcast of same-rank operands not yet implemented}}
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<16xf32>, tensor<1xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
