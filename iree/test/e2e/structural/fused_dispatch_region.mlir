func @fused_add_transpose_dot_tanh() attributes { iree.module.export } {
  %term0 = iree.unfoldable_constant dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf32>
  %term1 = iree.unfoldable_constant dense<[
    [0.1, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.5]
  ]> : tensor<3x4xf32>
  %workload = constant 9 : index
  %dr0 = flow.dispatch.region[%workload: index](%arg0 = %term0 : tensor<4xf32>, %arg1 = %term1 : tensor<3x4xf32>) -> tensor<3x3xf32> {
    %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<4x3xf32>
    %2 = "mhlo.dot"(%arg1, %1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<3x4xf32>, tensor<4x3xf32>) -> tensor<3x3xf32>
    %3 = "mhlo.tanh"(%2) : (tensor<3x3xf32>) -> (tensor<3x3xf32>)
    flow.return %3 : tensor<3x3xf32>
  }
  check.expect_almost_eq_const(%dr0,
    dense<[[0.0199973, 0.00999967, 0.00999967],
           [0.0599281, 0.148885, 0.0599281],
           [0.309507, 0.309507, 0.623065]]> : tensor<3x3xf32>) : tensor<3x3xf32>
  return
}
