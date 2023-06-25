module attributes {torch.debug_module_name = "TinyModel"} {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x1xf32> {
    %0 = stablehlo.constant dense<0.0893937349> : tensor<1xf32>
    %1 = stablehlo.constant dense<[[0.339793742, -0.249151886, -0.017383337, 0.154418945]]> : tensor<1x4xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<1x4xf32>) -> tensor<4x1xf32>
    %3 = stablehlo.dot %arg0, %2 : (tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<1x1xf32>
    %4 = chlo.broadcast_add %3, %0 : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
    return %4 : tensor<1x1xf32>
  }
}
