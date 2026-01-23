module @link_module_a {
  util.func public @compute(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = util.call @internal_helper(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    util.return %0 : tensor<4xf32>
  }

  util.func private @internal_helper(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    util.return %arg0 : tensor<4xf32>
  }
}
