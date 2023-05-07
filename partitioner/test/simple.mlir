module @jit__lambda_ {
  func.func public @main(%arg0: tensor<8xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}) -> (tensor<8xf32> {jax.result_info = ""}) {
    %0 = call @"<lambda>"(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @"<lambda>"(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{replicated}"} : (tensor<8xf32>) -> tensor<8xf32>
    %1 = stablehlo.constant dense<[5.000000e-01, 1.000000e+00, 5.000000e-01]> : tensor<3xf32>
    %2 = call @convolve(%0, %1) : (tensor<8xf32>, tensor<3xf32>) -> tensor<8xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {mhlo.sharding = "{maximal device=0}"} : (tensor<8xf32>) -> tensor<8xf32>
    return %3 : tensor<8xf32>
  }
  func.func private @convolve(%arg0: tensor<8xf32>, %arg1: tensor<3xf32>) -> tensor<8xf32> {
    %0 = call @_conv(%arg0, %arg1) : (tensor<8xf32>, tensor<3xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @_conv(%arg0: tensor<8xf32>, %arg1: tensor<3xf32>) -> tensor<8xf32> {
    %0 = call @_flip(%arg1) : (tensor<3xf32>) -> tensor<3xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<3xf32>) -> tensor<1x1x3xf32>
    %3 = stablehlo.convolution(%1, %2) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[1, 1]], lhs_dilate = [1], rhs_dilate = [1], reverse = [0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x1x8xf32>, tensor<1x1x3xf32>) -> tensor<1x1x8xf32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.concatenate %5, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = "stablehlo.gather"(%3, %8) {dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>} : (tensor<1x1x8xf32>, tensor<2xi32>) -> tensor<8xf32>
    return %9 : tensor<8xf32>
  }
  func.func private @_flip(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
