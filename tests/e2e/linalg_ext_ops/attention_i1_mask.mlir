func.func @attention1x4x4_i1_mask() {
  %init = tensor.empty() : tensor<1x4x4xf32>
  %query = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %key = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                          [0.5, 0.6, 0.7, 0.8],
                                          [0.9, 1.0, 1.1, 1.2],
                                          [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>
  %value = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %i8mask = util.unfoldable_constant dense<[165, 165]> : tensor<2xi8>
  %mask = flow.tensor.bitcast %i8mask : tensor<2xi8> -> tensor<1x4x4xi1>

  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>, f32, tensor<1x4x4xi1>) outs(%init : tensor<1x4x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x4x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.57895, 0.67895, 0.77895, 0.87895],
              [1.09108, 1.19108, 1.29108, 1.39108],
              [0.774324, 0.874324, 0.974324, 1.07432],
              [1.22842, 1.32842, 1.42842, 1.52842]]]> : tensor<1x4x4xf32>
  ) : tensor<1x4x4xf32>
  return
}

func.func @attention1x4x4_i1_mask_all_ones() {
  %init = tensor.empty() : tensor<1x4x4xf32>
  %query = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %key = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                          [0.5, 0.6, 0.7, 0.8],
                                          [0.9, 1.0, 1.1, 1.2],
                                          [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>
  %value = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %i8mask = util.unfoldable_constant dense<[255, 255]> : tensor<2xi8>
  %mask = flow.tensor.bitcast %i8mask : tensor<2xi8> -> tensor<1x4x4xi1>

  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>, f32, tensor<1x4x4xi1>) outs(%init : tensor<1x4x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x4x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.798884, 0.898884, 0.998884, 1.09888],
              [0.941939, 1.04194, 1.14194, 1.24194],
              [1.05371, 1.15371, 1.25371, 1.35371],
              [1.13295, 1.23295, 1.33295, 1.43295]]]> : tensor<1x4x4xf32>
  ) : tensor<1x4x4xf32>
  return
}

func.func @attention1x4x4_i1_mask_tril() {
  %init = tensor.empty() : tensor<1x4x4xf32>
  %query = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %key = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                          [0.5, 0.6, 0.7, 0.8],
                                          [0.9, 1.0, 1.1, 1.2],
                                          [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>
  %value = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %i8mask = util.unfoldable_constant dense<[140, 239]> : tensor<2xi8>
  %mask = flow.tensor.bitcast %i8mask : tensor<2xi8> -> tensor<1x4x4xi1>

  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>, f32, tensor<1x4x4xi1>) outs(%init : tensor<1x4x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x4x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[1.11993, 1.21993, 1.31993, 1.41993],
              [1.3, 1.4, 1.5, 1.6],
              [1.05371, 1.15371, 1.25371, 1.35371],
              [1.15549, 1.25549, 1.35549, 1.45549]]]> : tensor<1x4x4xf32>
  ) : tensor<1x4x4xf32>
  return
}
