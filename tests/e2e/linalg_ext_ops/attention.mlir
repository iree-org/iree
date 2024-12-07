func.func @attention1x3x4() {
  %init = tensor.empty() : tensor<1x3x4xf32>
  %query = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>

  %key = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                          [0.5, 0.6, 0.7, 0.8],
                                          [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>
  %value = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>
  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<1x3x4xf32>,
        tensor<1x3x4xf32>, tensor<1x3x4xf32>, f32) outs(%init : tensor<1x3x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x3x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.5530, 0.6530, 0.7530, 0.8530],
              [0.6328, 0.7328, 0.8328, 0.9328],
              [0.7011, 0.8011, 0.9011, 1.0011]]]> : tensor<1x3x4xf32>
  ) : tensor<1x3x4xf32>
  return
}

func.func @causal_attention1x3x4() {
  %init = tensor.empty() : tensor<1x3x4xf32>
  %query = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>

  %key = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                          [0.5, 0.6, 0.7, 0.8],
                                          [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>
  %value = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>
  %mask = util.unfoldable_constant dense<[[[true, false,  false],
                                           [true, true,   false],
                                           [true, true,   true]]]> : tensor<1x3x3xi1>
  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x3x4xf32>,
        tensor<1x3x4xf32>, tensor<1x3x4xf32>, f32, tensor<1x3x3xi1>) outs(%init : tensor<1x3x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x3x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.1000, 0.2000, 0.3000, 0.4000],
              [0.3509, 0.4509, 0.5509, 0.6509],
              [0.7011, 0.8011, 0.9011, 1.0011]]]> : tensor<1x3x4xf32>
  ) : tensor<1x3x4xf32>
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

  %mask = util.unfoldable_constant dense<[[[true, true, true, true],
                                           [true, true, true, true],
                                           [true, true, true, true],
                                           [true, true, true, true]]]> : tensor<1x4x4xi1>

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

func.func @softcap_attention1x3x4() {
  %init = tensor.empty() : tensor<1x3x4xf32>
  %query = util.unfoldable_constant dense<[[[-0.6523,  0.7802, -0.6770, -0.9474],
                                            [ 0.6822,  0.8594,  0.1035, -0.5115],
                                            [-0.7860, -0.5562, -0.2750, -0.0973]]]> : tensor<1x3x4xf32>

  %key = util.unfoldable_constant dense<[[[-0.3457,  0.5223, -0.0553,  0.5851],
                                          [ 0.4057, -0.8352,  0.1931,  0.1678],
                                          [ 0.1438,  0.7713, -0.6535,  0.9221]]]> : tensor<1x3x4xf32>
  %value = util.unfoldable_constant dense<[[[-0.2679,  0.6395,  0.4953,  0.3751],
                                            [ 0.8195, -0.5137,  0.2178,  0.9182],
                                            [ 0.8644,  0.7615,  0.2934,  0.9721]]]> : tensor<1x3x4xf32>
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<1x3x4xf32>,
        tensor<1x3x4xf32>, tensor<1x3x4xf32>, f32) outs(%init : tensor<1x3x4xf32>) {
          ^bb0(%arg0: f32):
          %tanh = math.tanh %arg0 : f32
          %cst = arith.constant 2.0 : f32
          %mul = arith.mulf %tanh, %cst : f32
          iree_linalg_ext.yield %mul : f32
        } -> tensor<1x3x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.331149, 0.576043, 0.380060, 0.6879],
              [0.519539, 0.532548, 0.342270, 0.78571],
              [0.442426, 0.125056, 0.329748, 0.735554]]]> : tensor<1x3x4xf32>
  ) : tensor<1x3x4xf32>
  return
}

func.func @attention1x4x4() {
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
  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>, f32) outs(%init : tensor<1x4x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x4x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.7989, 0.8989, 0.9989, 1.0989],
              [0.9419, 1.0419, 1.1419, 1.2419],
              [1.0537, 1.1537, 1.2537, 1.3537],
              [1.1329, 1.2329, 1.3329, 1.4329]]]> : tensor<1x4x4xf32>
  ) : tensor<1x4x4xf32>
  return
}

func.func @attention3x3x4() {
  %init = tensor.empty() : tensor<3x3x4xf32>
  %query = util.unfoldable_constant dense<[[[-1.5256, -0.7502, -0.6540, -1.6095],
                                            [-0.6092, -0.9798, -1.6091, -0.7121],
                                            [ 0.4676, -0.6970, -1.1608,  0.6995]],
                                           [[ 0.8657,  0.2444, -0.6629,  0.8073],
                                            [-0.1759, -2.2456, -1.4465,  0.0612],
                                            [-0.7735,  0.1991,  0.0457,  0.1530]],
                                           [[-0.1110,  0.2927, -0.1578, -0.0288],
                                            [ 1.1422, 0.2486,  -1.7754, -0.0255],
                                            [ 1.6103, -0.7040, -0.1853, -0.9962]]]> : tensor<3x3x4xf32>
  %key = util.unfoldable_constant dense<[[[-0.6092, -0.9798, -1.6091, -0.7121],
                                          [-0.7773, -0.2515, -0.2223,  1.6871],
                                          [ 0.4676, -0.6970, -1.1608,  0.6995]],
                                         [[ 0.8657,  0.2444, -0.6629,  0.8073],
                                          [-0.7981, -0.1316,  1.8793, -0.0721],
                                          [-0.7735,  0.1991,  0.0457,  0.1530]],
                                         [[-0.1110,  0.2927, -0.1578, -0.0288],
                                          [ 1.1422,  0.2486, -1.7754, -0.0255],
                                          [ 1.6103, -0.7040, -0.1853, -0.9962]]]> : tensor<3x3x4xf32>
  %value = util.unfoldable_constant dense<[[[-1.5256, -0.7502, -0.6540, -1.6095],
                                            [-0.6092, -0.9798, -1.6091, -0.7121],
                                            [ 0.4676, -0.6970, -1.1608,  0.6995]],
                                           [[ 0.8657,  0.2444, -0.6629,  0.8073],
                                            [-0.1759, -2.2456, -1.4465,  0.0612],
                                            [-0.7773, -0.2515, -0.2223,  1.6871]],
                                           [[-0.1110,  0.2927, -0.1578, -0.0288],
                                            [-0.5962, -1.0055,  0.4285,  1.4761],
                                            [ 1.6103, -0.7040, -0.1853, -0.9962]]]> : tensor<3x3x4xf32>
  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<3x3x4xf32>,
        tensor<3x3x4xf32>, tensor<3x3x4xf32>, f32) outs(%init : tensor<3x3x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<3x3x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[-1.2804, -0.7607, -0.7648, -1.3364],
              [-1.0711, -0.7572, -0.8238, -1.0953],
              [-0.4030, -0.7807, -1.1112, -0.3482]],
             [[ 0.4245, -0.1012, -0.6484,  0.9162],
              [ 0.1324, -0.2762, -0.6125,  1.0206],
              [-0.1866, -0.9266, -0.7977,  0.8593]],
             [[ 0.1917, -0.4658,  0.0510,  0.2561],
              [-0.1054, -0.8358,  0.2544,  0.8461],
              [ 0.9522, -0.7023, -0.0358, -0.3303]]]> : tensor<3x3x4xf32>
  ) : tensor<3x3x4xf32>
  return
}
