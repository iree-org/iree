func.func @dynamic_attention_3x4() {
  // batch=1, seq_len=3, head_dim=4
  %query = flow.tensor.dynamic_constant dense<[[[0.0721, 0.1443, 0.2164, 0.2885],
                                                [0.3607, 0.4328, 0.5049, 0.5771],
                                                [0.6492, 0.7213, 0.7935, 0.8656]]]>
    : tensor<1x3x4xf32> -> tensor<?x?x?xf32>
  %key = flow.tensor.dynamic_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                              [0.5, 0.6, 0.7, 0.8],
                                              [0.9, 1.0, 1.1, 1.2]]]>
    : tensor<1x3x4xf32> -> tensor<?x?x?xf32>
  %value = flow.tensor.dynamic_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                                [0.5, 0.6, 0.7, 0.8],
                                                [0.9, 1.0, 1.1, 1.2]]]>
    : tensor<1x3x4xf32> -> tensor<?x?x?xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %batch = tensor.dim %query, %c0 : tensor<?x?x?xf32>
  %seq_len = tensor.dim %query, %c1 : tensor<?x?x?xf32>
  %head_dim = tensor.dim %query, %c2 : tensor<?x?x?xf32>

  %init = tensor.empty(%batch, %seq_len, %head_dim) : tensor<?x?x?xf32>

  %result = iree_linalg_ext.attention {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
    ]
  } ins(%query, %key, %value : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%init : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<?x?x?xf32>

  %barrier = util.optimization_barrier %result : tensor<?x?x?xf32>
  %result_static = tensor.cast %barrier : tensor<?x?x?xf32> to tensor<1x3x4xf32>
  check.expect_almost_eq_const(
      %result_static,
      dense<[[[0.5530, 0.6530, 0.7530, 0.8530],
              [0.6328, 0.7328, 0.8328, 0.9328],
              [0.7011, 0.8011, 0.9011, 1.0011]]]> : tensor<1x3x4xf32>
  ) : tensor<1x3x4xf32>
  return
}

func.func @dynamic_attention_3x3x4() {
  // batch=3, seq_len=3, head_dim=4
  %query = flow.tensor.dynamic_constant dense<[[[-1.1005, -0.5412, -0.4718, -1.1610],
                                                [-0.4394, -0.7068, -1.1607, -0.5137],
                                                [ 0.3373, -0.5028, -0.8373,  0.5046]],
                                               [[ 0.6245,  0.1763, -0.4782,  0.5823],
                                                [-0.1269, -1.6199, -1.0434,  0.0441],
                                                [-0.5580,  0.1436,  0.0330,  0.1104]],
                                               [[-0.0801,  0.2111, -0.1138, -0.0208],
                                                [ 0.8239,  0.1793, -1.2807, -0.0184],
                                                [ 1.1616, -0.5078, -0.1337, -0.7186]]]>
    : tensor<3x3x4xf32> -> tensor<?x?x?xf32>
  %key = flow.tensor.dynamic_constant dense<[[[-0.6092, -0.9798, -1.6091, -0.7121],
                                              [-0.7773, -0.2515, -0.2223,  1.6871],
                                              [ 0.4676, -0.6970, -1.1608,  0.6995]],
                                             [[ 0.8657,  0.2444, -0.6629,  0.8073],
                                              [-0.7981, -0.1316,  1.8793, -0.0721],
                                              [-0.7735,  0.1991,  0.0457,  0.1530]],
                                             [[-0.1110,  0.2927, -0.1578, -0.0288],
                                              [ 1.1422,  0.2486, -1.7754, -0.0255],
                                              [ 1.6103, -0.7040, -0.1853, -0.9962]]]>
    : tensor<3x3x4xf32> -> tensor<?x?x?xf32>
  %value = flow.tensor.dynamic_constant dense<[[[-1.5256, -0.7502, -0.6540, -1.6095],
                                                [-0.6092, -0.9798, -1.6091, -0.7121],
                                                [ 0.4676, -0.6970, -1.1608,  0.6995]],
                                               [[ 0.8657,  0.2444, -0.6629,  0.8073],
                                                [-0.1759, -2.2456, -1.4465,  0.0612],
                                                [-0.7773, -0.2515, -0.2223,  1.6871]],
                                               [[-0.1110,  0.2927, -0.1578, -0.0288],
                                                [-0.5962, -1.0055,  0.4285,  1.4761],
                                                [ 1.6103, -0.7040, -0.1853, -0.9962]]]>
    : tensor<3x3x4xf32> -> tensor<?x?x?xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %batch = tensor.dim %query, %c0 : tensor<?x?x?xf32>
  %seq_len = tensor.dim %query, %c1 : tensor<?x?x?xf32>
  %head_dim = tensor.dim %query, %c2 : tensor<?x?x?xf32>

  %init = tensor.empty(%batch, %seq_len, %head_dim) : tensor<?x?x?xf32>

  %result = iree_linalg_ext.attention {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
    ]
  } ins(%query, %key, %value : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%init : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<?x?x?xf32>

  %barrier = util.optimization_barrier %result : tensor<?x?x?xf32>
  %result_static = tensor.cast %barrier : tensor<?x?x?xf32> to tensor<3x3x4xf32>
  check.expect_almost_eq_const(
      %result_static,
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
