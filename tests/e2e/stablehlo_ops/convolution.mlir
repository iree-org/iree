func.func @conv2d_nopadding() {
  %inputs = util.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0]]]]> : tensor<1x4x4x2xf32>
  %weights = util.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
        feature_group_count = 1 : i64,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>} : (tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
  check.expect_almost_eq_const(%res, dense<[[
      [[1310.0],[1466.0],[1622.0]],
      [[2090.0],[2246.0],[2402.0]]
  ]]> : tensor<1x2x3x1xf32>) : tensor<1x2x3x1xf32>
  return
}

func.func @conv2d_nopadding_batch_feature() {
  %inputs = util.unfoldable_constant dense<[
    [[[ 1.0], [ 3.0], [ 5.0], [ 7.0]],
     [[11.0], [13.0], [15.0], [17.0]],
     [[21.0], [23.0], [25.0], [27.0]],
     [[31.0], [33.0], [35.0], [37.0]]],
    [[[ 2.0], [ 4.0], [ 6.0], [ 8.0]],
     [[12.0], [14.0], [16.0], [18.0]],
     [[22.0], [24.0], [26.0], [28.0]],
     [[32.0], [34.0], [36.0], [38.0]]]
      ]> : tensor<2x4x4x1xf32>
  %weights = util.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 3,
          input_feature_dimension = 0,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
        feature_group_count = 1 : i64,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>} : (tensor<2x4x4x1xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
  check.expect_almost_eq_const(%res, dense<[[
      [[1310.0],[1466.0],[1622.0]],
      [[2090.0],[2246.0],[2402.0]]
  ]]> : tensor<1x2x3x1xf32>) : tensor<1x2x3x1xf32>
  return
}

func.func @conv2d_reorder_input_spatial() {
  %inputs = util.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [11.0, 12.0], [21.0, 22.0], [31.0, 32.0]],
      [[ 3.0,  4.0], [13.0, 14.0], [23.0, 24.0], [33.0, 34.0]],
      [[ 5.0,  6.0], [15.0, 16.0], [25.0, 26.0], [35.0, 36.0]],
      [[ 7.0,  8.0], [17.0, 18.0], [27.0, 28.0], [37.0, 38.0]]]]> : tensor<1x4x4x2xf32>
  %weights = util.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [2, 1],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
        feature_group_count = 1 : i64,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>} : (tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
  check.expect_almost_eq_const(%res, dense<[[
      [[1310.0],[1466.0],[1622.0]],
      [[2090.0],[2246.0],[2402.0]]
  ]]> : tensor<1x2x3x1xf32>) : tensor<1x2x3x1xf32>
  return
}

func.func @conv2d_reorder_kernel() {
  %inputs = util.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0]]]]> : tensor<1x4x4x2xf32>
  %weights = util.unfoldable_constant dense<
      [[[[ 1.0,  3.0], [ 2.0,  4.0]],
        [[ 5.0,  7.0], [ 6.0,  8.0]],
        [[ 9.0, 11.0], [10.0, 12.0]]]]> : tensor<1x3x2x2xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 0,
          kernel_spatial_dimensions = [1, 3],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
        feature_group_count = 1 : i64,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>} : (tensor<1x4x4x2xf32>, tensor<1x3x2x2xf32>) -> tensor<1x2x3x1xf32>
  check.expect_almost_eq_const(%res, dense<[[
      [[1310.0],[1466.0],[1622.0]],
      [[2090.0],[2246.0],[2402.0]]
  ]]> : tensor<1x2x3x1xf32>) : tensor<1x2x3x1xf32>
  return
}

func.func @conv2d_reorder_output() {
  %inputs = util.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0]]]]> : tensor<1x4x4x2xf32>
  %weights = util.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 2,
          output_feature_dimension = 0,
          output_spatial_dimensions = [3, 1]
        >,
        feature_group_count = 1 : i64,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>} : (tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x3x1x2xf32>
  check.expect_almost_eq_const(%res, dense<[[
      [[1310.0, 2090.0]],
      [[1466.0, 2246.0]],
      [[1622.0, 2402.0]]
      ]]> : tensor<1x3x1x2xf32>) : tensor<1x3x1x2xf32>
  return
}

func.func @conv2d_1452x3221_same() {
  %inputs = util.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0], [ 9.0, 10.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]]]> : tensor<1x4x5x2xf32>
  %weights = util.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
       batch_group_count = 1 : i64,
       dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
       feature_group_count = 1 : i64,
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>,
       rhs_dilation = array<i64: 1, 1>,
       window_strides = array<i64: 1, 1>} :
       (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x4x5x1xf32>
  check.expect_almost_eq_const(%res,  dense<[[
    [[ 600.0], [ 736.0], [ 872.0], [1008.0], [ 476.0]],
    [[1310.0], [1466.0], [1622.0], [1778.0], [ 805.0]],
    [[2090.0], [2246.0], [2402.0], [2558.0], [1135.0]],
    [[1080.0], [1152.0], [1224.0], [1296.0], [ 524.0]]]]> : tensor<1x4x5x1xf32>) : tensor<1x4x5x1xf32>
  return
}

func.func @conv2d_2451x2311_same() {
  %inputs = util.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0]],
       [[ 6.0], [ 7.0], [ 8.0], [ 9.0], [10.0]],
       [[11.0], [12.0], [13.0], [14.0], [15.0]],
       [[16.0], [17.0], [18.0], [19.0], [20.0]]],
      [[[21.0], [22.0], [23.0], [24.0], [25.0]],
       [[26.0], [27.0], [28.0], [29.0], [30.0]],
       [[31.0], [32.0], [33.0], [34.0], [35.0]],
       [[36.0], [37.0], [38.0], [39.0], [40.0]]]]> : tensor <2x4x5x1xf32>
  %weights = util.unfoldable_constant dense<[
      [[[1.0]], [[2.0]], [[3.0]]],
      [[[4.0]], [[5.0]], [[6.0]]]]> : tensor <2x3x1x1xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
       batch_group_count = 1 : i64,
       dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
       feature_group_count = 1 : i64,
       padding = dense<[[0, 1], [1, 1]]> : tensor<2x2xi64>,
       rhs_dilation = array<i64: 1, 1>,
       window_strides = array<i64: 1, 1>} :
       (tensor<2x4x5x1xf32>, tensor<2x3x1x1xf32>) -> tensor<2x4x5x1xf32>
  check.expect_almost_eq_const(%res, dense<[
    [[[ 80.0], [121.0], [142.0], [163.0], [100.0]],
     [[160.0], [226.0], [247.0], [268.0], [160.0]],
     [[240.0], [331.0], [352.0], [373.0], [220.0]],
     [[ 83.0], [104.0], [110.0], [116.0], [ 59.0]]],
    [[[400.0], [541.0], [562.0], [583.0], [340.0]],
     [[480.0], [646.0], [667.0], [688.0], [400.0]],
     [[560.0], [751.0], [772.0], [793.0], [460.0]],
     [[183.0], [224.0], [230.0], [236.0], [119.0]]]]> : tensor<2x4x5x1xf32>) : tensor<2x4x5x1xf32>
  return
}

func.func @conv2d_no_padding2() {
  %inputs = util.unfoldable_constant dense<[
       [[[  1.0,   2.0,   3.0],
         [  4.0,   5.0,   6.0],
         [  7.0,   8.0,   9.0],
         [ 10.0,  11.0,  12.0],
         [ 13.0,  14.0,  15.0]],
        [[ 16.0,  17.0,  18.0],
         [ 19.0,  20.0,  21.0],
         [ 22.0,  23.0,  24.0],
         [ 25.0,  26.0,  27.0],
         [ 28.0,  29.0,  30.0]],
        [[ 31.0,  32.0,  33.0],
         [ 34.0,  35.0,  36.0],
         [ 37.0,  38.0,  39.0],
         [ 40.0,  41.0,  42.0],
         [ 43.0,  44.0,  45.0]],
        [[ 46.0,  47.0,  48.0],
         [ 49.0,  50.0,  51.0],
         [ 52.0,  53.0,  54.0],
         [ 55.0,  56.0,  57.0],
         [ 58.0,  59.0,  60.0]]],
       [[[ 61.0,  62.0,  63.0],
         [ 64.0,  65.0,  66.0],
         [ 67.0,  68.0,  69.0],
         [ 70.0,  71.0,  72.0],
         [ 73.0,  74.0,  75.0]],
        [[ 76.0,  77.0,  78.0],
         [ 79.0,  80.0,  81.0],
         [ 82.0,  83.0,  84.0],
         [ 85.0,  86.0,  87.0],
         [ 88.0,  89.0,  90.0]],
        [[ 91.0,  92.0,  93.0],
         [ 94.0,  95.0,  96.0],
         [ 97.0,  98.0,  99.0],
         [100.0, 101.0, 102.0],
         [103.0, 104.0, 105.0]],
        [[106.0, 107.0, 108.0],
         [109.0, 110.0, 111.0],
         [112.0, 113.0, 114.0],
         [115.0, 116.0, 117.0],
         [118.0, 119.0, 120.0]]]]> : tensor<2x4x5x3xf32>
  %weights = util.unfoldable_constant dense<[
      [[[  1.0,   2.0,   3.0,   4.0,   5.0,   6.0],
        [  7.0,   8.0,   9.0,  10.0,  11.0,  12.0],
        [ 13.0,  14.0,  15.0,  16.0,  17.0,  18.0]],
       [[ 19.0,  20.0,  21.0,  22.0,  23.0,  24.0],
        [ 25.0,  26.0,  27.0,  28.0,  29.0,  30.0],
        [ 31.0,  32.0,  33.0,  34.0,  35.0,  36.0]],
       [[ 37.0,  38.0,  39.0,  40.0,  41.0,  42.0],
        [ 43.0,  44.0,  45.0,  46.0,  47.0,  48.0],
        [ 49.0,  50.0,  51.0,  52.0,  53.0,  54.0]]],
      [[[ 55.0,  56.0,  57.0,  58.0,  59.0,  60.0],
        [ 61.0,  62.0,  63.0,  64.0,  65.0,  66.0],
        [ 67.0,  68.0,  69.0,  70.0,  71.0,  72.0]],
       [[ 73.0,  74.0,  75.0,  76.0,  77.0,  78.0],
        [ 79.0,  80.0,  81.0,  82.0,  83.0,  84.0],
        [ 85.0,  86.0,  87.0,  88.0,  89.0,  90.0]],
       [[ 91.0,  92.0,  93.0,  94.0,  95.0,  96.0],
        [ 97.0,  98.0,  99.0, 100.0, 101.0, 102.0],
        [103.0, 104.0, 105.0, 106.0, 107.0, 108.0]]]]> : tensor<2x3x3x6xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
       batch_group_count = 1 : i64,
       dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
       feature_group_count = 1 : i64,
       rhs_dilation = array<i64: 1, 1>,
       window_strides = array<i64: 1, 1>} :
       (tensor<2x4x5x3xf32>, tensor<2x3x3x6xf32>) -> tensor<2x3x3x6xf32>
  check.expect_almost_eq_const(%res, dense<[
      [[[16065.0,  16290.0,  16515.0,  16740.0,  16965.0,  17190.0],
        [18873.0,  19152.0,  19431.0,  19710.0,  19989.0,  20268.0],
        [21681.0,  22014.0,  22347.0,  22680.0,  23013.0,  23346.0]],
       [[30105.0,  30600.0,  31095.0,  31590.0,  32085.0,  32580.0],
        [32913.0,  33462.0,  34011.0,  34560.0,  35109.0,  35658.0],
        [35721.0,  36324.0,  36927.0,  37530.0,  38133.0,  38736.0]],
       [[44145.0,  44910.0,  45675.0,  46440.0,  47205.0,  47970.0],
        [46953.0,  47772.0,  48591.0,  49410.0,  50229.0,  51048.0],
        [49761.0,  50634.0,  51507.0,  52380.0,  53253.0,  54126.0]]],
      [[[72225.0,  73530.0,  74835.0,  76140.0,  77445.0,  78750.0],
        [75033.0,  76392.0,  77751.0,  79110.0,  80469.0,  81828.0],
        [77841.0,  79254.0,  80667.0,  82080.0,  83493.0,  84906.0]],
       [[86265.0,  87840.0,  89415.0,  90990.0,  92565.0,  94140.0],
        [89073.0,  90702.0,  92331.0,  93960.0,  95589.0,  97218.0],
        [91881.0,  93564.0,  95247.0,  96930.0,  98613.0, 100296.0]],
       [[100305.0, 102150.0, 103995.0, 105840.0, 107685.0, 109530.0],
        [103113.0, 105012.0, 106911.0, 108810.0, 110709.0, 112608.0],
        [105921.0, 107874.0, 109827.0, 111780.0, 113733.0, 115686.0]]]]> : tensor<2x3x3x6xf32>) : tensor<2x3x3x6xf32>
  return
}

func.func @conv2d_1452x2223_dilated_valid() {
  %inputs = util.unfoldable_constant dense<
     [[[[0.09762701,  0.43037874],
       [ 0.20552675,  0.08976637],
       [-0.1526904,   0.29178822],
       [-0.12482557,  0.78354603],
       [ 0.92732555, -0.23311697]],
      [[ 0.5834501,   0.05778984],
       [ 0.13608912,  0.85119325],
       [-0.85792786, -0.8257414 ],
       [-0.9595632,   0.6652397 ],
       [ 0.5563135,   0.74002427]],
      [[ 0.9572367,   0.59831715],
       [-0.07704128,  0.56105834],
       [-0.76345116,  0.27984205],
       [-0.71329343,  0.88933784],
       [ 0.04369664, -0.17067613]],
      [[-0.47088876,  0.5484674 ],
       [-0.08769934,  0.1368679 ],
       [-0.9624204,   0.23527099],
       [ 0.22419144,  0.23386799],
       [ 0.8874962,   0.3636406 ]]]]> : tensor<1x4x5x2xf32>
  %weights = util.unfoldable_constant dense<
    [[[[-0.2809842,  -0.12593609,  0.3952624 ],
       [-0.8795491,   0.33353344,  0.34127575]],
      [[-0.5792349,  -0.7421474,  -0.3691433 ],
       [-0.27257845,  0.14039354, -0.12279698]]],
     [[[ 0.9767477,  -0.79591036, -0.5822465 ],
       [-0.677381,    0.30621666, -0.4934168 ]],
      [[-0.06737845, -0.5111488,  -0.68206084],
       [-0.7792497,   0.31265917, -0.7236341 ]]]]> : tensor<2x2x2x3xf32>
  %res = "stablehlo.convolution"(%inputs, %weights) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 2, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>) -> tensor<1x2x4x3xf32>
  check.expect_almost_eq_const(%res, dense<
    [[[[-0.45181108, -0.37253797, -1.1074474 ],
       [-0.74972206,  0.8691965,   0.21864426],
       [-1.9352274,   1.6551838,   0.13848126],
       [-2.296763,    0.32046723, -0.02542188]],
      [[-1.4578199,   0.59465677,  0.0599021 ],
       [-0.3617443,   1.4647548,   1.2320882 ],
       [ 0.04506956,  1.4347346,  -0.22625303],
       [-1.122044,   -0.41301775, -1.5628793 ]]]]> : tensor<1x2x4x3xf32>) : tensor<1x2x4x3xf32>
  return
}

func.func @depthwise_conv_non_1_channel_multiplier() {
  %arg0 = util.unfoldable_constant dense<1.0> : tensor<2x4x5x2xf32>
  %arg1 = util.unfoldable_constant dense<1.0> : tensor<2x2x1x6xf32>
  %res = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
    feature_group_count = 2 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>} : (tensor<2x4x5x2xf32>, tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32>
  check.expect_almost_eq_const(%res, dense<4.0> : tensor<2x3x4x6xf32>) : tensor<2x3x4x6xf32>
  return
}
