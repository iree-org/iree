// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -split-input-file -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -split-input-file -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

func @conv2d_nopadding() -> tensor<1x2x3x1xf32> {
  %0 = iree.unfoldable_constant dense<[[
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
        [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
        [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]]]> : tensor<1x4x5x2xf32>
  %1 = iree.unfoldable_constant dense<[[
        [[1.0], [2.0]], [[3.0], [4.0]]],
        [[[5.0], [6.0]], [[7.0], [8.0]]],
        [[[9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %2 = "xla_hlo.conv"(%0, %1) {
        batch_group_count = 1 : i64,
        dimension_numbers = {
          input_batch_dimension = 0 : i64,
          input_feature_dimension = 3 : i64,
          input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
          kernel_input_feature_dimension = 2 : i64,
          kernel_output_feature_dimension = 3 : i64,
          kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
          output_batch_dimension = 0 : i64,
          output_feature_dimension = 3 : i64,
          output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
        feature_group_count = 1 : i64,
        rhs_dilation = dense<1> : tensor<2xi64>,
        window_strides = dense<1> : tensor<2xi64>} : (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
  return %2 : tensor<1x2x3x1xf32>
}

// CHECK: EXEC @conv2d_nopadding
// CHECK: 1x2x3x1xf32=[
// CHECK-SAME: [
// CHECK-SAME: [1310][1466][1622]
// CHECK-SAME: ][
// CHECK-SAME: [2090][2246][2402]
// CHECK-SAME: ]
// CHECK-SAME: ]

// -----

func @conv2d_1452x3221_same() -> tensor<1x4x5x1xf32> {
  %0 = iree.unfoldable_constant dense<
       [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
         [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
         [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
         [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]
       ]]> : tensor<1x4x5x2xf32>
  %1 = iree.unfoldable_constant dense<
       [[[[1.0], [2.0]],
         [[3.0], [4.0]]
        ],
        [[[5.0], [6.0]],
         [[7.0], [8.0]]
        ],
        [[[9.0], [10.0]],
         [[11.0], [12.0]]
        ]]> : tensor<3x2x2x1xf32>
  %2 = "xla_hlo.conv"(%0, %1) {
       batch_group_count = 1 : i64,
       dimension_numbers = {
         input_batch_dimension = 0 : i64,
         input_feature_dimension = 3 : i64,
         input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
         kernel_input_feature_dimension = 2 : i64,
         kernel_output_feature_dimension = 3 : i64,
         kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
         output_batch_dimension = 0 : i64,
         output_feature_dimension = 3 : i64,
         output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
       feature_group_count = 1 : i64,
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>,
       rhs_dilation = dense<1> : tensor<2xi64>,
       window_strides = dense<1> : tensor<2xi64>} :
       (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x4x5x1xf32>
  return %2 : tensor<1x4x5x1xf32>
}

// CHECK: EXEC @conv2d_1452x3221_same
// CHECK: 1x4x5x1xf32=[
// CHECK-SAME: [
// CHECK-SAME: [600][736][872][1008][476]
// CHECK-SAME: ][
// CHECK-SAME: [1310][1466][1622][1778][805]
// CHECK-SAME: ][
// CHECK-SAME: [2090][2246][2402][2558][1135]
// CHECK-SAME: ][
// CHECK-SAME: [1080][1152][1224][1296][524]
// CHECK-SAME: ]
// CHECK-SAME: ]

// -----

func @conv2d_2451x2311_same() -> tensor<2x4x5x1xf32> {
  %0 = iree.unfoldable_constant dense<
       [[[[1.0], [2.0], [3.0], [4.0], [5.0]],
         [[6.0], [7.0], [8.0], [9.0], [10.0]],
         [[11.0], [12.0], [13.0], [14.0], [15.0]],
         [[16.0], [17.0], [18.0], [19.0], [20.0]]
        ],
        [[[21.0], [22.0], [23.0], [24.0], [25.0]],
         [[26.0], [27.0], [28.0], [29.0], [30.0]],
         [[31.0], [32.0], [33.0], [34.0], [35.0]],
         [[36.0], [37.0], [38.0], [39.0], [40.0]]
        ]]> : tensor <2x4x5x1xf32>
  %1 = iree.unfoldable_constant dense<
       [[[[1.0]],
         [[2.0]],
         [[3.0]]
        ],
        [[[4.0]],
         [[5.0]],
         [[6.0]]
        ]]> : tensor <2x3x1x1xf32>
  %2 = "xla_hlo.conv"(%0, %1) {
       batch_group_count = 1 : i64,
       dimension_numbers = {
         input_batch_dimension = 0 : i64,
         input_feature_dimension = 3 : i64,
         input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
         kernel_input_feature_dimension = 2 : i64,
         kernel_output_feature_dimension = 3 : i64,
         kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
         output_batch_dimension = 0 : i64,
         output_feature_dimension = 3 : i64,
         output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
       feature_group_count = 1 : i64,
       padding = dense<[[0, 1], [1, 1]]> : tensor<2x2xi64>,
       rhs_dilation = dense<1> : tensor<2xi64>,
       window_strides = dense<1> : tensor<2xi64>} :
       (tensor<2x4x5x1xf32>, tensor<2x3x1x1xf32>) -> tensor<2x4x5x1xf32>
  return %2 : tensor<2x4x5x1xf32>
}

// CHECK: EXEC @conv2d_2451x2311_same
// CHECK: 2x4x5x1xf32=[
// CHECK-SAME: [
// CHECK-SAME: [80][121][142][163][100]
// CHECK-SAME: ][
// CHECK-SAME: [160][226][247][268][160]
// CHECK-SAME: ][
// CHECK-SAME: [240][331][352][373][220]
// CHECK-SAME: ][
// CHECK-SAME: [83][104][110][116][59]
// CHECK-SAME: ]
// CHECK-SAME: ][
// CHECK-SAME: [
// CHECK-SAME: [400][541][562][583][340]
// CHECK-SAME: ][
// CHECK-SAME: [480][646][667][688][400]
// CHECK-SAME: ][
// CHECK-SAME: [560][751][772][793][460]
// CHECK-SAME: ][
// CHECK-SAME: [183][224][230][236][119]
// CHECK-SAME: ]
// CHECK-SAME: ]

// -----

func @conv2d_no_padding() -> tensor<2x3x3x6xf32> {
  %0 = iree.unfoldable_constant dense<
       [[[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0],
          [13.0, 14.0, 15.0]
         ],
         [[16.0, 17.0, 18.0],
          [19.0, 20.0, 21.0],
          [22.0, 23.0, 24.0],
          [25.0, 26.0, 27.0],
          [28.0, 29.0, 30.0]
         ],
         [[31.0, 32.0, 33.0],
          [34.0, 35.0, 36.0],
          [37.0, 38.0, 39.0],
          [40.0, 41.0, 42.0],
          [43.0, 44.0, 45.0]
         ],
         [[46.0, 47.0, 48.0],
          [49.0, 50.0, 51.0],
          [52.0, 53.0, 54.0],
          [55.0, 56.0, 57.0],
          [58.0, 59.0, 60.0]
         ]
        ],
        [[[61.0, 62.0, 63.0],
          [64.0, 65.0, 66.0],
          [67.0, 68.0, 69.0],
          [70.0, 71.0, 72.0],
          [73.0, 74.0, 75.0]
         ],
         [[76.0, 77.0, 78.0],
          [79.0, 80.0, 81.0],
          [82.0, 83.0, 84.0],
          [85.0, 86.0, 87.0],
          [88.0, 89.0, 90.0]
         ],
         [[91.0, 92.0, 93.0],
          [94.0, 95.0, 96.0],
          [97.0, 98.0, 99.0],
          [100.0, 101.0, 102.0],
          [103.0, 104.0, 105.0]
         ],
         [[106.0, 107.0, 108.0],
          [109.0, 110.0, 111.0],
          [112.0, 113.0, 114.0],
          [115.0, 116.0, 117.0],
          [118.0, 119.0, 120.0]
         ]
        ]]> : tensor<2x4x5x3xf32>
  %1 = iree.unfoldable_constant dense<
       [[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
         ],
         [[19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
          [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
          [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]
         ],
         [[37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
          [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
          [49.0, 50.0, 51.0, 52.0, 53.0, 54.0]
         ]
        ],
        [[[55.0, 56.0, 57.0, 58.0, 59.0, 60.0],
          [61.0, 62.0, 63.0, 64.0, 65.0, 66.0],
          [67.0, 68.0, 69.0, 70.0, 71.0, 72.0]
         ],
         [[73.0, 74.0, 75.0, 76.0, 77.0, 78.0],
          [79.0, 80.0, 81.0, 82.0, 83.0, 84.0],
          [85.0, 86.0, 87.0, 88.0, 89.0, 90.0]
         ],
         [[91.0, 92.0, 93.0, 94.0, 95.0, 96.0],
          [97.0, 98.0, 99.0, 100.0, 101.0, 102.0],
          [103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
         ]
        ]]> : tensor<2x3x3x6xf32>
  %2 = "xla_hlo.conv"(%0, %1) {
       batch_group_count = 1 : i64,
       dimension_numbers = {
         input_batch_dimension = 0 : i64,
         input_feature_dimension = 3 : i64,
         input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
         kernel_input_feature_dimension = 2 : i64,
         kernel_output_feature_dimension = 3 : i64,
         kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
         output_batch_dimension = 0 : i64,
         output_feature_dimension = 3 : i64,
         output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
       feature_group_count = 1 : i64,
       rhs_dilation = dense<1> : tensor<2xi64>,
       window_strides = dense<1> : tensor<2xi64>} :
       (tensor<2x4x5x3xf32>, tensor<2x3x3x6xf32>) -> tensor<2x3x3x6xf32>
  return %2 : tensor<2x3x3x6xf32>
}

//      CHECK: EXEC @conv2d_no_padding
//      CHECK: [
// CHECK-SAME: [
// CHECK-SAME: [16065 16290 16515 16740 16965 17190]
// CHECK-SAME: [18873 19152 19431 19710 19989 20268]
// CHECK-SAME: [21681 22014 22347 22680 23013 23346]
// CHECK-SAME: ]
// CHECK-SAME: [
// CHECK-SAME: [30105 30600 31095 31590 32085 32580]
// CHECK-SAME: [32913 33462 34011 34560 35109 35658]
// CHECK-SAME: [35721 36324 36927 37530 38133 38736]
// CHECK-SAME: ]
// CHECK-SAME: [
// CHECK-SAME: [44145 44910 45675 46440 47205 47970]
// CHECK-SAME: [46953 47772 48591 49410 50229 51048]
// CHECK-SAME: [49761 50634 51507 52380 53253 54126]
// CHECK-SAME: ]
// CHECK-SAME: ]
// CHECK-SAME: [
// CHECK-SAME: [
// CHECK-SAME: [72225 73530 74835 76140 77445 78750]
// CHECK-SAME: [75033 76392 77751 79110 80469 81828]
// CHECK-SAME: [77841 79254 80667 82080 83493 84906]
// CHECK-SAME: ]
// CHECK-SAME: [
// CHECK-SAME: [86265 87840 89415 90990 92565 94140]
// CHECK-SAME: [89073 90702 92331 93960 95589 97218]
// CHECK-SAME: [91881 93564 95247 96930 98613 100296]
// CHECK-SAME: ]
// CHECK-SAME: [
// CHECK-SAME: [100305 102150 103995 105840 107685 109530]
// CHECK-SAME: [103113 105012 106911 108810 110709 112608]
// CHECK-SAME: [105921 107874 109827 111780 113733 115686]
// CHECK-SAME: ]
// CHECK-SAME: ]
