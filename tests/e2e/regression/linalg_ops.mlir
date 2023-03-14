func.func @multi_result() {
  %input1 = util.unfoldable_constant dense<[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<3x4xi32>
  %input2 = util.unfoldable_constant dense<[
      [13, 14, 15, 16],
      [17, 18, 19, 20],
      [21, 22, 23, 24]]> : tensor<3x4xi32>
  %init = tensor.empty() : tensor<3x4xi32>
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input1, %input2 : tensor<3x4xi32>, tensor<3x4xi32>)
      outs(%init, %init : tensor<3x4xi32>, tensor<3x4xi32>) {
      ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) :
          %1 = arith.addi %arg0, %arg1 : i32
          %2 = arith.muli %arg0, %arg1 : i32
          linalg.yield %1, %2 : i32, i32
      } -> (tensor<3x4xi32>, tensor<3x4xi32>)
  check.expect_eq_const(%0#0, dense<[
      [14, 16, 18, 20],
      [22, 24, 26, 28],
      [30, 32, 34, 36]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  check.expect_eq_const(%0#1, dense<[
      [13, 28, 45, 64],
      [85, 108, 133, 160],
      [189, 220, 253, 288]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}

func.func @operand_fusion() {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x225x225x3xf32>
  %filter = util.unfoldable_constant dense<1.0> : tensor<3x3x3x16xf32>
  %bias = util.unfoldable_constant dense<1.0> : tensor<16xf32>
  %init = tensor.empty() : tensor<1x112x112x16xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>)
      outs(%fill : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias : tensor<1x112x112x16xf32>, tensor<16xf32>)
      outs(%init : tensor<1x112x112x16xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
      } -> tensor<1x112x112x16xf32>
  check.expect_eq_const(%result, dense<28.0> : tensor<1x112x112x16xf32>) : tensor<1x112x112x16xf32>
  return
}
