// Max pool 2x2 stride 2, NHWC, 1x112x112x64 -> 1x56x56x64 f32.

func.func @maxpool_nhwc_2x2_s2(%x: tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %win = tensor.empty() : tensor<2x2xf32>
  %e = tensor.empty() : tensor<1x56x56x64xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %r = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%x, %win : tensor<1x112x112x64xf32>, tensor<2x2xf32>) outs(%f : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %r : tensor<1x56x56x64xf32>
}
