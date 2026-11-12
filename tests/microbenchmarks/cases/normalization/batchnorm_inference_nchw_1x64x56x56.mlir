// Inference batch norm folded to a per-channel scale and shift (constant), NCHW,
// 1x64x56x56 f32.

util.global private @scale = #util.byte_pattern<63> : tensor<64xf32>
util.global private @shift = #util.byte_pattern<63> : tensor<64xf32>

func.func @batchnorm_inference_nchw_1x64x56x56(%x: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
  %scale = util.global.load @scale : tensor<64xf32>
  %shift = util.global.load @shift : tensor<64xf32>
  %e = tensor.empty() : tensor<1x64x56x56xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (c)>,
                       affine_map<(n, c, h, w) -> (c)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x, %scale, %shift : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<1x64x56x56xf32>) {
  ^bb0(%in: f32, %s: f32, %b: f32, %out: f32):
    %m = arith.mulf %in, %s : f32
    %a = arith.addf %m, %b : f32
    linalg.yield %a : f32
  } -> tensor<1x64x56x56xf32>
  return %r : tensor<1x64x56x56xf32>
}
