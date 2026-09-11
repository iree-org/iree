// Fully-masked attention with low-precision (f16 / bf16) Q/K/V/mask and scale.
// Softmax internals run in f32 via the region's score type.
func.func @attention1x16x16_f16_mask_fully_masked() {
  %init = tensor.empty() : tensor<1x16x16xf16>
  %query = util.unfoldable_constant dense<0.1> : tensor<1x16x16xf16>
  %key = util.unfoldable_constant dense<0.2> : tensor<1x16x16xf16>
  %value = util.unfoldable_constant dense<0.3> : tensor<1x16x16xf16>
  %mask = util.unfoldable_constant dense<0xFC00> : tensor<1x16x16xf16>
  %scale = arith.constant 0.5 : f16
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x16x16xf16>,
        tensor<1x16x16xf16>, tensor<1x16x16xf16>, f16, tensor<1x16x16xf16>) outs(%init : tensor<1x16x16xf16>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x16x16xf16>
  check.expect_almost_eq_const(
      %1,
      dense<0.0> : tensor<1x16x16xf16>
  ) : tensor<1x16x16xf16>
  return
}

func.func @attention1x16x16_bf16_mask_fully_masked() {
  %init = tensor.empty() : tensor<1x16x16xbf16>
  %query = util.unfoldable_constant dense<0.1> : tensor<1x16x16xbf16>
  %key = util.unfoldable_constant dense<0.2> : tensor<1x16x16xbf16>
  %value = util.unfoldable_constant dense<0.3> : tensor<1x16x16xbf16>
  %mask = util.unfoldable_constant dense<0xFF80> : tensor<1x16x16xbf16>
  %scale = arith.constant 0.5 : bf16
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x16x16xbf16>,
        tensor<1x16x16xbf16>, tensor<1x16x16xbf16>, bf16, tensor<1x16x16xbf16>) outs(%init : tensor<1x16x16xbf16>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x16x16xbf16>
  check.expect_almost_eq_const(
      %1,
      dense<0.0> : tensor<1x16x16xbf16>
  ) : tensor<1x16x16xbf16>
  return
}
