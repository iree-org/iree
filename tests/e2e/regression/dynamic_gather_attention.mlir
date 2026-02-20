#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d6)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @gather_attention() {
  %0 = util.unfoldable_constant dense<1.000000e+00> : tensor<32x4x2x32xf16>
  %1 = flow.tensor.dynamic_constant dense<5.000000e-01> : tensor<2x4x16x32xf16> -> tensor<?x4x16x32xf16>
  %2 = flow.tensor.dynamic_constant dense<1.500000e+00> : tensor<2x4x16x32xf16> -> tensor<?x4x16x32xf16>
  %3 = flow.tensor.dynamic_constant dense<1> : tensor<32x2xi64> -> tensor<32x?xi64>
  %4 = flow.tensor.dynamic_constant dense<1.500000e+00> : tensor<32x4x2x2x16xf16> -> tensor<32x4x2x?x16xf16>
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %3, %c1 : tensor<32x?xi64>
  %5 = tensor.empty(%dim) : tensor<32x?x4x16x32xf16>
  %6 = iree_linalg_ext.gather dimension_map = [0] ins(%1, %3 : tensor<?x4x16x32xf16>, tensor<32x?xi64>) outs(%5 : tensor<32x?x4x16x32xf16>) -> tensor<32x?x4x16x32xf16>
  %7 = iree_linalg_ext.gather dimension_map = [0] ins(%2, %3 : tensor<?x4x16x32xf16>, tensor<32x?xi64>) outs(%5 : tensor<32x?x4x16x32xf16>) -> tensor<32x?x4x16x32xf16>
  %8 = tensor.empty() : tensor<32x4x2x32xf16>
  %9 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%0, %6, %7, %4 : tensor<32x4x2x32xf16>, tensor<32x?x4x16x32xf16>, tensor<32x?x4x16x32xf16>, tensor<32x4x2x?x16xf16>) outs(%8 : tensor<32x4x2x32xf16>) {
  ^bb0(%arg0: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<32x4x2x32xf16>
  check.expect_almost_eq_const(%9, dense<1.500000e+00> : tensor<32x4x2x32xf16>) : tensor<32x4x2x32xf16>
  return
}
