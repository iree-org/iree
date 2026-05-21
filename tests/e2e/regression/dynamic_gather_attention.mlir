#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d6)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>
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
  %cst = arith.constant 1.767580e-01 : f16
  %8 = tensor.empty() : tensor<32x4x2x32xf32>
  %9 = tensor.empty() : tensor<32x4x2xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %10 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<32x4x2x32xf32>) -> tensor<32x4x2x32xf32>
  %11 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<32x4x2xf32>) -> tensor<32x4x2xf32>
  %12 = linalg.fill ins(%cst_2 : f32) outs(%9 : tensor<32x4x2xf32>) -> tensor<32x4x2xf32>
  %13:3 = iree_linalg_ext.online_attention {indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5, #map6, #map6]} ins(%0, %6, %7, %cst, %4 : tensor<32x4x2x32xf16>, tensor<32x?x4x16x32xf16>, tensor<32x?x4x16x32xf16>, f16, tensor<32x4x2x?x16xf16>) outs(%10, %11, %12 : tensor<32x4x2x32xf32>, tensor<32x4x2xf32>, tensor<32x4x2xf32>) {
  ^bb0(%arg0: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<32x4x2x32xf32>, tensor<32x4x2xf32>, tensor<32x4x2xf32>
  %out_e = tensor.empty() : tensor<32x4x2x32xf32>
  %cst_one = arith.constant 1.000000e+00 : f32
  %normalized = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%13#0, %13#2 : tensor<32x4x2x32xf32>, tensor<32x4x2xf32>)
    outs(%out_e : tensor<32x4x2x32xf32>) {
  ^bb0(%a: f32, %s: f32, %_: f32):
    %inv = arith.divf %cst_one, %s : f32
    %v = arith.mulf %a, %inv : f32
    linalg.yield %v : f32
  } -> tensor<32x4x2x32xf32>
  check.expect_almost_eq_const(%normalized, dense<1.500000e+00> : tensor<32x4x2x32xf32>) : tensor<32x4x2x32xf32>
  return
}
