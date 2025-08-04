// This file contains tests of linalg dialect operations with dynamic shapes.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// Batched matmul where B=32 M=1 N=128 K=dynamic.
// Tested with K=3 and splat operands.
func.func @batch_matmul_dynamic_reduction_size_B32_M1_N128_K3() {
  %lhs = flow.tensor.dynamic_constant dense<1.0> : tensor<32x1x3xf16> -> tensor<32x1x?xf16>
  %rhs = flow.tensor.dynamic_constant dense<2.0> : tensor<32x3x128xf16> -> tensor<32x?x128xf16>

  %cst = arith.constant 0.000000 : f16
  %2 = tensor.empty() : tensor<32x1x128xf16>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
  %observed = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%3 : tensor<32x1x128xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %6 = arith.mulf %in, %in_0 : f16
    %7 = arith.addf %out, %6 : f16
    linalg.yield %7 : f16
  } -> tensor<32x1x128xf16>

  %expected = flow.tensor.dynamic_constant dense<6.0> : tensor<32x1x128xf16> -> tensor<32x1x128xf16>
  check.expect_almost_eq(%observed, %expected, atol 1.0e-04) : tensor<32x1x128xf16>
  return
}

// Batched matmul where B=2 M=1 N=3 K=dynamic.
// Tested with K=5 and operands with varying values.
func.func @dynamic_matmul_dynamic_reduction_size_B2_M1_N3_K5() {
  %lhs = flow.tensor.dynamic_constant dense<[[[1.0, 2.0, 3.0, 4.0, 5.0]],
                                             [[6.0, 7.0, 8.0, 9.0, 10.0]]]>
     : tensor<2x1x5xf16> -> tensor<2x1x?xf16>
  %rhs = flow.tensor.dynamic_constant dense<[[[11.0, 12.0, 13.0],
                                              [14.0, 15.0, 16.0],
                                              [17.0, 18.0, 19.0],
                                              [20.0, 21.0, 22.0],
                                              [23.0, 24.0, 25.0]],
                                             [[26.0, 27.0, 28.0],
                                              [29.0, 30.0, 31.0],
                                              [32.0, 33.0, 34.0],
                                              [35.0, 36.0, 37.0],
                                              [38.0, 39.0, 40.0]]]>
       : tensor<2x5x3xf16> -> tensor<2x?x3xf16>

  %cst = arith.constant 0.000000 : f16
  %2 = tensor.empty() : tensor<2x1x3xf16>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<2x1x3xf16>) -> tensor<2x1x3xf16>
  %observed = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<2x1x?xf16>, tensor<2x?x3xf16>) outs(%3 : tensor<2x1x3xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %6 = arith.mulf %in, %in_0 : f16
    %7 = arith.addf %out, %6 : f16
    linalg.yield %7 : f16
  } -> tensor<2x1x3xf16>

  %expected = flow.tensor.dynamic_constant dense<[
        [[285.0, 300.0, 315.0]],
        [[1310.0, 1350.0, 1390.0]]
   ]> : tensor<2x1x3xf16> -> tensor<2x1x3xf16>
  check.expect_almost_eq(%observed, %expected, atol 1.0e-04) : tensor<2x1x3xf16>
  return
}
