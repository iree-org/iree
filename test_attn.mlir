func.func @main(%q : tensor<1x1x4096x64xf16>, %k : tensor<1x1x64x64xf16>, %v : tensor<1x1x64x64xf16>, %m : tensor<1x1x4096x64xf16>) -> tensor<1x1x4096x64xf32> {
  %cst = arith.constant 1.250000e-01 : f16
  %o = tensor.empty() : tensor<1x1x4096x64xf32>
  %r = iree_linalg_ext.attention {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
    ]}
    ins(%q, %k, %v, %cst, %m: tensor<1x1x4096x64xf16>, tensor<1x1x64x64xf16>, tensor<1x1x64x64xf16>, f16, tensor<1x1x4096x64xf16>) outs(%o : tensor<1x1x4096x64xf32>) -> tensor<1x1x4096x64xf32>
  return %r : tensor<1x1x4096x64xf32>
}

// func.func @main(%q : tensor<1x1x4096x64xf16>, %k : tensor<1x1x64x64xf16>, %v : tensor<1x1x64x64xf16>) -> tensor<1x1x4096x64xf32> {
//   %cst = arith.constant 1.250000e-01 : f16
//   %o = tensor.empty() : tensor<1x1x4096x64xf32>
//   %r = iree_linalg_ext.attention {
//     indexing_maps = [
//       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
//       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
//       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
//       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//     ]}
//     ins(%q, %k, %v, %cst : tensor<1x1x4096x64xf16>, tensor<1x1x64x64xf16>, tensor<1x1x64x64xf16>, f16) outs(%o : tensor<1x1x4096x64xf32>) -> tensor<1x1x4096x64xf32>
//   return %r : tensor<1x1x4096x64xf32>
// }