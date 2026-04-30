// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-split-reduction-sizes{low-parallelism=true}))" --split-input-file %s | FileCheck %s

util.func public @split_balanced_matmul(%arg0: tensor<128x32768xbf16>, %arg1: tensor<32768x128xbf16>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x32768xbf16>, tensor<32768x128xbf16>) outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  util.return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: @split_balanced_matmul
//       CHECK: iree_linalg_ext.split_reduction = [4096 : index]

// -----

util.func public @split_mid_output_small_k(%arg0: tensor<128x131072xbf16>, %arg1: tensor<131072x256xbf16>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x131072xbf16>, tensor<131072x256xbf16>) outs(%arg2 : tensor<128x256xf32>) -> tensor<128x256xf32>
  util.return %0 : tensor<128x256xf32>
}

// CHECK-LABEL: @split_mid_output_small_k
//       CHECK: iree_linalg_ext.split_reduction = [16384 : index]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d5, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
util.func public @conv_1d_chn_chf(%arg0: tensor<16x50x32x96xf32>, %arg1: tensor<16x48x32x96xf32>, %arg2: tensor<96x3x96xf32>) -> tensor<96x3x96xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x50x32x96xf32>, tensor<16x48x32x96xf32>) outs(%arg2 : tensor<96x3x96xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<96x3x96xf32>
  util.return %0 : tensor<96x3x96xf32>
}

// CHECK-LABEL: @conv_1d_chn_chf
//       CHECK: iree_linalg_ext.split_reduction = [2 : index, 48 : index, 32 : index]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d5, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
util.func public @conv_1d_chn_chf_mid_reduction(%arg0: tensor<16x130x32x96xf32>, %arg1: tensor<16x128x32x96xf32>, %arg2: tensor<96x3x96xf32>) -> tensor<96x3x96xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x130x32x96xf32>, tensor<16x128x32x96xf32>) outs(%arg2 : tensor<96x3x96xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<96x3x96xf32>
  util.return %0 : tensor<96x3x96xf32>
}

// CHECK-LABEL: @conv_1d_chn_chf_mid_reduction
//       CHECK: iree_linalg_ext.split_reduction = [1 : index, 128 : index, 32 : index]
