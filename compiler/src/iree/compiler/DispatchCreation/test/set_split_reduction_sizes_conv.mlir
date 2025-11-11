// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-split-reduction-sizes))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d5, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
module {
  util.func public @conv_1d_chn_chf(%arg0: tensor<16x50x32x96xf32>, %arg1: tensor<16x48x32x96xf32>, %arg2: tensor<96x3x96xf32>) -> tensor<96x3x96xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x50x32x96xf32>, tensor<16x48x32x96xf32>) outs(%arg2 : tensor<96x3x96xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<96x3x96xf32>
    util.return %0 : tensor<96x3x96xf32>
  }
}

// CHECK-LABEL: @conv_1d_chn_chf
//       CHECK: iree_linalg_ext.split_reduction = [1 : index, 12 : index, 32 : index]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_chwn_chwf_large(%arg0: tensor<16x227x227x16xf32>, %arg1: tensor<16x225x225x64xf32>, %arg2: tensor<64x3x3x16xf32>) -> tensor<64x3x3x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x227x227x16xf32>, tensor<16x225x225x64xf32>) outs(%arg2 : tensor<64x3x3x16xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.mulf %in, %in_3 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<64x3x3x16xf32>
  util.return %0 : tensor<64x3x3x16xf32>
}

// CHECK-LABEL: @conv_2d_chwn_chwf_large
//       CHECK: iree_linalg_ext.split_reduction = [1 : index, 45 : index, 225 : index]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @no_split_conv_2d_nhwc_fhwc(%arg0: tensor<16x227x227x16xf32>, %arg1: tensor<64x3x3x16xf32>, %arg2: tensor<16x225x225x64xf32>) -> tensor<16x225x225x64xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x227x227x16xf32>, tensor<64x3x3x16xf32>) outs(%arg2 : tensor<16x225x225x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x225x225x64xf32>
  util.return %0 : tensor<16x225x225x64xf32>
}

// CHECK-LABEL: @no_split_conv_2d_nhwc_fhwc
//   CHECK-NOT: iree_linalg_ext.split_reduction

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @no_split_large_N_F_sizes(%arg0: tensor<16x98x50x1024xf32>, %arg1: tensor<16x96x48x1024xf32>, %arg2: tensor<1024x3x3x1024xf32>) -> tensor<1024x3x3x1024xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x98x50x1024xf32>, tensor<16x96x48x1024xf32>) outs(%arg2 : tensor<1024x3x3x1024xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.mulf %in, %in_3 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<1024x3x3x1024xf32>
  util.return %0 : tensor<1024x3x3x1024xf32>
}

// CHECK-LABEL:  @no_split_large_N_F_sizes
//   CHECK-NOT:  iree_linalg_ext.split_reduction

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @no_split_small_H_W_sizes(%arg0: tensor<16x26x18x96xf32>, %arg1: tensor<16x24x16x96xf32>, %arg2: tensor<96x3x3x96xf32>) -> tensor<96x3x3x96xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x26x18x96xf32>, tensor<16x24x16x96xf32>) outs(%arg2 : tensor<96x3x3x96xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.mulf %in, %in_3 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<96x3x3x96xf32>
  util.return %0 : tensor<96x3x3x96xf32>
}

// CHECK-LABEL:  @no_split_small_H_W_sizes
//   CHECK-NOT:  iree_linalg_ext.split_reduction
