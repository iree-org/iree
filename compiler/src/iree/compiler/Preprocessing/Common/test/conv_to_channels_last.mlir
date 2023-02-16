// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last))" %s | \
// RUN:   FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last{tile-size=16}))" %s | \
// RUN:   FileCheck %s --check-prefix=TILE16

util.func @conv_nhwc_hwcf_no_transpose(%arg0: tensor<1x16x16x256xf32>, %arg1: tensor<3x3x256x128xf32>, %arg2: tensor<1x14x14x128xf32>) -> tensor<1x14x14x128xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x16x16x256xf32>, tensor<3x3x256x128xf32>)
      outs(%arg2: tensor<1x14x14x128xf32>) -> tensor<1x14x14x128xf32>
    util.return %0 : tensor<1x14x14x128xf32>
}
// CHECK-LABEL: @conv_nhwc_hwcf_no_transpose
// CHECK: linalg.conv_2d_nhwc_hwcf

// TILE16-LABEL: @conv_nhwc_hwcf_no_transpose
// TILE16: linalg.conv_2d_nhwc_hwcf

// -----

util.func @conv_nchw_nhwc(%arg0: tensor<8x256x16x16xf32>, %arg1: tensor<16x256x3x3xf32>, %arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32> {
    %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<8x256x16x16xf32>, tensor<16x256x3x3xf32>)
      outs(%arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32>
    util.return %0 : tensor<8x16x14x14xf32>
}

// CHECK-LABEL: util.func public @conv_nchw_nhwc

// CHECK:         %[[IMG:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x256x16x16xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<8x16x16x256xf32>) permutation = [0, 2, 3, 1]
// CHECK:         %[[FILTER:.+]] = linalg.transpose ins(%{{.*}} : tensor<16x256x3x3xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<3x3x256x16xf32>) permutation = [2, 3, 1, 0]
// CHECK:         %[[OUT:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x16x14x14xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<8x14x14x16xf32>) permutation = [0, 2, 3, 1]

// CHECK:         %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:      ins(%[[IMG]], %[[FILTER]] : tensor<8x16x16x256xf32>, tensor<3x3x256x16xf32>) outs(%[[OUT]] : tensor<8x14x14x16xf32>) -> tensor<8x14x14x16xf32>
// CHECK:         linalg.transpose ins(%[[CONV]] : tensor<8x14x14x16xf32>) outs(%{{.*}} : tensor<8x16x14x14xf32>)

// TILE16: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d7)>
// TILE16: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d7, d8)>
// TILE16: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d8)>

// TILE16-LABEL: util.func public @conv_nchw_nhwc
// TILE16:      %[[IMG:.+]] = linalg.transpose ins(%{{[A-Za-z0-9]+}} : tensor<8x16x16x16x16xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<8x16x16x16x16xf32>) permutation = [0, 1, 3, 4, 2]
// TILE16:      %[[FILTER:.+]] = linalg.transpose ins(%{{.*}} : tensor<1x16x16x16x3x3xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<1x16x3x3x16x16xf32>) permutation = [0, 2, 4, 5, 3, 1]
// TILE16:      %[[OUT:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x16x14x14xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<8x14x14x16xf32>) permutation = [0, 2, 3, 1]
// TILE16:      %[[OUT_EXPAND:.+]] = tensor.expand_shape %[[OUT]]
// TILE16:      %[[TILED_CONV:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// TILE16-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction", "parallel"]}
// TILE16:         ins(%[[IMG]], %[[FILTER]] : tensor<8x16x16x16x16xf32>, tensor<1x16x3x3x16x16xf32>)
// TILE16:         outs(%[[OUT_EXPAND]] : tensor<8x1x14x14x16xf32>) {
// TILE16:        ^bb0
// TILE16:          arith.mulf
// TILE16:          arith.addf
// TILE16:        } -> tensor<8x1x14x14x16xf32>
// TILE16:      %[[RES_COLLAPSE:.+]] = tensor.collapse_shape %[[TILED_CONV:.+]]
// TILE16:      linalg.transpose ins(%[[RES_COLLAPSE]] : tensor<8x14x14x16xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<8x16x14x14xf32>) permutation = [0, 3, 1, 2]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  util.func public @generic_conv_nchw(%arg0: tensor<8x256x16x16xf32>, %arg1: tensor<64x256x3x3xf32>, %arg2: tensor<8x64x14x14xf32>) -> tensor<8x64x14x14xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%arg0, %arg1 : tensor<8x256x16x16xf32>, tensor<64x256x3x3xf32>) outs(%arg2 : tensor<8x64x14x14xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<8x64x14x14xf32>
    util.return %0 : tensor<8x64x14x14xf32>
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>


// CHECK-LABEL: @generic_conv_nchw

// CHECK:      %[[IMG:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x256x16x16xf32>) outs(%{{.*}} : tensor<8x16x16x256xf32>)
// CHECK:      %[[FILTER:.+]] = linalg.transpose ins(%{{.*}} : tensor<64x256x3x3xf32>) outs(%{{.*}} : tensor<3x3x256x64xf32>)
// CHECK:      %[[OUT:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x64x14x14xf32>) outs(%{{.*}} : tensor<8x14x14x64xf32>)
// CHECK:      %[[CONV:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
// CHECK-SAME:   ins(%[[IMG]], %[[FILTER]] : tensor<8x16x16x256xf32>, tensor<3x3x256x64xf32>)
// CHECK-SAME:   outs(%[[OUT]] : tensor<8x14x14x64xf32>)
// CHECK:      linalg.transpose ins(%[[CONV]] : tensor<8x14x14x64xf32>) outs(%{{.*}} : tensor<8x64x14x14xf32>)
