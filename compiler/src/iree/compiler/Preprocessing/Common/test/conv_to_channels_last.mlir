// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last))" %s | \
// RUN:   FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last{tiling-factor=16}))" %s | \
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
// TILE16:      %[[IMG:.+]] = linalg.pack {{.*}} inner_dims_pos = [1] inner_tiles = [16]
// TILE16-SAME:   tensor<8x256x16x16xf32> -> tensor<8x16x16x16x16xf32>
// TILE16:      %[[FILTER:.+]] = linalg.pack {{.*}} inner_dims_pos = [1, 0] inner_tiles = [16, 16]
// TILE16-SAME:   tensor<16x256x3x3xf32> -> tensor<1x16x3x3x16x16xf32>
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

// TILE16: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d7)>
// TILE16: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d7, d8)>
// TILE16: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d8)>

// TILE16-LABEL: util.func public @generic_conv_nchw
// TILE16:      %[[IMG:.+]] = linalg.pack {{.*}} inner_dims_pos = [1] inner_tiles = [16]
// TILE16-SAME:   tensor<8x256x16x16xf32> -> tensor<8x16x16x16x16xf32>
// TILE16:      %[[FILTER:.+]] = linalg.pack {{.*}} inner_dims_pos = [1, 0] inner_tiles = [16, 16]
// TILE16-SAME:   tensor<64x256x3x3xf32> -> tensor<4x16x3x3x16x16xf32>
// TILE16:      %[[OUT:.+]] = linalg.pack {{.*}} inner_dims_pos = [1] inner_tiles = [16]
// TILE16-SAME:   tensor<8x64x14x14xf32> -> tensor<8x4x14x14x16xf32>
// TILE16:      %[[TILED_CONV:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// TILE16-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction", "parallel"]}
// TILE16-SAME:    ins(%[[IMG]], %[[FILTER]] : tensor<8x16x16x16x16xf32>, tensor<4x16x3x3x16x16xf32>)
// TILE16-SAME:    outs(%[[OUT]] : tensor<8x4x14x14x16xf32>) {
// TILE16:      linalg.unpack %[[TILED_CONV]] inner_dims_pos = [1] inner_tiles = [16]
// TILE16-SAME:   tensor<8x4x14x14x16xf32> -> tensor<8x64x14x14xf32>

// -----

// Not a convolution, should not be transposed.

util.func @mmt_no_transpose(%arg0: tensor<2048x1280xf16>, %arg1: tensor<1280x1280xf16>) -> tensor<2048x1280xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2048x1280xf32>
  %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
  %res = linalg.matmul_transpose_b
    ins(%arg0, %arg1 : tensor<2048x1280xf16>, tensor<1280x1280xf16>)
    outs(%filled : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
  util.return %res : tensor<2048x1280xf32>
}
// CHECK-LABEL: @mmt_no_transpose
// CHECK-NOT:     linalg.generic
// CHECK:         linalg.matmul_transpose_b

// TILE16-LABEL: @mmt_no_transpose
// TILE16-NOT:     linalg.generic
// TILE16:         linalg.matmul_transpose_b


// -----

util.func @test_unit_dims_pack(%arg0: tensor<10x20x5xf32>) -> tensor<1x1x5x20x10xf32> {
  %dst = tensor.empty() : tensor<1x1x5x20x10xf32>
  %packed = linalg.pack %arg0 inner_dims_pos = [1, 0] inner_tiles = [20, 10]
    into %dst : tensor<10x20x5xf32> -> tensor<1x1x5x20x10xf32>

  util.return %packed : tensor<1x1x5x20x10xf32>
}

// CHECK-LABEL: @test_unit_dims_pack
// CHECK:       %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[ARG0:.+]] : tensor<10x20x5xf32>)
// CHECK-SAME:    outs(%[[DST:.+]] : tensor<5x20x10xf32>) permutation = [2, 1, 0]
// CHECK:       %[[EXPANDED:.+]] = tensor.expand_shape
// CHECK-SAME:    [0, 1, 2], [3], [4]
// CHECK-SAME:    tensor<5x20x10xf32> into tensor<1x1x5x20x10xf32>
// CHECK:       util.return %[[EXPANDED]] : tensor<1x1x5x20x10xf32>
