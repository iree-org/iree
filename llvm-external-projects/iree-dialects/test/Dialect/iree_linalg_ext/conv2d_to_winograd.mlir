// RUN: iree-dialects-opt --split-input-file -iree-linalg-ext-convert-conv2d-to-winograd -mlir-elide-elementsattrs-if-larger=4 %s | FileCheck %s

func.func @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %c0 = arith.constant dense<0.1> : tensor<3x3x4x16xf32>
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %c0: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  return %0 : tensor<1x14x14x16xf32>
}
// CHECK:      func.func @conv_16433136(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
// CHECK:        %[[CST:.+]] = arith.constant dense_resource<__elided__> : tensor<64x4x16xf32>
// CHECK:        %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x1x3x3x4xf32>
// CHECK:        %[[D1:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<1x16x16x4xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<8x8x1x3x3x4xf32>) -> tensor<8x8x1x3x3x4xf32>
// CHECK:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[D1]]
// CHECK-SAME{LITERAL}:  [[0, 1], [2, 3, 4], [5]]
// CHECK-SAME:           tensor<8x8x1x3x3x4xf32> into tensor<64x9x4xf32>
// CHECK:        %[[D2:.+]] = tensor.empty() : tensor<64x9x16xf32>
// CHECK:        %[[D3:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D2]] : tensor<64x9x16xf32>) ->
// CHECK-SAME:     tensor<64x9x16xf32>
// CHECK:        %[[D4:.+]] = linalg.batch_matmul ins(%[[COLLAPSED]], %[[CST]] : tensor<64x9x4xf32>,
// CHECK-SAME:     tensor<64x4x16xf32>) outs(%[[D3]] : tensor<64x9x16xf32>) -> tensor<64x9x16xf32>
// CHECK:        %[[EXPANDED:.+]] = tensor.expand_shape %[[D4]]
// CHECK-SAME{LITERAL}: [[0, 1], [2, 3, 4], [5]]
// CHECK-SAME:          tensor<64x9x16xf32> into tensor<8x8x1x3x3x16xf32>
// CHECK:        %[[D5:.+]] = tensor.empty() : tensor<1x18x18x16xf32>
// CHECK:        %[[D6:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[EXPANDED]] : tensor<8x8x1x3x3x16xf32>) outs(%[[D5]] :
// CHECK-SAME:     tensor<1x18x18x16xf32>) -> tensor<1x18x18x16xf32>
// CHECK:        %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D6]][0, 0, 0, 0] [1, 14, 14, 16] [1, 1, 1, 1] :
// CHECK-SAME:     tensor<1x18x18x16xf32> to tensor<1x14x14x16xf32>
// CHECK:        return %[[EXTRACTED_SLICE]] : tensor<1x14x14x16xf32>
// CHECK:      }

// -----

func.func @conv2d_non_splat_weights(%inputs : tensor<1x4x4x1xf32>, %arg2: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32> {
  %c0 = arith.constant dense<[[ [[1.0]],  [[3.0]],  [[5.0]]  ],
                              [ [[7.0]],  [[9.0]],  [[11.0]] ],
                              [ [[13.0]], [[15.0]], [[17.0]] ]]> : tensor<3x3x1x1xf32>
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%inputs, %c0: tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>)
    outs(%arg2: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %0 : tensor<1x2x2x1xf32>
}
// CHECK:      func.func @conv2d_non_splat_weights(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x4x4x1xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32> {
// CHECK:        %[[CST:.+]] = arith.constant dense_resource<__elided__> : tensor<64x1x1xf32>
// CHECK:        %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x1x1x1x1xf32>
// CHECK:        %[[D1:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<1x4x4x1xf32>) outs(%[[D0]] : tensor<8x8x1x1x1x1xf32>)
// CHECK-SAME:     -> tensor<8x8x1x1x1x1xf32>
// CHECK:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[D1]]
// CHECK-SAME{LITERAL}:   [[0, 1], [2, 3, 4], [5]]
// CHECK-SAME:            tensor<8x8x1x1x1x1xf32> into tensor<64x1x1xf32>
// CHECK:        %[[D2:.+]] = tensor.empty() : tensor<64x1x1xf32>
// CHECK:        %[[D3:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D2]] : tensor<64x1x1xf32>) ->
// CHECK-SAME:     tensor<64x1x1xf32>
// CHECK:        %[[D4:.+]] = linalg.batch_matmul ins(%[[COLLAPSED]], %[[CST]] : tensor<64x1x1xf32>, tensor<64x1x1xf32>)
// CHECK-SAME:     outs(%[[D3]] : tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
// CHECK:        %[[EXPANDED:.+]] = tensor.expand_shape %[[D4]]
// CHECK-SAME{LITERAL}:   [[0, 1], [2, 3, 4], [5]]
// CHECK-SAME:            tensor<64x1x1xf32> into tensor<8x8x1x1x1x1xf32>
// CHECK:        %[[D5:.+]] = tensor.empty() : tensor<1x6x6x1xf32>
// CHECK:        %[[D6:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[EXPANDED]] : tensor<8x8x1x1x1x1xf32>) outs(%[[D5]] :
// CHECK-SAME:     tensor<1x6x6x1xf32>) -> tensor<1x6x6x1xf32>
// CHECK:        %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D6]][0, 0, 0, 0] [1, 2, 2, 1] [1, 1, 1, 1] :
// CHECK-SAME:     tensor<1x6x6x1xf32> to tensor<1x2x2x1xf32>
// CHECK:        return %[[EXTRACTED_SLICE]] : tensor<1x2x2x1xf32>
// CHECK:      }

// -----

func.func @conv_16433136_nchw_fchw(%arg0: tensor<1x4x16x16xf32>, %arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32> {
  %c0 = arith.constant dense<0.1> : tensor<16x4x3x3xf32>
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %c0: tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
    outs(%arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
  return %0 : tensor<1x16x14x14xf32>
}
// CHECK:      func.func @conv_16433136_nchw_fchw(%[[ARG0]]: tensor<1x4x16x16xf32>, %[[ARG1]]: tensor<1x16x14x14xf32>)
// CHECK-SAME:   -> tensor<1x16x14x14xf32> {
// CHECK:        %[[CST]] = arith.constant dense_resource<__elided__> : tensor<64x4x16xf32>
// CHECK:        %[[CST_0]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0]] = tensor.empty() : tensor<8x8x1x3x3x4xf32>
// CHECK:        %[[D1]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([2, 3]) ins(%[[ARG0]] : tensor<1x4x16x16xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<8x8x1x3x3x4xf32>) -> tensor<8x8x1x3x3x4xf32>
// CHECK:        %[[COLLAPSED]] = tensor.collapse_shape %[[D1]]
// CHECK:        %[[D2]] = tensor.empty() : tensor<64x9x16xf32>
// CHECK:        %[[D3]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D2]] : tensor<64x9x16xf32>) -> tensor<64x9x16xf32>
// CHECK:        %[[D4]] = linalg.batch_matmul ins(%[[COLLAPSED]], %[[CST]] : tensor<64x9x4xf32>, tensor<64x4x16xf32>)
// CHECK-SAME:     outs(%[[D3]] : tensor<64x9x16xf32>) -> tensor<64x9x16xf32>
// CHECK:        %[[EXPANDED]] = tensor.expand_shape %[[D4]]
// CHECK:        %[[D5]] = tensor.empty() : tensor<1x16x18x18xf32>
// CHECK:        %[[D6]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([2, 3]) ins(%[[EXPANDED]] : tensor<8x8x1x3x3x16xf32>) outs(%[[D5]] :
// CHECK-SAME:     tensor<1x16x18x18xf32>) -> tensor<1x16x18x18xf32>
// CHECK:        %[[EXTRACTED_SLICE]] = tensor.extract_slice %[[D6]][0, 0, 0, 0] [1, 16, 14, 14] [1, 1, 1, 1] :
// CHECK-SAME:     tensor<1x16x18x18xf32> to tensor<1x16x14x14xf32>
// CHECK:        return %[[EXTRACTED_SLICE]] : tensor<1x16x14x14xf32>
// CHECK:      }
// CHECK:    }

// -----

func.func @conv2d_nchw_non_splat_weights(%inputs : tensor<1x1x4x4xf32>, %arg2: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
  %c0 = arith.constant dense<[[[[ 1.0,  3.0,  5.0  ],
                                [ 7.0,  9.0,  11.0 ],
                                [ 13.0, 15.0, 17.0 ]]]]> : tensor<1x1x3x3xf32>
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%inputs, %c0: tensor<1x1x4x4xf32>, tensor<1x1x3x3xf32>)
    outs(%arg2: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}
// CHECK:      func.func @conv2d_nchw_non_splat_weights(%[[ARG0]]: tensor<1x1x4x4xf32>, %[[ARG1]]: tensor<1x1x2x2xf32>)
// CHECK-SAME:   -> tensor<1x1x2x2xf32> {
// CHECK:        %[[CST]] = arith.constant dense_resource<__elided__> : tensor<64x1x1xf32>
// CHECK:        %[[CST_0]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0]] = tensor.empty() : tensor<8x8x1x1x1x1xf32>
// CHECK:        %[[D1]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([2, 3]) ins(%[[ARG0]] : tensor<1x1x4x4xf32>) outs(%[[D0]] : tensor<8x8x1x1x1x1xf32>)
// CHECK-SAME:     -> tensor<8x8x1x1x1x1xf32>
// CHECK:        %[[COLLAPSED]] = tensor.collapse_shape %[[D1]]
// CHECK:        %[[D2]] = tensor.empty() : tensor<64x1x1xf32>
// CHECK:        %[[D3]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D2]] : tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
// CHECK:        %[[D4]] = linalg.batch_matmul ins(%[[COLLAPSED]], %[[CST]] : tensor<64x1x1xf32>, tensor<64x1x1xf32>)
// CHECK-SAME:     outs(%[[D3]] : tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
// CHECK:        %[[EXPANDED]] = tensor.expand_shape %[[D4]]
// CHECK:        %[[D5]] = tensor.empty() : tensor<1x1x6x6xf32>
// CHECK:        %[[D6]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([2, 3]) ins(%[[EXPANDED]] : tensor<8x8x1x1x1x1xf32>) outs(%[[D5]] :
// CHECK-SAME:     tensor<1x1x6x6xf32>) -> tensor<1x1x6x6xf32>
// CHECK:        %[[EXTRACTED_SLICE]] = tensor.extract_slice %[[D6]][0, 0, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] :
// CHECK-SAME:     tensor<1x1x6x6xf32> to tensor<1x1x2x2xf32>
// CHECK:        return %[[EXTRACTED_SLICE]] : tensor<1x1x2x2xf32>
// CHECK:      }
