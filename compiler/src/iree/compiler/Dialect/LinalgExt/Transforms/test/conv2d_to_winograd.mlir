// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-linalg-ext-convert-conv2d-to-winograd{replace-all-convs}))" --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK,CHECK-OUTER
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-linalg-ext-convert-conv2d-to-winograd{replace-all-convs inner-input-tile}))" --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK,CHECK-INNER
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-linalg-ext-convert-conv2d-to-winograd))" --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-ANNOTATED

util.func public @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-ALL:        util.func public @conv_16433136(
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf32>
// CHECK-DAG:          %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-OUTER-DAG:    %[[EMPTY0:.+]] = tensor.empty() : tensor<8x8x4x16xf32>
// CHECK-INNER-DAG:    %[[EMPTY0:.+]] = tensor.empty() : tensor<4x16x8x8xf32>
// CHECK:              %[[FILTER_TF:.+]] = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[ARG1]] : tensor<3x3x4x16xf32>) outs(%[[EMPTY0]] :
// CHECK-INNER-SAME:     input_tile_dimensions([2, 3]) ins(%[[ARG1]] : tensor<3x3x4x16xf32>) outs(%[[EMPTY0]] :
// CHECK-OUTER-SAME:     tensor<8x8x4x16xf32>) -> tensor<8x8x4x16xf32>
// CHECK-INNER-SAME:     tensor<4x16x8x8xf32>) -> tensor<4x16x8x8xf32>
// CHECK:              %[[COLLAPSED_FILTER:.+]] = tensor.collapse_shape %[[FILTER_TF]]
// CHECK-OUTER-SAME{LITERAL}: [[0, 1], [2], [3]]
// CHECK-INNER-SAME{LITERAL}: [[0], [1], [2, 3]]
// CHECK-OUTER-SAME:           tensor<8x8x4x16xf32> into tensor<64x4x16xf32>
// CHECK-INNER-SAME:           tensor<4x16x8x8xf32> into tensor<4x16x64xf32>
// CHECK-OUTER:        %[[EMPTY1:.+]] = tensor.empty() : tensor<8x8x1x3x3x4xf32>
// CHECK-INNER:        %[[EMPTY1:.+]] = tensor.empty() : tensor<1x3x3x4x8x8xf32>
// CHECK:              %[[INPUT_TF:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[ARG0]] : tensor<1x16x16x4xf32>) outs(%[[EMPTY1]] :
// CHECK-INNER-SAME:     input_tile_dimensions([4, 5]) ins(%[[ARG0]] : tensor<1x16x16x4xf32>) outs(%[[EMPTY1]] :
// CHECK-OUTER-SAME:     tensor<8x8x1x3x3x4xf32>) -> tensor<8x8x1x3x3x4xf32>
// CHECK-INNER-SAME:     tensor<1x3x3x4x8x8xf32>) -> tensor<1x3x3x4x8x8xf32>
// CHECK:              %[[COLLAPSED_INPUT:.+]] = tensor.collapse_shape %[[INPUT_TF]]
// CHECK-OUTER-SAME{LITERAL}:  [[0, 1], [2, 3, 4], [5]]
// CHECK-INNER-SAME{LITERAL}:  [[0, 1, 2], [3], [4, 5]]
// CHECK-OUTER-SAME:           tensor<8x8x1x3x3x4xf32> into tensor<64x9x4xf32>
// CHECK-INNER-SAME:           tensor<1x3x3x4x8x8xf32> into tensor<9x4x64xf32>
// CHECK-OUTER:        %[[EMPTY2:.+]] = tensor.empty() : tensor<64x9x16xf32>
// CHECK-INNER:        %[[EMPTY2:.+]] = tensor.empty() : tensor<9x16x64xf32>
// CHECK:              %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY2]]
// CHECK:              %[[BMM:.+]] = linalg.generic
// CHECK-OUTER-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
// CHECK-INNER-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d0)>]
// CHECK-SAME:           iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:           ins(%[[COLLAPSED_INPUT]], %[[COLLAPSED_FILTER]]
// CHECK-SAME:           outs(%[[FILL]]
// CHECK:              %[[EXPANDED:.+]] = tensor.expand_shape %[[BMM]]
// CHECK-OUTER-SAME{LITERAL}: [[0, 1], [2, 3, 4], [5]]
// CHECK-INNER-SAME{LITERAL}: [[0, 1, 2], [3], [4, 5]]
// CHECK-OUTER-SAME:          tensor<64x9x16xf32> into tensor<8x8x1x3x3x16xf32>
// CHECK-INNER-SAME:          tensor<9x16x64xf32> into tensor<1x3x3x16x8x8xf32>
// CHECK:              %[[EMPTY3:.+]] = tensor.empty() : tensor<1x18x18x16xf32>
// CHECK:              %[[OUTPUT_TF:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[EXPANDED]] : tensor<8x8x1x3x3x16xf32>) outs(%[[EMPTY3]] :
// CHECK-INNER-SAME:     input_tile_dimensions([4, 5]) ins(%[[EXPANDED]] : tensor<1x3x3x16x8x8xf32>) outs(%[[EMPTY3]] :
// CHECK-SAME:           tensor<1x18x18x16xf32>) -> tensor<1x18x18x16xf32>
// CHECK:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[OUTPUT_TF]][0, 0, 0, 0] [1, 14, 14, 16] [1, 1, 1, 1] :
// CHECK-SAME:           tensor<1x18x18x16xf32> to tensor<1x14x14x16xf32>
// CHECK:              util.return %[[EXTRACTED_SLICE]] : tensor<1x14x14x16xf32>
// CHECK:            }

// -----

util.func public @conv_16433136_nchw_fchw(%arg0: tensor<1x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32> {
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
    outs(%arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
  util.return %0 : tensor<1x16x14x14xf32>
}
// CHECK-ALL:        util.func public @conv_16433136_nchw_fchw(
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x4x16x16xf32>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9_]+]]: tensor<16x4x3x3xf32>
// CHECK-DAG:          %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-OUTER-DAG:    %[[EMPTY0:.+]] = tensor.empty() : tensor<8x8x4x16xf32>
// CHECK-INNER-DAG:    %[[EMPTY0:.+]] = tensor.empty() : tensor<4x16x8x8xf32>
// CHECK:              %[[FILTER_TF:.+]] = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([2, 3])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[ARG1]] : tensor<16x4x3x3xf32>) outs(%[[EMPTY0]] :
// CHECK-INNER-SAME:     input_tile_dimensions([2, 3]) ins(%[[ARG1]] : tensor<16x4x3x3xf32>) outs(%[[EMPTY0]] :
// CHECK-OUTER-SAME:     tensor<8x8x4x16xf32>) -> tensor<8x8x4x16xf32>
// CHECK-INNER-SAME:     tensor<4x16x8x8xf32>) -> tensor<4x16x8x8xf32>
// CHECK:              %[[COLLAPSED_FILTER:.+]] = tensor.collapse_shape %[[FILTER_TF]]
// CHECK-OUTER-SAME{LITERAL}: [[0, 1], [2], [3]]
// CHECK-INNER-SAME{LITERAL}: [[0], [1], [2, 3]]
// CHECK-OUTER-SAME:           tensor<8x8x4x16xf32> into tensor<64x4x16xf32>
// CHECK-INNER-SAME:           tensor<4x16x8x8xf32> into tensor<4x16x64xf32>
// CHECK-OUTER:        %[[EMPTY1:.+]] = tensor.empty() : tensor<8x8x1x3x3x4xf32>
// CHECK-INNER:        %[[EMPTY1:.+]] = tensor.empty() : tensor<1x3x3x4x8x8xf32>
// CHECK:              %[[INPUT_TF:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[ARG0]] : tensor<1x4x16x16xf32>) outs(%[[EMPTY1]] :
// CHECK-INNER-SAME:     input_tile_dimensions([4, 5]) ins(%[[ARG0]] : tensor<1x4x16x16xf32>) outs(%[[EMPTY1]] :
// CHECK-OUTER-SAME:     tensor<8x8x1x3x3x4xf32>) -> tensor<8x8x1x3x3x4xf32>
// CHECK-INNER-SAME:     tensor<1x3x3x4x8x8xf32>) -> tensor<1x3x3x4x8x8xf32>
// CHECK:              %[[COLLAPSED_INPUT:.+]] = tensor.collapse_shape %[[INPUT_TF]]
// CHECK-OUTER-SAME{LITERAL}: [[0, 1], [2, 3, 4], [5]]
// CHECK-INNER-SAME{LITERAL}: [[0, 1, 2], [3], [4, 5]]
// CHECK-OUTER-SAME:           tensor<8x8x1x3x3x4xf32> into tensor<64x9x4xf32>
// CHECK-INNER-SAME:           tensor<1x3x3x4x8x8xf32> into tensor<9x4x64xf32>
// CHECK-OUTER:        %[[EMPTY2:.+]] = tensor.empty() : tensor<64x9x16xf32>
// CHECK-INNER:        %[[EMPTY2:.+]] = tensor.empty() : tensor<9x16x64xf32>
// CHECK:              %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY2]]
// CHECK:              %[[BMM:.+]] = linalg.generic
// CHECK-OUTER-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
// CHECK-INNER-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d0)>]
// CHECK-SAME:           iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:           ins(%[[COLLAPSED_INPUT]], %[[COLLAPSED_FILTER]]
// CHECK-SAME:           outs(%[[FILL]]
// CHECK:              %[[EXPANDED:.+]] = tensor.expand_shape %[[BMM]]
// CHECK-OUTER-SAME{LITERAL}: [[0, 1], [2, 3, 4], [5]]
// CHECK-INNER-SAME{LITERAL}: [[0, 1, 2], [3], [4, 5]]
// CHECK-OUTER-SAME:          tensor<64x9x16xf32> into tensor<8x8x1x3x3x16xf32>
// CHECK-INNER-SAME:          tensor<9x16x64xf32> into tensor<1x3x3x16x8x8xf32>
// CHECK:              %[[EMPTY3:.+]] = tensor.empty() : tensor<1x16x18x18xf32>
// CHECK:              %[[OUTPUT_TF:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[EXPANDED]] : tensor<8x8x1x3x3x16xf32>) outs(%[[EMPTY3]] :
// CHECK-INNER-SAME:     input_tile_dimensions([4, 5]) ins(%[[EXPANDED]] : tensor<1x3x3x16x8x8xf32>) outs(%[[EMPTY3]] :
// CHECK-SAME:           tensor<1x16x18x18xf32>) -> tensor<1x16x18x18xf32>
// CHECK:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[OUTPUT_TF]][0, 0, 0, 0] [1, 16, 14, 14] [1, 1, 1, 1] :
// CHECK-SAME:           tensor<1x16x18x18xf32> to tensor<1x16x14x14xf32>
// CHECK:              util.return %[[EXTRACTED_SLICE]] : tensor<1x16x14x14xf32>
// CHECK:            }

// -----

util.func public @conv_mixed_types(%arg0: tensor<1x16x16x4xf16>, %arg1: tensor<3x3x4x16xf16>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf16>, tensor<3x3x4x16xf16>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-ALL:        util.func public @conv_mixed_types(
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf16>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf16>
// CHECK-DAG:          %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-OUTER-DAG:    %[[EMPTY0:.+]] = tensor.empty() : tensor<8x8x4x16xf16>
// CHECK-INNER-DAG:    %[[EMPTY0:.+]] = tensor.empty() : tensor<4x16x8x8xf16>
// CHECK:              %[[FILTER_TF:.+]] = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[ARG1]] : tensor<3x3x4x16xf16>) outs(%[[EMPTY0]] :
// CHECK-INNER-SAME:     input_tile_dimensions([2, 3]) ins(%[[ARG1]] : tensor<3x3x4x16xf16>) outs(%[[EMPTY0]] :
// CHECK-OUTER-SAME:     tensor<8x8x4x16xf16>) -> tensor<8x8x4x16xf16>
// CHECK-INNER-SAME:     tensor<4x16x8x8xf16>) -> tensor<4x16x8x8xf16>
// CHECK:              %[[COLLAPSED_FILTER:.+]] = tensor.collapse_shape %[[FILTER_TF]]
// CHECK-OUTER-SAME{LITERAL}:  [[0, 1], [2], [3]]
// CHECK-INNER-SAME{LITERAL}:  [[0], [1], [2, 3]]
// CHECK-OUTER-SAME:           tensor<8x8x4x16xf16> into tensor<64x4x16xf16>
// CHECK-INNER-SAME:           tensor<4x16x8x8xf16> into tensor<4x16x64xf16>
// CHECK-OUTER:        %[[EMPTY1:.+]] = tensor.empty() : tensor<8x8x1x3x3x4xf16>
// CHECK-INNER:        %[[EMPTY1:.+]] = tensor.empty() : tensor<1x3x3x4x8x8xf16>
// CHECK:              %[[INPUT_TF:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[ARG0]] : tensor<1x16x16x4xf16>) outs(%[[EMPTY1]] :
// CHECK-INNER-SAME:     input_tile_dimensions([4, 5]) ins(%[[ARG0]] : tensor<1x16x16x4xf16>) outs(%[[EMPTY1]] :
// CHECK-OUTER-SAME:     tensor<8x8x1x3x3x4xf16>) -> tensor<8x8x1x3x3x4xf16>
// CHECK-INNER-SAME:     tensor<1x3x3x4x8x8xf16>) -> tensor<1x3x3x4x8x8xf16>
// CHECK:              %[[COLLAPSED_INPUT:.+]] = tensor.collapse_shape %[[INPUT_TF]]
// CHECK-OUTER-SAME{LITERAL}:  [[0, 1], [2, 3, 4], [5]]
// CHECK-INNER-SAME{LITERAL}:  [[0, 1, 2], [3], [4, 5]]
// CHECK-OUTER-SAME:           tensor<8x8x1x3x3x4xf16> into tensor<64x9x4xf16>
// CHECK-INNER-SAME:           tensor<1x3x3x4x8x8xf16> into tensor<9x4x64xf16>
// CHECK-OUTER:        %[[EMPTY2:.+]] = tensor.empty() : tensor<64x9x16xf32>
// CHECK-INNER:        %[[EMPTY2:.+]] = tensor.empty() : tensor<9x16x64xf32>
// CHECK:              %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY2]]
// CHECK:              %[[BMM:.+]] = linalg.generic
// CHECK-OUTER-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
// CHECK-INNER-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d0)>]
// CHECK-SAME:           iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:           ins(%[[COLLAPSED_INPUT]], %[[COLLAPSED_FILTER]]
// CHECK-SAME:           outs(%[[FILL]]
// CHECK:              %[[EXPANDED:.+]] = tensor.expand_shape %[[BMM]]
// CHECK-OUTER-SAME{LITERAL}: [[0, 1], [2, 3, 4], [5]]
// CHECK-INNER-SAME{LITERAL}: [[0, 1, 2], [3], [4, 5]]
// CHECK-OUTER-SAME:          tensor<64x9x16xf32> into tensor<8x8x1x3x3x16xf32>
// CHECK-INNER-SAME:          tensor<9x16x64xf32> into tensor<1x3x3x16x8x8xf32>
// CHECK:              %[[EMPTY3:.+]] = tensor.empty() : tensor<1x18x18x16xf32>
// CHECK:              %[[OUTPUT_TF:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-OUTER-SAME:     input_tile_dimensions([0, 1]) ins(%[[EXPANDED]] : tensor<8x8x1x3x3x16xf32>) outs(%[[EMPTY3]] :
// CHECK-INNER-SAME:     input_tile_dimensions([4, 5]) ins(%[[EXPANDED]] : tensor<1x3x3x16x8x8xf32>) outs(%[[EMPTY3]] :
// CHECK-SAME:           tensor<1x18x18x16xf32>) -> tensor<1x18x18x16xf32>
// CHECK:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[OUTPUT_TF]][0, 0, 0, 0] [1, 14, 14, 16] [1, 1, 1, 1] :
// CHECK-SAME:           tensor<1x18x18x16xf32> to tensor<1x14x14x16xf32>
// CHECK:              util.return %[[EXTRACTED_SLICE]] : tensor<1x14x14x16xf32>
// CHECK:            }

// -----

util.func public @conv_rewrite_annotated(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, __winograd_conv}
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-ALL:        util.func public @conv_rewrite_annotated(
// CHECK-ANNOTATED:    iree_linalg_ext.winograd.filter_transform
// CHECK-ANNOTATED:    iree_linalg_ext.winograd.input_transform
// CHECK-ANNOTATED:    linalg.generic
// CHECK-ANNOTATED:    iree_linalg_ext.winograd.output_transform

// -----

util.func public @conv_skip_unannotated(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-ALL:            util.func public @conv_skip_unannotated(
// CHECK-ANNOTATED-NOT:    iree_linalg_ext.winograd.filter_transform
// CHECK-ANNOTATED-NOT:    iree_linalg_ext.winograd.input_transform
// CHECK-ANNOTATED-NOT:    linalg.generic
// CHECK-ANNOTATED-NOT:    iree_linalg_ext.winograd.output_transform
