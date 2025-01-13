// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-bubble-up-expand-shapes))" %s | FileCheck %s

util.func public @bubbble_expand_through_extract(%arg0 : tensor<2x4096x5120xf16>) -> (tensor<2x64x64x2560xf16>) {
  %extracted_slice_237 = tensor.extract_slice %arg0[0, 0, 0] [2, 4096, 2560] [1, 1, 1] : tensor<2x4096x5120xf16> to tensor<2x4096x2560xf16>
  %expanded_239 = tensor.expand_shape %extracted_slice_237 [[0], [1, 2], [3]] output_shape [2, 64, 64, 2560] : tensor<2x4096x2560xf16> into tensor<2x64x64x2560xf16>
  util.return %expanded_239 : tensor<2x64x64x2560xf16>
}

// CHECK-LABEL:  @bubbble_expand_through_extract
//       CHECK:    %[[EXPAND:.+]] = tensor.expand_shape
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice %[[EXPAND]]

// -----

util.func public @unsupported_bubbble_expand_through_extract(%arg0 : tensor<2x4096x5120xf16>) -> (tensor<2x32x64x2560xf16>) {
  %extracted_slice_237 = tensor.extract_slice %arg0[0, 0, 0] [2, 2048, 2560] [1, 1, 1] : tensor<2x4096x5120xf16> to tensor<2x2048x2560xf16>
  %expanded_239 = tensor.expand_shape %extracted_slice_237 [[0], [1, 2], [3]] output_shape [2, 32, 64, 2560] : tensor<2x2048x2560xf16> into tensor<2x32x64x2560xf16>
  util.return %expanded_239 : tensor<2x32x64x2560xf16>
}

// CHECK-LABEL:  @unsupported_bubbble_expand_through_extract
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:    %[[EXPAND:.+]] = tensor.expand_shape %[[EXTRACT]]

// -----

// Checks two things
// 1. Propagation of reshapes across attention operations
// 2. Use of folders to convert (expand(collapse)) -> (collapse)
util.func public @attention_v_reshape_propagation(%arg0: index,
    %arg1: tensor<4x8x4x128x?xf16>, %arg2: tensor<128x?x128xf16>,
    %arg3: tensor<128x?x128xf16>, %arg4: f16, %arg5: tensor<128x?x?xf16>)
    -> tensor<4x?x32x128xf16> {
  %0 = tensor.empty(%arg0) : tensor<4x?x32x128xf16>
  %1 = tensor.empty(%arg0) : tensor<128x?x128xf16>
  %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2], [3], [4]]
      : tensor<4x8x4x128x?xf16> into tensor<128x128x?xf16>
  %4 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]}
      ins(%arg2, %arg3, %collapsed, %arg4, %arg5
        : tensor<128x?x128xf16>, tensor<128x?x128xf16>, tensor<128x128x?xf16>,
          f16, tensor<128x?x?xf16>)
      outs(%1 : tensor<128x?x128xf16>) {
      ^bb0(%arg6: f32):
    iree_linalg_ext.yield %arg6 : f32
  } -> tensor<128x?x128xf16>
  %expanded = tensor.expand_shape %4 [[0, 1], [2], [3]]
      output_shape [4, 32, %arg0, 128]
      : tensor<128x?x128xf16> into tensor<4x32x?x128xf16>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<4x32x?x128xf16>) outs(%0 : tensor<4x?x32x128xf16>) {                                                                                                                                        ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x?x32x128xf16>
  util.return %5 : tensor<4x?x32x128xf16>
}
// CHECK-LABEL: func public @attention_v_reshape_propagation
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<4x8x4x128x?xf16>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}}, %[[ARG1]],
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ATTENTION]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[GENERIC]]
//       CHECK:   return %[[COLLAPSE]]
