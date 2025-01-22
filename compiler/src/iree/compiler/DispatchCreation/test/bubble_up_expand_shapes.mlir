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

// -----

// Multiple uses of the producer

util.func @multiple_users(%arg0 : tensor<?x128xf16>,
    %arg1 : tensor<4x?x32x128xf16>) -> tensor<4x?x32x8x16xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x128xf16>
  %empty = tensor.empty(%dim) : tensor<4x?x32x128xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<?x128xf16>)
      outs(%empty : tensor<4x?x32x128xf16>) {
    ^bb0(%b0: f16, %b1 : f16) :
      %iv0 = linalg.index 0 : index
      %iv1 = linalg.index 0 : index
      %iv2 = linalg.index 0 : index
      %iv3 = linalg.index 0 : index
      // This is not technically a gather, but doing this way to mimic
      // use case of rope computation in LLaMa
      %1 = tensor.extract %arg1[%iv0, %iv1, %iv2, %iv3] : tensor<4x?x32x128xf16>
      %2 = arith.addf %1, %b0 : f16
      linalg.yield %2 : f16
  } -> tensor<4x?x32x128xf16>
  %1 = tensor.dim %0, %c1 : tensor<4x?x32x128xf16>
  %2 = tensor.expand_shape %0 [[0], [1], [2], [3, 4]] output_shape [4, %1, 32, 8, 16]
      : tensor<4x?x32x128xf16> into tensor<4x?x32x8x16xf16>
  util.return %2 : tensor<4x?x32x8x16xf16>
}
// CHECK-LABEL: func public @multiple_users(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x128xf16>
//       CHECK:   %[[EXPAND_SHAPE:.+]] = tensor.expand_shape %[[ARG0]]
//       CHECK:   %[[EXPANDED_GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[EXPAND_SHAPE]] :
//       CHECK:   return %[[EXPANDED_GENERIC]]
