// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-bubble-up-expand-shapes, canonicalize, cse, canonicalize))" %s | FileCheck %s

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
