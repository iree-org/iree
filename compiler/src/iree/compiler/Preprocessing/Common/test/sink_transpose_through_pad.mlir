// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-preprocessing-sink-transpose-through-pad))" --split-input-file %s | FileCheck %s

util.func public @sink_pad_through_transpose(%arg0 : tensor<16x64x64x128xf16>) -> (tensor<16x128x66x66xf16>) {
  %2 = tensor.empty() : tensor<16x128x64x64xf16>
  %cst = arith.constant 0.000000e+00 : f16
  %transposed = linalg.transpose ins(%arg0 : tensor<16x64x64x128xf16>) outs(%2 : tensor<16x128x64x64xf16>) permutation = [0, 3, 1, 2]
  %padded = tensor.pad %transposed low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
    tensor.yield %cst : f16
  } : tensor<16x128x64x64xf16> to tensor<16x128x66x66xf16>
  util.return %padded : tensor<16x128x66x66xf16>
}
// CHECK-LABEL:  util.func public @sink_pad_through_transpose
//       CHECK:    %[[PAD:.+]] = tensor.pad
//       CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose
//  CHECK-SAME:      ins(%[[PAD]]
//       CHECK:    util.return %[[TRANSPOSE]]
