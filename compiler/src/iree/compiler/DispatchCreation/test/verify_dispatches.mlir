// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-verify-dispatches))" %s

util.func public @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<2x32xf32>
  %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<2x32x10x16384xf16>
  %collapsed = tensor.collapse_shape %1 [[0, 1, 2, 3]] : tensor<2x32x10x16384xf16> into tensor<10485760xf16>
  %collapsed_0 = tensor.collapse_shape %0 [[0, 1]] : tensor<2x32xf32> into tensor<64xf32>
  // expected-error @+1 {{contains a non-tileable op in a dispatch between tileable ops}}
  %2 = flow.dispatch.region -> (tensor<64xf32>) {
    %4 = tensor.empty() : tensor<10485760xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed : tensor<10485760xf16>) outs(%4 : tensor<10485760xf32>) {
    ^bb0(%in: f16, %out: f32):
      %9 = arith.extf %in : f16 to f32
      linalg.yield %9 : f32
    } -> tensor<10485760xf32>
    // expected-note-re @+1 {{non-tileable op:}}
    %expanded_1 = tensor.expand_shape %5 [[0, 1]] output_shape [64, 163840] : tensor<10485760xf32> into tensor<64x163840xf32>
    %6 = tensor.empty() : tensor<64xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%expanded_1, %collapsed_0 : tensor<64x163840xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.subf %in, %in_2 : f32
      %10 = arith.mulf %9, %9 : f32
      %11 = arith.addf %10, %out : f32
      linalg.yield %11 : f32
    } -> tensor<64xf32>
    flow.return %8 : tensor<64xf32>
  }
  %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [2, 32] : tensor<64xf32> into tensor<2x32xf32>
  %3 = hal.tensor.export %expanded "output0" : tensor<2x32xf32> -> !hal.buffer_view
  util.return %3 : !hal.buffer_view
}
