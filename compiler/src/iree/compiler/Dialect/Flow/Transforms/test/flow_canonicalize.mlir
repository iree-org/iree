// RUN: iree-opt --iree-flow-canonicalize %s --split-input-file | FileCheck %s

util.func public @merge_constant_padding(%arg0: tensor<2x3xf32>, %pad_value: f32) -> tensor<7x8xf32> {
  %pad0 = tensor.pad %arg0 low[1, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<2x3xf32> to tensor<4x4xf32>
  %pad1 = tensor.pad %pad0 low[0, 2] high[3, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value : f32
    } : tensor<4x4xf32> to tensor<7x8xf32>
  util.return %pad1 : tensor<7x8xf32>
}
// CHECK-LABEL: util.func public @merge_constant_padding
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<2x3xf32>
//  CHECK-SAME:   %[[PADVAL:[A-Za-z0-9]+]]: f32
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]] low[1, 3] high[4, 2]
//       CHECK:     tensor.yield %[[PADVAL]]
//       CHECK:   util.return %[[PAD]]
