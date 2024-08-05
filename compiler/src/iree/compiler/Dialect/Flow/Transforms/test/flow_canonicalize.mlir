// RUN: iree-opt --iree-flow-canonicalize %s --split-input-file --mlir-print-local-scope | FileCheck %s

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

// -----

util.func public @merge_constant_padding_dynamic(%arg0: tensor<?x?xf32>, %idx: index, %pad_value: f32) -> tensor<?x?xf32> {
  %pad0 = tensor.pad %arg0 low[%idx, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  %pad1 = tensor.pad %pad0 low[0, 2] high[%idx, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  util.return %pad1 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @merge_constant_padding_dynamic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[IDX:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[PADVAL:[A-Za-z0-9]+]]: f32
//       CHECK:   %[[HIGH:.+]] = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%[[IDX]]]
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]] low[%[[IDX]], 3] high[%[[HIGH]], 2]
//       CHECK:     tensor.yield %[[PADVAL]]
//       CHECK:   util.return %[[PAD]]

// -----

util.func public @dont_merge_constant_padding_nofold(%arg0: tensor<2x3xf32>, %pad_value: f32) -> tensor<7x8xf32> {
  %pad0 = tensor.pad %arg0 low[1, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<2x3xf32> to tensor<4x4xf32>
  %pad1 = tensor.pad %pad0 nofold low[0, 2] high[3, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value : f32
    } : tensor<4x4xf32> to tensor<7x8xf32>
  util.return %pad1 : tensor<7x8xf32>
}

// Verify that folding does not happen if it would drop a nofold attribute

// CHECK-LABEL: util.func public @dont_merge_constant_padding_nofold
//       CHECK:   tensor.pad
//       CHECK:   tensor.pad {{.*}} nofold

// -----

util.func public @dont_merge_constant_padding_different_vals(
    %arg0: tensor<2x3xf32>,
    %pad_value0: f32,
    %pad_value1: f32) -> tensor<7x8xf32> {
  %pad0 = tensor.pad %arg0 low[1, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value0 : f32
    } : tensor<2x3xf32> to tensor<4x4xf32>
  %pad1 = tensor.pad %pad0 nofold low[0, 2] high[3, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value1 : f32
    } : tensor<4x4xf32> to tensor<7x8xf32>
  util.return %pad1 : tensor<7x8xf32>
}

// Verify that folding does not happen if it would drop a nofold attribute

// CHECK-LABEL: util.func public @dont_merge_constant_padding_different_vals
//       CHECK:   tensor.pad
//       CHECK:   tensor.pad