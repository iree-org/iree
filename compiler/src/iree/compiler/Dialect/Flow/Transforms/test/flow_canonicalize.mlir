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

// -----

util.func public @tensor_cast_to_reshape(%reshape_17 : tensor<?x?x?x?xf32>, %65 : tensor<?x12x?x64xf32>, %0 : index, %1 : index) -> tensor<?x?x?x?xf32> {
  %cast = tensor.cast %reshape_17 : tensor<?x?x?x?xf32> to tensor<?x?x12x64xf32>
  %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%cast : tensor<?x?x12x64xf32>) outs(%65 : tensor<?x12x?x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x12x?x64xf32>
  %cast_18 = tensor.cast %66 : tensor<?x12x?x64xf32> to tensor<?x?x?x?xf32>
  util.return  %cast_18 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: util.func public @tensor_cast_to_reshape
//       CHECK:   flow.tensor.reshape
//       CHECK-SAME: tensor<?x?x?x?xf32>
//       CHECK-SAME: -> tensor<?x?x12x64xf32>
//       CHECK:   linalg.generic
//       CHECK:   flow.tensor.reshape
//       CHECK-SAME: tensor<?x12x?x64xf32>
//       CHECK-SAME: -> tensor<?x?x?x?xf32>
