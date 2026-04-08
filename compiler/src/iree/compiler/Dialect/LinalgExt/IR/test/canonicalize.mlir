// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

func.func @sort_drop_unused_results(%arg0 : tensor<?x10xf32>,
    %arg1 : tensor<?x10xi64>) -> tensor<?x10xf32> {
  %0:2 = iree_linalg_ext.sort dimension(1) outs(%arg0, %arg1: tensor<?x10xf32>,
      tensor<?x10xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %42 = arith.cmpf oge, %arg2, %arg3 : f32
    iree_linalg_ext.yield %42 : i1
  } -> tensor<?x10xf32>, tensor<?x10xi64>
  return %0#0 : tensor<?x10xf32>
}
// CHECK-LABEL: func.func @sort_drop_unused_results
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x10xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?x10xi64>
//       CHECK:   %[[SORT:.+]] = iree_linalg_ext.sort dimension(1) outs(%[[ARG0]] : tensor<?x10xf32>)

// -----

func.func @gather_to_extract_slice_expand(%source : tensor<1024x128xi32>, %indices : tensor<1x1xi32>) -> (tensor<1x1x128xi32>) {
  %empty = tensor.empty() : tensor<1x1x128xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
    ins(%source, %indices : tensor<1024x128xi32>, tensor<1x1xi32>)
    outs(%empty: tensor<1x1x128xi32>) -> tensor<1x1x128xi32>
  return %result : tensor<1x1x128xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_expand
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[IDX:.+]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]]]
//       CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST]], 0] [1, 128] [1, 1]
//  CHECK-SAME:     tensor<1024x128xi32> to tensor<1x128xi32>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]]
//  CHECK-SAME:     tensor<1x128xi32> into tensor<1x1x128xi32>

// -----

func.func @gather_to_extract_slice_no_reshape(%source : tensor<1024x128xi32>, %indices : tensor<1xi32>) -> (tensor<1x128xi32>) {
  %empty = tensor.empty() : tensor<1x128xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
    ins(%source, %indices : tensor<1024x128xi32>, tensor<1xi32>)
    outs(%empty: tensor<1x128xi32>) -> tensor<1x128xi32>
  return %result : tensor<1x128xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_no_reshape
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[IDX:.+]] = tensor.extract %[[ARG1]][%[[C0]]]
//       CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST]], 0] [1, 128] [1, 1]
//  CHECK-SAME:     tensor<1024x128xi32> to tensor<1x128xi32>

// -----

func.func @gather_to_extract_slice_perm(%source : tensor<10x1024x128xi32>, %indices : tensor<1x1x2xi32>) -> (tensor<1x1x128xi32>) {
  %empty = tensor.empty() : tensor<1x1x128xi32>
  %result = iree_linalg_ext.gather dimension_map = [1, 0]
    ins(%source, %indices : tensor<10x1024x128xi32>, tensor<1x1x2xi32>)
    outs(%empty: tensor<1x1x128xi32>) -> tensor<1x1x128xi32>
  return %result : tensor<1x1x128xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_perm
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[IDX0:.+]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]], %[[C0]]]
//   CHECK-DAG:   %[[IDX1:.+]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]], %[[C1]]]
//   CHECK-DAG:   %[[CAST0:.+]] = arith.index_cast %[[IDX0]]
//   CHECK-DAG:   %[[CAST1:.+]] = arith.index_cast %[[IDX1]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST1]], %[[CAST0]], 0] [1, 1, 128] [1, 1, 1]
//  CHECK-SAME:     tensor<10x1024x128xi32> to tensor<1x1x128xi32>

// -----

func.func @gather_to_extract_slice_full_collapse(%source : tensor<2x2x1xi32>, %indices : tensor<2xi32>) -> (tensor<1xi32>) {
  %empty = tensor.empty() : tensor<1xi32>
  %result = iree_linalg_ext.gather dimension_map = [0, 1]
    ins(%source, %indices : tensor<2x2x1xi32>, tensor<2xi32>)
    outs(%empty: tensor<1xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_full_collapse
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[IDX0:.+]] = tensor.extract %[[ARG1]][%[[C0]]]
//   CHECK-DAG:   %[[IDX1:.+]] = tensor.extract %[[ARG1]][%[[C1]]]
//   CHECK-DAG:   %[[CAST0:.+]] = arith.index_cast %[[IDX0]]
//   CHECK-DAG:   %[[CAST1:.+]] = arith.index_cast %[[IDX1]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST0]], %[[CAST1]], 0] [1, 1, 1] [1, 1, 1]
//  CHECK-SAME:     tensor<2x2x1xi32> to tensor<1x1x1xi32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SLICE]]
//  CHECK-SAME:     tensor<1x1x1xi32> into tensor<1xi32>

// -----

func.func @gather_to_extract_slice_partial_collapse(%source : tensor<2x2x100x100xi32>, %indices : tensor<2xi32>) -> (tensor<100x100xi32>) {
  %empty = tensor.empty() : tensor<100x100xi32>
  %result = iree_linalg_ext.gather dimension_map = [1, 0]
    ins(%source, %indices : tensor<2x2x100x100xi32>, tensor<2xi32>)
    outs(%empty: tensor<100x100xi32>) -> tensor<100x100xi32>
  return %result : tensor<100x100xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_partial_collapse
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[IDX0:.+]] = tensor.extract %[[ARG1]][%[[C0]]]
//   CHECK-DAG:   %[[IDX1:.+]] = tensor.extract %[[ARG1]][%[[C1]]]
//   CHECK-DAG:   %[[CAST0:.+]] = arith.index_cast %[[IDX0]]
//   CHECK-DAG:   %[[CAST1:.+]] = arith.index_cast %[[IDX1]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST1]], %[[CAST0]], 0, 0] [1, 1, 100, 100] [1, 1, 1, 1]
//  CHECK-SAME:     tensor<2x2x100x100xi32> to tensor<1x1x100x100xi32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SLICE]]
//  CHECK-SAME:     tensor<1x1x100x100xi32> into tensor<100x100xi32>

// -----

func.func public @staticize_attention_from_operand(%arg0: tensor<?x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16) -> tensor<20x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16) outs(%0 : tensor<20x4096x64xf16>) {
  ^bb0(%arg4: f16):
    iree_linalg_ext.yield %arg4 : f16
  } -> tensor<20x4096x64xf16>
  return %1 : tensor<20x4096x64xf16>
}
//CHECK-LABEL: func public @staticize_attention_from_operand(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//      CHECK:   %[[CAST:.+]] = tensor.cast %[[ARG0]]
// CHECK-SAME:     : tensor<?x4096x16xf16> to tensor<20x4096x16xf16>
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       ins(%[[CAST]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//      CHECK:   return %[[ATTENTION]]

// -----

func.func public @staticize_attention_from_cast(%arg0: tensor<?x4096x16xf16>, %arg1: tensor<20x?x16xf16>, %arg2: tensor<20x?x64xf16>, %arg3: f16) -> tensor<20x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %cast0 = tensor.cast %arg1 : tensor<20x?x16xf16> to tensor<?x?x16xf16>
  %cast1 = tensor.cast %arg2 : tensor<20x?x64xf16> to tensor<?x?x64xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%arg0, %cast0, %cast1, %arg3 : tensor<?x4096x16xf16>, tensor<?x?x16xf16>, tensor<?x?x64xf16>, f16) outs(%0 : tensor<20x4096x64xf16>) {
  ^bb0(%arg4: f16):
    iree_linalg_ext.yield %arg4 : f16
  } -> tensor<20x4096x64xf16>
  return %1 : tensor<20x4096x64xf16>
}
//CHECK-LABEL: func public @staticize_attention_from_cast(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x?x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x?x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//      CHECK:   %[[CAST:.+]] = tensor.cast %[[ARG0]]
// CHECK-SAME:     : tensor<?x4096x16xf16> to tensor<20x4096x16xf16>
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       ins(%[[CAST]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//      CHECK:   return %[[ATTENTION]]

// -----

func.func public @staticize_online_attention_from_cast(%arg0: tensor<?x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16) -> tensor<20x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = tensor.empty() : tensor<20x4096xf16>
  %cast0 = tensor.cast %arg1 : tensor<20x1024x16xf16> to tensor<?x1024x16xf16>
  %cast1 = tensor.cast %arg2 : tensor<20x1024x64xf16> to tensor<?x1024x64xf16>
  %2:3 = iree_linalg_ext.online_attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>]} ins(%arg0, %cast0, %cast1, %arg3 : tensor<?x4096x16xf16>, tensor<?x1024x16xf16>, tensor<?x1024x64xf16>, f16) outs(%0, %1, %1 : tensor<20x4096x64xf16>, tensor<20x4096xf16>, tensor<20x4096xf16>) {
  ^bb0(%arg4: f16):
    iree_linalg_ext.yield %arg4 : f16
  } -> tensor<20x4096x64xf16>, tensor<20x4096xf16>, tensor<20x4096xf16>
  return %2#0 : tensor<20x4096x64xf16>
}
//CHECK-LABEL: func public @staticize_online_attention_from_cast(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//      CHECK:   %[[CAST:.+]] = tensor.cast %[[ARG0]]
// CHECK-SAME:     : tensor<?x4096x16xf16> to tensor<20x4096x16xf16>
//      CHECK:   %[[ATTENTION:.+]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:       ins(%[[CAST]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//      CHECK:   return %[[ATTENTION]]#0

// -----

func.func public @convert_identity_map_store_into_copy(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %true = arith.constant true
  %map_store = iree_linalg_ext.map_store %arg0 into %arg1 {
  ^bb0(%arg2: index, %arg3: index):
    iree_linalg_ext.yield %arg2, %arg3, %true : index, index, i1
  } : tensor<?x?xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
  return %map_store : tensor<?x?xf32>
}
//CHECK-LABEL: func public @convert_identity_map_store_into_copy(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-NOT:   iree_linalg_ext.map_store
//      CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[ARG0]]{{.*}} outs(%[[ARG1]]
//      CHECK:   return %[[COPY]]

// -----

func.func public @convert_identity_map_load_into_copy(
    %arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>
) -> tensor<16x32xf32> {
  %cst = arith.constant 0.0 : f32
  %map_load = iree_linalg_ext.map_load %arg0 into %arg1 {
  ^bb0(%arg2: index, %arg3: index):
    iree_linalg_ext.yield %arg2, %arg3, %cst : index, index, f32
  } : tensor<16x32xf32> into tensor<16x32xf32> -> tensor<16x32xf32>
  return %map_load : tensor<16x32xf32>
}
//CHECK-LABEL: func public @convert_identity_map_load_into_copy(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x32xf32>
//  CHECK-NOT:   iree_linalg_ext.map_load
//      CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[ARG0]]{{.*}} outs(%[[ARG1]]
//      CHECK:   return %[[COPY]]

// -----

// Verify that identity map_load with dynamic shapes is not converted to copy.
func.func public @no_convert_dynamic_identity_map_load_into_copy(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %map_load = iree_linalg_ext.map_load %arg0 into %arg1 {
  ^bb0(%arg2: index, %arg3: index):
    iree_linalg_ext.yield %arg2, %arg3, %cst : index, index, f32
  } : tensor<?x?xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
  return %map_load : tensor<?x?xf32>
}
//CHECK-LABEL: func public @no_convert_dynamic_identity_map_load_into_copy(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//      CHECK:   iree_linalg_ext.map_load
//  CHECK-NOT:   linalg.copy

// -----

// Test: fold input tensor.pad into im2col padding attributes.

func.func @fold_input_pad_into_im2col(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1296x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1296x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      ins(%padded : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1296x5760xf32>) -> tensor<2x1296x5760xf32>
  return %result : tensor<2x1296x5760xf32>
}
// CHECK-LABEL: func.func @fold_input_pad_into_im2col(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1296x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%[[CST]] : f32)
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1296x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold output tensor.pad into im2col by expanding the output tensor.

func.func @fold_output_pad_into_im2col(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1040x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x1024x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%cst : f32)
      ins(%arg0 : tensor<2x34x34x640xf32>)
      outs(%empty : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  %padded = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1024x5760xf32> to tensor<2x1040x5760xf32>
  return %padded : tensor<2x1040x5760xf32>
}
// CHECK-LABEL: func.func @fold_output_pad_into_im2col(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1040x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1040x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: both input+output pads with incompatible pad values — neither folds.

func.func @no_fold_both_pads_incompatible(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1040x5760xf32> {
  %cst_pad = arith.constant 1.0 : f32
  %cst_im2col = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst_pad : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1024x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 0, 0, 0] input_pad_high = [0, 0, 0, 0] pad_value(%cst_im2col : f32)
      ins(%padded : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  %padded_output = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst_pad : f32
  } : tensor<2x1024x5760xf32> to tensor<2x1040x5760xf32>
  return %padded_output : tensor<2x1040x5760xf32>
}
// CHECK-LABEL: func.func @no_fold_both_pads_incompatible(
//       CHECK:   tensor.pad
//       CHECK:   iree_linalg_ext.im2col
//       CHECK:   tensor.pad

// -----

// Test: both input+output pads with compatible pad values — both fold.

func.func @fold_both_pads_compatible(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1312x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %padded_input = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1296x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      ins(%padded_input : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1296x5760xf32>) -> tensor<2x1296x5760xf32>
  %padded_output = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1296x5760xf32> to tensor<2x1312x5760xf32>
  return %padded_output : tensor<2x1312x5760xf32>
}
// CHECK-LABEL: func.func @fold_both_pads_compatible(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1312x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
//  CHECK-SAME:     output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
//  CHECK-SAME:     pad_value(%[[CST]] : f32)
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1312x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold input tensor.pad into im2col that already has input padding
// (composing by adding element-wise).

func.func @fold_input_pad_compose(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1296x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1296x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%cst : f32)
      ins(%padded : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1296x5760xf32>) -> tensor<2x1296x5760xf32>
  return %result : tensor<2x1296x5760xf32>
}
// CHECK-LABEL: func.func @fold_input_pad_compose(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     input_pad_low = [0, 2, 2, 0] input_pad_high = [0, 2, 2, 0] pad_value(%[[CST]] : f32)
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold output tensor.pad with non-zero low padding into im2col.

func.func @fold_output_pad_low(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1040x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x1024x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%cst : f32)
      ins(%arg0 : tensor<2x34x34x640xf32>)
      outs(%empty : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  %padded = tensor.pad %result low[0, 16, 0] high[0, 0, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1024x5760xf32> to tensor<2x1040x5760xf32>
  return %padded : tensor<2x1040x5760xf32>
}
// CHECK-LABEL: func.func @fold_output_pad_low(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1040x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     output_pad_low = [0, 16, 0] output_pad_high = [0, 0, 0]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1040x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold output tensor.pad composing with existing output padding.

func.func @fold_output_pad_compose(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1056x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x1040x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
      pad_value(%cst : f32)
      ins(%arg0 : tensor<2x34x34x640xf32>)
      outs(%empty : tensor<2x1040x5760xf32>) -> tensor<2x1040x5760xf32>
  %padded = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1040x5760xf32> to tensor<2x1056x5760xf32>
  return %padded : tensor<2x1056x5760xf32>
}
// CHECK-LABEL: func.func @fold_output_pad_compose(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1056x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     output_pad_low = [0, 0, 0] output_pad_high = [0, 32, 0]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1056x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]
