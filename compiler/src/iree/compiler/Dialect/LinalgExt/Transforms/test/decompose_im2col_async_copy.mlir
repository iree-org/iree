// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col{unroll=false}))" \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

// Positive cases for decompose_mode = async_copy.

// Basic 3x3 NHWC conv im2col with async_copy decompose mode and k_off = 0.
// Expected IR:
//   - collapse source to 2D
//   - linalg.generic that builds a 1D index tensor via linalg.index +
//     affine.delinearize_index + affine.linearize_index
//   - iree_linalg_ext.gather with dimension_map = [0]
//   - expand shape back to the original output shape.
func.func @im2col_async_basic_3x3_nhwc(
    %input: tensor<1x16x16x512xf16>, %output: tensor<1x196x512xf16>)
    -> tensor<1x196x512xf16> {
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0] output_sizes = [[1], [14, 14], [3, 3, 512]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<1x16x16x512xf16>)
          outs(%output : tensor<1x196x512xf16>) -> tensor<1x196x512xf16>
  return %0 : tensor<1x196x512xf16>
}
// CHECK-LABEL: func.func @im2col_async_basic_3x3_nhwc
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0, 1, 2], [3]{{\]}}
//  CHECK-SAME:     tensor<1x16x16x512xf16> into tensor<256x512xf16>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<196xindex>
//       CHECK:   %[[INDICES:.+]] = linalg.generic
//  CHECK-SAME:     outs(%[[EMPTY]]
//       CHECK:     linalg.index 0
//       CHECK:     affine.delinearize_index
//       CHECK:     affine.linearize_index
//       CHECK:     linalg.yield {{.*}} : index
//       CHECK:   } -> tensor<196xindex>
//       CHECK:   %[[COLL_OUT:.+]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0, 1], [2]{{\]}}
//  CHECK-SAME:     tensor<1x196x512xf16> into tensor<196x512xf16>
//       CHECK:   %[[GATHER:.+]] = iree_linalg_ext.gather
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     ins(%[[COLLAPSED]], %[[INDICES]] : tensor<256x512xf16>, tensor<196xindex>)
//  CHECK-SAME:     outs(%[[COLL_OUT]] : tensor<196x512xf16>)
//       CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[GATHER]] {{\[}}[0, 1], [2]{{\]}}
//  CHECK-SAME:     tensor<196x512xf16> into tensor<1x196x512xf16>
//       CHECK:   return %[[RESULT]]

// -----

// k_off is a compile-time constant non-zero value that is channel-aligned
// (k_off = 512 = C), so the async-copy path accepts it.
func.func @im2col_async_constant_channel_aligned_koff(
    %input: tensor<1x16x16x512xf16>, %output: tensor<1x196x512xf16>)
    -> tensor<1x196x512xf16> {
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 512] output_sizes = [[1], [14, 14], [3, 3, 512]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<1x16x16x512xf16>)
          outs(%output : tensor<1x196x512xf16>) -> tensor<1x196x512xf16>
  return %0 : tensor<1x196x512xf16>
}
// CHECK-LABEL: func.func @im2col_async_constant_channel_aligned_koff
//       CHECK:   tensor.collapse_shape
//       CHECK:   linalg.generic
//       CHECK:   iree_linalg_ext.gather
//  CHECK-SAME:     dimension_map = [0]
//       CHECK:   tensor.expand_shape
//       CHECK:   return

// -----

// Dynamic k_off wrapped in an affine.apply that guarantees channel-aligned
// advancement (multiple of C = 512). contiguousSize (= 512) <= C (= 512),
// so the predicate's dynamic-offset branch accepts it.
#map = affine_map<(d0) -> (d0 * 512)>
func.func @im2col_async_dynamic_koff_channel_aligned(
    %input: tensor<1x16x16x512xf16>, %output: tensor<1x196x512xf16>,
    %k: index) -> tensor<1x196x512xf16> {
  %k_off = affine.apply #map(%k)
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, %k_off] output_sizes = [[1], [14, 14], [3, 3, 512]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<1x16x16x512xf16>)
          outs(%output : tensor<1x196x512xf16>) -> tensor<1x196x512xf16>
  return %0 : tensor<1x196x512xf16>
}
// CHECK-LABEL: func.func @im2col_async_dynamic_koff_channel_aligned
//       CHECK:   tensor.collapse_shape
//       CHECK:   linalg.generic
//       CHECK:   iree_linalg_ext.gather
//  CHECK-SAME:     dimension_map = [0]
//       CHECK:   tensor.expand_shape
//       CHECK:   return

// -----

// Negative cases below. Each violates exactly one async-copy precondition.

// Non-identity output_perm is rejected.
func.func @im2col_async_reject_output_perm(
    %input: tensor<1x16x16x512xf16>, %output: tensor<196x1x512xf16>)
    -> tensor<196x1x512xf16> {
  // expected-error @+2 {{async_copy decomposition preconditions not satisfied}}
  // expected-error @+1 {{im2col decomposition failed}}
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0] output_sizes = [[1], [14, 14], [3, 3, 512]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [1, 0, 2]
          ins(%input : tensor<1x16x16x512xf16>)
          outs(%output : tensor<196x1x512xf16>) -> tensor<196x1x512xf16>
  return %0 : tensor<196x1x512xf16>
}

// -----

// Constant non-channel-aligned k_off is rejected (k_off = 17, C = 512).
func.func @im2col_async_reject_unaligned_const_koff(
    %input: tensor<1x16x16x512xf16>, %output: tensor<1x196x512xf16>)
    -> tensor<1x196x512xf16> {
  // expected-error @+2 {{async_copy decomposition preconditions not satisfied}}
  // expected-error @+1 {{im2col decomposition failed}}
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 17] output_sizes = [[1], [14, 14], [3, 3, 512]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<1x16x16x512xf16>)
          outs(%output : tensor<1x196x512xf16>) -> tensor<1x196x512xf16>
  return %0 : tensor<1x196x512xf16>
}

// -----

// Non-identity input_k_perm is rejected.
func.func @im2col_async_reject_input_k_perm(
    %input: tensor<1x16x16x512xf16>, %output: tensor<1x196x512xf16>)
    -> tensor<1x196x512xf16> {
  // expected-error @+2 {{async_copy decomposition preconditions not satisfied}}
  // expected-error @+1 {{im2col decomposition failed}}
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0] output_sizes = [[1], [14, 14], [3, 3, 512]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [2, 1, 0] output_perm = [0, 1, 2]
          ins(%input : tensor<1x16x16x512xf16>)
          outs(%output : tensor<1x196x512xf16>) -> tensor<1x196x512xf16>
  return %0 : tensor<1x196x512xf16>
}

// -----

// Expanded-K layout: multiple non-unit K output dims are rejected. The
// predicate requires all non-vectorized K output dims to have size 1.
// Here dims 2 and 3 (kH=3, kW=3) are non-unit, violating that rule.
func.func @im2col_async_reject_expanded_k(
    %input: tensor<1x14x14x512xf16>, %output: tensor<1x196x3x3x512xf16>)
    -> tensor<1x196x3x3x512xf16> {
  // expected-error @+2 {{async_copy decomposition preconditions not satisfied}}
  // expected-error @+1 {{im2col decomposition failed}}
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0, 0, 0] output_sizes = [[1], [14, 14], [3], [3], [512]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3, 4]
          ins(%input : tensor<1x14x14x512xf16>)
          outs(%output : tensor<1x196x3x3x512xf16>) -> tensor<1x196x3x3x512xf16>
  return %0 : tensor<1x196x3x3x512xf16>
}

// -----

// Padded im2col with decompose_mode = async_copy is rejected in the
// dispatcher's hasPadding() guard, which emits a diagnostic so that the
// failure is observable under --verify-diagnostics.
func.func @im2col_async_reject_padding(
    %input: tensor<2x34x34x640xf32>, %output: tensor<2x1296x5760xf32>)
    -> tensor<2x1296x5760xf32> {
  %cst = arith.constant 0.0 : f32
  // expected-error @+2 {{im2col decomposition with padding is not yet implemented}}
  // expected-error @+1 {{im2col decomposition failed}}
  %0 = iree_linalg_ext.im2col
          {decompose_mode = #iree_linalg_ext.im2col_decompose_mode<async_copy>}
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0] output_sizes = [[2], [36, 36], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%output : tensor<2x1296x5760xf32>) -> tensor<2x1296x5760xf32>
  return %0 : tensor<2x1296x5760xf32>
}
