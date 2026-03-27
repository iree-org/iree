// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization))" --split-input-file %s | FileCheck %s

// Tests for im2col op vectorization via VectorizableOpInterface.

// Standard NHWC layout, K tile size (4) divides innermost input dim C (640).
// Vectorizes along K (output dim 2) with vector width 4.
// Non-vectorized dims: batch (2) x M (2) = 4 iterations.
#im2col_map_k = affine_map<(d0) -> (d0 * 4)>
func.func @im2col_vectorize_nhwc(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply #im2col_map_k(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_nhwc
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[POISON:.+]] = ub.poison : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   %[[R0:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R0]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R1:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R1]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R2:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R2]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R3:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write %[[R3]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   return %[[FINAL]] : tensor<2x2x4xf32>

// -----

// Dynamic output shape: vectorization pattern should not match.
func.func @im2col_no_vectorize_dynamic(
    %input: tensor<2x34x34x640xf32>, %m_size: index, %m_off: index, %k: index
) -> tensor<2x?x4xf32> {
  %0 = tensor.empty(%m_size) : tensor<2x?x4xf32>
  %k_off = affine.apply affine_map<(d0) -> (d0 * 4)>(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
  return %1 : tensor<2x?x4xf32>
}
// CHECK-LABEL: func.func @im2col_no_vectorize_dynamic
//       CHECK:   iree_linalg_ext.im2col
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write

// -----

// Source padding (conv padding folded into im2col). NHWC layout.
// Vectorizes along K with masked transfer_read.
#im2col_map_k_pad = affine_map<(d0) -> (d0 * 4)>
func.func @im2col_vectorize_source_padding(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply #im2col_map_k_pad(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [34, 34], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_source_padding
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   return %[[FINAL]] : tensor<2x2x4xf32>

// -----

// Non-vectorizable due to input_k_perm = [1, 0] making innermost K
// non-contiguous in input. Falls back to scalar unrolling (vector<1>).
func.func @im2col_scalar_fallback(
    %input: tensor<1x3x2xf32>
) -> tensor<1x2x4xf32> {
  %0 = tensor.empty() : tensor<1x2x4xf32>
  %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                          offsets = [0, 0, 0] output_sizes = [[1], [2], [2, 2]]
                          batch_pos = [0] m_pos = [1] k_pos = [2]
                          input_k_perm = [1, 0] output_perm = [0, 1, 2]
                          ins(%input : tensor<1x3x2xf32>)
                          outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  return %1 : tensor<1x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_scalar_fallback
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x3x2xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<1x3x2xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, tensor<1x2x4xf32>

// -----

// High-side input padding on the vectorized input dimension (channels).
// Verifies: masked vector transfer_read with pad_value, im2col fully lowered.
func.func @im2col_vectorize_channel_pad_high(
    %input: tensor<59x91x16x56xbf16>, %output: tensor<1x1x1x8xbf16>,
    %off0: index
) -> tensor<1x1x1x8xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [59, 91]
      offsets = [%off0, %c3, %c5, %c100]
      output_sizes = [[64], [16], [3, 3], [59, 91]]
      batch_pos = [3, 2] m_pos = [0, 1] k_pos = []
      input_k_perm = [0, 1] output_perm = [2, 3, 1, 0]
      input_pad_low = [1, 1, 0, 0] input_pad_high = [1, 1, 0, 8]
      pad_value(%cst : bf16)
      ins(%input : tensor<59x91x16x56xbf16>)
      outs(%output : tensor<1x1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
  return %result : tensor<1x1x1x8xbf16>
}
// CHECK-LABEL: func.func @im2col_vectorize_channel_pad_high
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<59x91x16x56xbf16>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : bf16
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.create_mask {{.*}} : vector<8xi1>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} {in_bounds = [true]} : tensor<59x91x16x56xbf16>, vector<8xbf16>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write {{.*}} : vector<8xbf16>, tensor<1x1x1x8xbf16>
//       CHECK:   return %[[FINAL]] : tensor<1x1x1x8xbf16>

// -----

// Low-side input padding on the vectorized input dimension: falls back to
// scalar unrolling (vector<1>) because chooseDimToVectorize returns nullopt.
func.func @im2col_scalar_fallback_channel_pad_low(
    %input: tensor<59x91x16x56xbf16>, %output: tensor<1x1x1x8xbf16>
) -> tensor<1x1x1x8xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %c42 = arith.constant 42 : index
  %c100 = arith.constant 100 : index
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [59, 91]
      offsets = [%c42, %c3, %c5, %c100]
      output_sizes = [[64], [16], [3, 3], [59, 91]]
      batch_pos = [3, 2] m_pos = [0, 1] k_pos = []
      input_k_perm = [0, 1] output_perm = [2, 3, 1, 0]
      input_pad_low = [1, 1, 0, 8] input_pad_high = [1, 1, 0, 0]
      pad_value(%cst : bf16)
      ins(%input : tensor<59x91x16x56xbf16>)
      outs(%output : tensor<1x1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
  return %result : tensor<1x1x1x8xbf16>
}
// All offsets are constant and in-bounds, so masks fold away. The im2col is
// fully lowered to 8 scalar (vector<1>) transfer_read/write pairs.
// CHECK-LABEL: func.func @im2col_scalar_fallback_channel_pad_low
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<59x91x16x56xbf16>
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} {in_bounds = [true]} : tensor<59x91x16x56xbf16>, vector<1xbf16>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xbf16>, tensor<1x1x1x8xbf16>

// -----

// Output-only padding (GEMM alignment). Vectorizes along K with masked reads.
// The output has 16 extra M positions filled with pad_value.
func.func @im2col_vectorize_output_padding(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply affine_map<(d0) -> (d0 * 4)>(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// Vectorizes along K (dim 2) with vector width 4. The output M-dim padding
// produces arith.select between the k-dim mask and all-false for each
// non-vectorized output dim. No input padding, so reads are from the
// unpadded tensor with clamped indices.
// CHECK-LABEL: func.func @im2col_vectorize_output_padding
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<2x2x4xf32>

// -----

// Output low-padding on the vectorized dim: falls back to scalar unrolling
// because chooseDimToVectorize skips dims with non-zero output_pad_low.
func.func @im2col_scalar_fallback_output_pad_low(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply affine_map<(d0) -> (d0 * 4)>(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          output_pad_low = [0, 0, 2] output_pad_high = [0, 0, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// Scalar fallback: output_pad_low on the K dim (dim 2) prevents vectorization.
// CHECK-LABEL: func.func @im2col_scalar_fallback_output_pad_low
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read {{.*}} : tensor<2x34x34x640xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, tensor<2x2x4xf32>
