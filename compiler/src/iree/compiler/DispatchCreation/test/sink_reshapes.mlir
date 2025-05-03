// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-dispatch-creation-sink-reshapes))" --split-input-file %s | FileCheck %s

/// If for a `tensor.expand_shape` -> consumer pair if the consumer
/// can already be fused with an op by tile and fuse, do nothing. In
/// this example that would limit the sinking of `tensor.expand_shape`
/// operation by just one step

func.func @do_not_sink_across_already_fusable_ops(
    %arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>,
    %arg2 : tensor<?xf16>, %arg3 : tensor<2x?x?xf32>) -> tensor<2x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %n = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %cst = arith.constant 0.0: f32
  %0 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %m_by_2 = arith.divsi %m, %c2 : index
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = tensor.expand_shape %2 [[0, 1], [2]] output_shape [2, %m, %n]: tensor<?x?xf32> into tensor<2x?x?xf32>
  %4 = tensor.empty(%m_by_2, %n) : tensor<2x?x?xf16>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%3, %arg2 : tensor<2x?x?xf32>, tensor<?xf16>)
      outs(%4 : tensor<2x?x?xf16>) {
    ^bb0(%b0 : f32, %b1 : f16, %b2 : f16):
      %6 = arith.truncf %b0 : f32 to f16
      %7 = arith.addf %6, %b1 : f16
      linalg.yield %7 : f16
  } -> tensor<2x?x?xf16>
  %6 = tensor.empty(%m_by_2) : tensor<2x?xf32>
  %7 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg3 : tensor<2x?x?xf32>)
      outs(%6 : tensor<2x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %8 = arith.addf %b0, %b1 : f32
      linalg.yield %8 : f32
  } -> tensor<2x?xf32>
  %8 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%7, %5 : tensor<2x?xf32>, tensor<2x?x?xf16>)
      outs(%4 : tensor<2x?x?xf16>) {
    ^bb0(%b0 : f32, %b1 : f16, %b2 : f16) :
      %9 = arith.truncf %b0 : f32 to f16
      %10 = arith.addf %9, %b1 : f16
      linalg.yield %10 : f16
  } -> tensor<2x?x?xf16>
  func.return %8 : tensor<2x?x?xf16>
}

// CHECK-LABEL: func @do_not_sink_across_already_fusable_ops
//       CHECK:   %[[GEMM:.+]] = linalg.matmul_transpose_b
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[GEMM]],
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[GENERIC1]]
//       CHECK:   %[[REDUCE:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//       CHECK:   %[[RETURN:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[REDUCE]], %[[EXPAND]] :
//       CHECK:   return %[[RETURN]]

// -----

/// Do not sink the operations past dequantization-like operations.
/// The dequantize operations must be cloned into all consumers, which
/// will be prevented by the reshape being pushed down.

func.func @do_not_sink_across_dequantize_ops(%arg0: tensor<?x?xf32>) -> tensor<2x?xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %m = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %n = tensor.dim %arg0, %c2 : tensor<?x?xf32>
  %empty = tensor.empty(%m) : tensor<?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %reduce =  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
                  iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%fill : tensor<?xf32>) {
    ^bb0(%b0: f32, %b1 : f32):
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<?xf32>
  %m_by_2 = arith.divsi %m, %c2 : index
  %expand = tensor.expand_shape %reduce [[0, 1]] output_shape [2, %m] : tensor<?xf32> into tensor<2x?xf32>
  %empty1 = tensor.empty(%m_by_2) : tensor<2x?xf16>
  %generic = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
      ins(%expand : tensor<2x?xf32>) outs(%empty1 : tensor<2x?xf16>) {
    ^bb0(%b0 : f32, %b1 : f16):
      %0 = arith.truncf %b0 : f32 to f16
      linalg.yield %0 : f16
  } -> tensor<2x?xf16>
  %empty2 = tensor.empty(%m_by_2) : tensor<2x?xf32>
  %dequant = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
      ins(%generic : tensor<2x?xf16>) outs(%empty2 : tensor<2x?xf32>) {
    ^bb0(%b0 : f16, %b1 : f32):
      %0 = arith.extf %b0 : f16 to f32
      linalg.yield %0 : f32
  } -> tensor<2x?xf32>
  func.return %dequant : tensor<2x?xf32>
}
// CHECK-LABEL: func @do_not_sink_across_dequantize_ops
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[REDUCE:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] :
//       CHECK:   %[[QUANT:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[REDUCE]] :
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[QUANT]]
//       CHECK:   %[[DEQUANT:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[EXPAND]] :
//       CHECK:   return %[[DEQUANT]]

// -----

// Check that reshape sinks based with better estimate of what producers
// -> consumer are fusable.
func.func @better_producer_estimate(%lhs : tensor<2x4096x640xi32>, %rhs : tensor<2x640x640xi32>,
    %fill0 : tensor<2x4096x640xi32>, %fill1 : tensor<2x4096xi32>) -> tensor<2x4096x640x1xf16> {
  %bmm = linalg.batch_matmul_transpose_b ins(%lhs, %rhs : tensor<2x4096x640xi32>, tensor<2x640x640xi32>)
      outs(%fill0 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %reduction = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs : tensor<2x4096x640xi32>) outs(%fill1 : tensor<2x4096xi32>) {
    ^bb0(%in: i32, %out: i32):
      %12 = arith.addi %in, %out : i32
      linalg.yield %12 : i32
    } -> tensor<2x4096xi32>
  %expanded = tensor.expand_shape %bmm [[0], [1], [2, 3]] output_shape [2, 4096, 640, 1]
      : tensor<2x4096x640xi32> into tensor<2x4096x640x1xi32>
  %empty = tensor.empty() : tensor<2x4096x640x1xf16>
  %quant = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded, %reduction : tensor<2x4096x640x1xi32>, tensor<2x4096xi32>)
      outs(%empty : tensor<2x4096x640x1xf16>) {
    ^bb0(%in: i32, %in_3: i32, %out: f16):
      %14 = arith.subi %in, %in_3 : i32
      %16 = arith.sitofp %14 : i32 to f32
      %18 = arith.truncf %16 : f32 to f16
      linalg.yield %18 : f16
    } -> tensor<2x4096x640x1xf16>
  return %quant : tensor<2x4096x640x1xf16>
}
// CHECK-LABEL: func @better_producer_estimate(
//       CHECK:   %[[BMM:.+]] = linalg.batch_matmul_transpose_b
//       CHECK:   %[[REDUCTION:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[BMM]], %[[REDUCTION]] :
//       CHECK:   %[[COLLAPSE:.+]] = tensor.expand_shape %[[GENERIC]]
//       CHECK:   return %[[COLLAPSE]]

// -----

func.func @reduce_broadcast(%arg0: tensor<4x768xf32>, %arg1: tensor<4xf32>,
    %arg2: tensor<4xf32>, %arg3: tensor<1x4x768xf32>) -> tensor<1x4x768xf32> {
  %cst = arith.constant 9.000000e+00 : f32
  %cst_0 = arith.constant 8.000000e+00 : f32
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<4x768xf32>, tensor<4xf32>)
      outs(%arg2 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %3 = arith.subf %in, %in_2 : f32
    %4 = arith.mulf %3, %3 : f32
    %5 = arith.addf %out, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<4xf32>
  %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [1, 4]
      : tensor<4xf32> into tensor<1x4xf32>
  %1 = tensor.empty() : tensor<1x4x768xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg3, %expanded : tensor<1x4x768xf32>, tensor<1x4xf32>)
      outs(%1 : tensor<1x4x768xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %9 = arith.mulf %in, %in_2 : f32
    linalg.yield %9 : f32
  } -> tensor<1x4x768xf32>
  return %2 : tensor<1x4x768xf32>
}
// CHECK-LABEL: func @reduce_broadcast(
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       ins(%{{.+}}, %[[GENERIC1]] :
//       CHECK:   tensor.expand_shape %[[GENERIC2]]

// -----

func.func @fuse_softmax_with_truncate(%arg0 : tensor<4x64x?xf32>) -> tensor<4x64x1x?xf16> {
  %cst = arith.constant 0xFC00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant 11.3137083 : f32
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %arg0, %c2 : tensor<4x64x?xf32>
  %0 = tensor.empty(%dim) : tensor<4x64x?xf32>
  %2 = linalg.softmax dimension(2) ins(%arg0 : tensor<4x64x?xf32>) outs(%0 : tensor<4x64x?xf32>) -> tensor<4x64x?xf32>
  %expanded = tensor.expand_shape %2 [[0], [1, 2], [3]] output_shape [4, 64, 1, %dim] : tensor<4x64x?xf32> into tensor<4x64x1x?xf32>
  %3 = tensor.empty(%dim) : tensor<4x64x1x?xf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<4x64x1x?xf32>) outs(%3 : tensor<4x64x1x?xf16>) {
  ^bb0(%in: f32, %out: f16):
    %5 = arith.truncf %in : f32 to f16
    linalg.yield %5 : f16
  } -> tensor<4x64x1x?xf16>
  func.return %4 : tensor<4x64x1x?xf16>
}
// CHECK-LABEL: func @fuse_softmax_with_truncate
//       CHECK:   %[[SOFTMAX:.+]] = linalg.softmax
//       CHECK:   %[[TRUNC:.+]] = linalg.generic {{.*}} ins(%[[SOFTMAX]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[TRUNC]]
//       CHECK:   return %[[EXPAND]]

// -----

func.func @bubble_across_bit_extend(%arg0: tensor<2x64x32xf16>, %arg1 : tensor<2xf32>) -> tensor<2xf32> {
  %empty = tensor.empty() : tensor<2x64x32xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                  iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<2x64x32xf16>) outs(%empty : tensor<2x64x32xf32>) {
    ^bb0(%b0 : f16, %b1 : f32):
      %0 = arith.extf %b0 : f16 to f32
      linalg.yield %0 : f32
  } -> tensor<2x64x32xf32>
  %collapse = tensor.collapse_shape %0 [[0], [1, 2]] : tensor<2x64x32xf32> into tensor<2x2048xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
                  iterator_types = ["parallel", "reduction"]}
      ins(%collapse : tensor<2x2048xf32>) outs(%arg1 : tensor<2xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2  : f32
  } -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}
// CHECK-LABEL: func @bubble_across_bit_extend
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<2x64x32xf16>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<2xf32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]]
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[COLLAPSE]] :
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[GEN0]] :
//       CHECK:   return %[[GEN1]]
