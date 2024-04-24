// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-sink-reshapes))" --split-input-file %s | FileCheck %s

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
  %3 = tensor.expand_shape %2 [[0, 1], [2]] : tensor<?x?xf32> into tensor<2x?x?xf32>
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
  %expand = tensor.expand_shape %reduce [[0, 1]] : tensor<?xf32> into tensor<2x?xf32>
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
