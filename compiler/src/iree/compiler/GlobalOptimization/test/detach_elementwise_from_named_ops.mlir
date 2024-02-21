// RUN: iree-opt --split-input-file --iree-global-opt-detach-elementwise-from-named-ops --mlir-print-local-scope %s | FileCheck %s

util.func public @matmul(%a: tensor<?x64xf32>, %b: tensor<64x?xf32>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%c : tensor<?x?xf32>) outs(%c : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %1 = arith.addf %b0, %b0 : f32
      linalg.yield %1 : f32
    } -> tensor<?x?xf32>
  %1 = linalg.matmul ins(%a, %b : tensor<?x64xf32>, tensor<64x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: util.func public @matmul
//  CHECK-SAME: (%[[A:.+]]: tensor<?x64xf32>, %[[B:.+]]: tensor<64x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[C:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[ARG2]] :
//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[C]], %[[C0]]
//       CHECK:   %[[DIM1:.+]] = tensor.dim %[[C]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]])
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[F0]] : f32) outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:   %[[MM:.+]] = linalg.matmul
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<?x64xf32>, tensor<64x?xf32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x?xf32>)
//       CHECK:   %[[EW:.+]] = linalg.generic {
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]}
//  CHECK-SAME:     ins(%[[MM]], %[[C]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x?xf32>)
//       CHECK:   ^{{.+}}(%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %{{.+}}: f32):
//       CHECK:     %[[ADD:.+]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
//       CHECK:     linalg.yield %[[ADD]] : f32
//       CHECK:   util.return %[[EW]]

// -----

util.func public @batch_matmul(%a: tensor<?x8x?xi32>, %b: tensor<?x?x16xi32>, %c: tensor<?x8x16xi32>) -> tensor<?x8x16xi32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%c : tensor<?x8x16xi32>) outs(%c : tensor<?x8x16xi32>) {
    ^bb0(%b0 : i32, %b1 : i32):
      %1 = arith.addi %b0, %b0 : i32
      linalg.yield %1 : i32
    } -> tensor<?x8x16xi32>
  %1 = linalg.batch_matmul ins(%a, %b : tensor<?x8x?xi32>, tensor<?x?x16xi32>) outs(%0 : tensor<?x8x16xi32>) -> tensor<?x8x16xi32>
  util.return %1 : tensor<?x8x16xi32>
}

// CHECK-LABEL: util.func public @batch_matmul
//  CHECK-SAME: (%[[A:.+]]: tensor<?x8x?xi32>, %[[B:.+]]: tensor<?x?x16xi32>, %[[ARG2:.+]]: tensor<?x8x16xi32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[I0:.+]] = arith.constant 0 : i32
//       CHECK:   %[[C:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[ARG2]] :
//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[C]], %[[C0]] : tensor<?x8x16xi32>
//       CHECK:   %[[INIT:.+]] = tensor.empty(%[[DIM0]]) : tensor<?x8x16xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[I0]] : i32) outs(%[[INIT]] : tensor<?x8x16xi32>) -> tensor<?x8x16xi32>
//       CHECK:   %[[MM:.+]] = linalg.batch_matmul
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<?x8x?xi32>, tensor<?x?x16xi32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x8x16xi32>)
//       CHECK:   %[[EW:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[MM]], %[[C]] : tensor<?x8x16xi32>, tensor<?x8x16xi32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x8x16xi32>)
//       CHECK:     %[[ADD:.+]] = arith.addi
//       CHECK:     linalg.yield %[[ADD]] : i32
//       CHECK:   util.return %[[EW]]

// -----

util.func public @conv(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>, %init: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
  %init0 = tensor.empty() : tensor<1x112x112x32xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%init : tensor<32xf32>) outs(%init0 : tensor<1x112x112x32xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      linalg.yield %b0 : f32
    } -> tensor<1x112x112x32xf32>
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%0 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  util.return %1 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: util.func public @conv
//  CHECK-SAME: (%{{.+}}: tensor<1x225x225x3xf32>, %{{.+}}: tensor<3x3x3x32xf32>, %[[BIAS:.+]]: tensor<32xf32>)
//       CHECK:   %[[INIT:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[BIAS]] :
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[CONV]], %[[INIT]] : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<1x112x112x32xf32>)

// -----

util.func public @keep_fill(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %gemm : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @keep_fill
//   CHECK-NOT: linalg.generic

// -----

util.func public @keep_arg(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @keep_arg
//   CHECK-NOT: linalg.generic

// -----

util.func public @fft_cst_output(%arg0 : tensor<3x2190x1x512xf32>)
    -> (tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<1.000000e+00> : tensor<1xf32>
  %cst_0 = arith.constant dense<-0.000000e+00> : tensor<1xf32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<3x2190x1x512xf32>
  %0:2 = iree_linalg_ext.fft
      ins(%c1, %cst, %cst_0 : index, tensor<1xf32>, tensor<1xf32>)
      outs(%arg0, %cst_1 : tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>)
      : tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>
  util.return %0#0, %0#1 : tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>
}
// CHECK-LABEL: util.func public @fft_cst_output(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<3x2190x1x512xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty()
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[C0]] : f32)
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[FFT:.+]]:2 = iree_linalg_ext.fft
//  CHECK-SAME:       outs(%[[ARG0]], %[[FILL]] :
//       CHECK:   util.return %[[FFT]]#0, %[[FFT]]#1

// -----

/// Generic op with const operand.

#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
util.func public @generic_cst_output(%arg0 : tensor<114x114x64xf32>) -> tensor<56x56x64xf32> {
  %cst = arith.constant dense<0xFF800000> : tensor<56x56x64xf32>
  %1 = tensor.empty() : tensor<3x3xf32>
  %2 = linalg.generic {
      indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %1 : tensor<114x114x64xf32>, tensor<3x3xf32>) outs(%cst : tensor<56x56x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.maximumf %out, %in : f32
    linalg.yield %3 : f32
  } -> tensor<56x56x64xf32>
  util.return %2 : tensor<56x56x64xf32>
}
// CHECK-LABEL: util.func public @generic_cst_output
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<114x114x64xf32>
//       CHECK:   %[[CST:.+]] = arith.constant 0xFF800000 : f32
//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<56x56x64xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[CST]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   util.return %[[GENERIC]]
