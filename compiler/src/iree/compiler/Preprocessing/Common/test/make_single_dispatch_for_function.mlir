// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-preprocessing-make-single-dispatch-for-function))" --split-input-file --mlir-print-local-scope %s | FileCheck %s

util.func @simple_test() -> tensor<10x20xf32> {
  %0 = tensor.empty() : tensor<10x20xf32>
  util.return %0 : tensor<10x20xf32>
}
// CHECK-LABEL: func public @simple_test() -> tensor<10x20xf32>
//  CHECK-NEXT:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     flow.return %[[EMPTY]]
//       CHECK:   return %[[DISPATCH]]

// -----

// Generic test excercising some basic use case.
util.func public @conv_2d(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view,
    %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.tensor.import wait(%arg2) => %arg0 : !hal.buffer_view -> tensor<4x3x234x234xbf16>
  %1 = hal.tensor.import wait(%arg2) => %arg1 : !hal.buffer_view -> tensor<10x3x3x3xbf16>
  %2 = tensor.empty() : tensor<4x10x232x232xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x10x232x232xf32>) -> tensor<4x10x232x232xf32>
  %4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%0, %1 : tensor<4x3x234x234xbf16>, tensor<10x3x3x3xbf16>) outs(%3 : tensor<4x10x232x232xf32>) -> tensor<4x10x232x232xf32>
  %5 = tensor.empty() : tensor<4x10x232x232xbf16>
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%4 : tensor<4x10x232x232xf32>) outs(%5 : tensor<4x10x232x232xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %9 = arith.truncf %in : f32 to bf16
    linalg.yield %9 : bf16
  } -> tensor<4x10x232x232xbf16>
  %7 = hal.tensor.barrier join(%6 : tensor<4x10x232x232xbf16>) => %arg3 : !hal.fence
  %8 = hal.tensor.export %7 : tensor<4x10x232x232xbf16> -> !hal.buffer_view
  util.return %8 : !hal.buffer_view
}
// CHECK-LABEL: func public @conv_2d
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !hal.buffer_view
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: !hal.buffer_view
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: !hal.fence
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: !hal.fence
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0
//   CHECK-DAG:   %[[IMPORT0:.+]] = hal.tensor.import wait(%[[ARG2]]) => %[[ARG0]]
//   CHECK-DAG:   %[[IMPORT1:.+]] = hal.tensor.import wait(%[[ARG2]]) => %[[ARG1]]
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty()
//       CHECK:     %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:         ins(%[[CST]] :
//  CHECK-SAME:         outs(%[[EMPTY]] :
//       CHECK:     %[[CONV:.+]] = linalg.conv_2d_nchw_fchw
//  CHECK-SAME:         ins(%[[IMPORT0]], %[[IMPORT1]] :
//  CHECK-SAME:         outs(%[[FILL]] :
//       CHECK:     %[[TRUNC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[CONV]] :
//       CHECK:     flow.return %[[TRUNC]]
//       CHECK:   %[[JOIN:.+]] = hal.tensor.barrier join(%[[DISPATCH]] :
//       CHECK:   %[[EXPORT:.+]] = hal.tensor.export %[[JOIN]]
//       CHECK:   util.return %[[EXPORT]]

// -----

// Check handling of interleaved ABI ops
util.func @interleaved_import(%arg0 : !hal.buffer_view, %arg1 : !hal.fence,
    %arg2 : !hal.buffer_view, %arg3 : !hal.fence,
    %arg4 : !hal.buffer_view, %arg5 : !hal.fence) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<10xf32>
  %1 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<10xf32>
  %2 = tensor.empty() : tensor<10xf32>
  %3 = linalg.add ins(%0, %1 : tensor<10xf32>, tensor<10xf32>)
      outs(%2 : tensor<10xf32>) -> tensor<10xf32>
  %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<10xf32>
  %5 = linalg.add ins(%3, %4 : tensor<10xf32>, tensor<10xf32>)
      outs(%2 : tensor<10xf32>) -> tensor<10xf32>
  util.return %3, %5 : tensor<10xf32>, tensor<10xf32>
}
// CHECK-LABEL: func public @interleaved_import(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !hal.buffer_view
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: !hal.fence
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: !hal.buffer_view
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: !hal.fence
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: !hal.buffer_view
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: !hal.fence
//   CHECK-DAG:   %[[IMPORT0:.+]] = hal.tensor.import wait(%[[ARG1]]) => %[[ARG0]]
//   CHECK-DAG:   %[[IMPORT1:.+]] = hal.tensor.import wait(%[[ARG3]]) => %[[ARG2]]
//   CHECK-DAG:   %[[IMPORT2:.+]] = hal.tensor.import wait(%[[ARG5]]) => %[[ARG4]]
//       CHECK:   %[[DISPATCH:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty()
//       CHECK:     %[[ADD1:.+]] = linalg.add ins(%[[IMPORT0]], %[[IMPORT1]] :
//       CHECK:     %[[ADD2:.+]] = linalg.add ins(%[[ADD1]], %[[IMPORT2]] :
//       CHECK:     flow.return %[[ADD1]], %[[ADD2]]
//       CHECK:   return %[[DISPATCH]]#0, %[[DISPATCH]]#1

// -----

// Check handling of interleaved result ABI ops
util.func @interleaved_export(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>,
    %arg2 : tensor<10xf32>, %arg3 : !hal.fence)
    -> (!hal.buffer_view, !hal.buffer_view) {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
      outs(%0 : tensor<10xf32>) -> tensor<10xf32>
  %2 = hal.tensor.barrier join(%1 : tensor<10xf32>) => %arg3 : !hal.fence
  %3 = hal.tensor.export %2 : tensor<10xf32> -> !hal.buffer_view
  %4 = linalg.add ins(%arg2, %1 : tensor<10xf32>, tensor<10xf32>)
      outs(%0 : tensor<10xf32>) -> tensor<10xf32>
  %5 = hal.tensor.barrier join(%4 : tensor<10xf32>) => %arg3 : !hal.fence
  %6 = hal.tensor.export %5 : tensor<10xf32> -> !hal.buffer_view
  util.return %3, %6 : !hal.buffer_view, !hal.buffer_view
}
// CHECK-LABEL: func public @interleaved_export(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<10xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10xf32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: !hal.fence
//       CHECK:   %[[DISPATCH:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty()
//       CHECK:     %[[ADD1:.+]] = linalg.add ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:     %[[ADD2:.+]] = linalg.add ins(%[[ARG2]], %[[ADD1]] :
//       CHECK:     flow.return %[[ADD1]], %[[ADD2]]
//   CHECK-DAG:   %[[BARRIER0:.+]] = hal.tensor.barrier join(%[[DISPATCH]]#0 :
//   CHECK-DAG:   %[[EXPORT0:.+]] = hal.tensor.export %[[BARRIER0]]
//   CHECK-DAG:   %[[BARRIER1:.+]] = hal.tensor.barrier join(%[[DISPATCH]]#1 :
//   CHECK-DAG:   %[[EXPORT1:.+]] = hal.tensor.export %[[BARRIER1]]
//       CHECK:   return %[[EXPORT0]], %[[EXPORT1]]

// -----

// Check handling of values returned from dispatches used multiple times outside
// of it.
util.func @multi_use_return(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>,
    %arg2 : tensor<10xf32>, %arg3 : !hal.fence)
    -> (!hal.buffer_view, !hal.buffer_view) {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
      outs(%0 : tensor<10xf32>) -> tensor<10xf32>
  %2 = hal.tensor.barrier join(%1 : tensor<10xf32>) => %arg3 : !hal.fence
  %3 = hal.tensor.export %2 : tensor<10xf32> -> !hal.buffer_view
  %5 = hal.tensor.barrier join(%1 : tensor<10xf32>) => %arg3 : !hal.fence
  %6 = hal.tensor.export %5 : tensor<10xf32> -> !hal.buffer_view
  util.return %3, %6 : !hal.buffer_view, !hal.buffer_view
}
// CHECK-LABEL: func public @multi_use_return
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:   hal.tensor.barrier join(%[[DISPATCH]]
//       CHECK:   hal.tensor.barrier join(%[[DISPATCH]]
