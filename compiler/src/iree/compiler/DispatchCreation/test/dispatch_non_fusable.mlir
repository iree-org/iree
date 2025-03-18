// RUN: iree-opt %s --split-input-file --verify-diagnostics \
// RUN: --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}, \
// RUN:                  iree-dispatch-creation-clone-producers-into-dispatch-regions), cse, canonicalize, cse)" \
// RUN: | FileCheck %s

// Check that a simple elementwise bit extend producer is assigned to a separate dispatch
// (until fusion is supported)
#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @linalgext_scan_inclusive_dispatch_non_fusable(%arg0: tensor<8x32xi32>) -> tensor<8x32xi64> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<8x32xi64>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x32xi32>) outs(%0 : tensor<8x32xi64>) {
  ^bb0(%in: i32, %out: i64):
    %6 = arith.extsi %in : i32 to i64
    linalg.yield %6 : i64
  } -> tensor<8x32xi64>
  %2 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<8x32xi64>) -> tensor<8x32xi64>
  %3 = tensor.empty() : tensor<8xi64>
  %4 = linalg.fill ins(%c0_i64 : i64) outs(%3 : tensor<8xi64>) -> tensor<8xi64>
  %5:2 = iree_linalg_ext.scan dimension(1) inclusive(true) ins(%1 : tensor<8x32xi64>) outs(%2, %4 : tensor<8x32xi64>, tensor<8xi64>) {
  ^bb0(%arg3: i64, %arg4: i64):
    %6 = arith.addi %arg3, %arg4 : i64
    iree_linalg_ext.yield %6 : i64
  } -> tensor<8x32xi64>, tensor<8xi64>
  util.return %5#0 : tensor<8x32xi64>
}

// CHECK-LABEL:     util.func public @linalgext_scan_inclusive_dispatch_non_fusable(
//  CHECK-SAME:         %[[ARG:.+]]: tensor<8x32xi32>) -> tensor<8x32xi64>
//       CHECK:       %[[ZERO_CONST:.+]] = arith.constant 0 : i64
//       CHECK:       %[[PRODUCER_REGION:.+]] = flow.dispatch.region -> (tensor<8x32xi64>)
//       CHECK:         %[[EMPTY:.+]] = tensor.empty()
//       CHECK:         %[[PRODUCER:.+]] = linalg.generic {{.+}} ins(%[[ARG]] : tensor<8x32xi32>) outs(%[[EMPTY]] : tensor<8x32xi64>)
//       CHECK:         flow.return %[[PRODUCER]]
//       CHECK:       %[[LINALGEXT_REGION:.+]] = flow.dispatch.region -> (tensor<8x32xi64>)
//       CHECK:         %[[CUMULATIVE_FILL:.+]] = linalg.fill ins(%[[ZERO_CONST]] : i64) outs(%{{.+}} : tensor<8x32xi64>)
//       CHECK:         %[[REDUCED_FILL:.+]] = linalg.fill ins(%[[ZERO_CONST]] : i64) outs(%{{.+}} : tensor<8xi64>)
//       CHECK:         %[[SCAN_RESULT:.+]]:2 = iree_linalg_ext.scan dimension(1) inclusive(true)
//  CHECK-SAME:             ins(%[[PRODUCER_REGION]] : tensor<8x32xi64>) outs(%[[CUMULATIVE_FILL]], %[[REDUCED_FILL]] : {{.+}}) {
//       CHECK:         flow.return %[[SCAN_RESULT]]#0
//       CHECK:       util.return %[[LINALGEXT_REGION]] : tensor<8x32xi64>
