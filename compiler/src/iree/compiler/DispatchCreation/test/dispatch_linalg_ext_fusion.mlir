// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}, iree-dispatch-creation-clone-producers-into-dispatch-regions), cse, canonicalize, cse)" %s | FileCheck %s

util.func public @attention_dispatch(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: tensor<?x?x?xf16>, %arg3: f16, %arg4: tensor<?x?x?xf16>, %arg5: tensor<?x?x?xf16>, %arg6: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>

  %3 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%0, %1, %2, %arg3 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16) outs(%arg4 : tensor<?x?x?xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<?x?x?xf16>

  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  util.return %4 : tensor<?x?x?xf16>
}

// CHECK-LABEL:     util.func public @attention_dispatch
//       CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN0:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN0]]
//       CHECK:       %[[DISPATCH1:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN1:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN1]]
//       CHECK:       %[[DISPATCH2:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN2:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN2]]
//       CHECK:       %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:         %[[ATTN:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:           ins(%[[DISPATCH0]], %[[DISPATCH1]], %[[DISPATCH2]]
//       CHECK:         %[[GEN2:.+]] = linalg.generic
//  CHECK-SAME:           ins(%[[ATTN]]
//       CHECK:         flow.return %[[GEN2]]

// -----

util.func public @attention_dispatch_masked(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: tensor<?x?x?xf16>, %arg3: f16, %arg4: tensor<?x?x?xf16>, %arg5: tensor<?x?x?xf16>, %arg6: tensor<?x?x?xf16>, %arg7: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>

  %3 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%0, %1, %2, %arg3, %arg4: tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16, tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<?x?x?xf16>

  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  util.return %4 : tensor<?x?x?xf16>
}

// CHECK-LABEL:     util.func public @attention_dispatch_masked
//       CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN0:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN0]]
//       CHECK:       %[[DISPATCH1:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN1:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN1]]
//       CHECK:       %[[DISPATCH2:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN2:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN2]]
//       CHECK:       %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:         %[[ATTN:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:           ins(%[[DISPATCH0]], %[[DISPATCH1]], %[[DISPATCH2]]
//       CHECK:         %[[GEN2:.+]] = linalg.generic
//  CHECK-SAME:           ins(%[[ATTN]]
//       CHECK:         flow.return %[[GEN2]]
