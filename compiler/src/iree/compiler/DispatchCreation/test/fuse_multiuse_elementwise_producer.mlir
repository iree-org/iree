// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-multi-use-elementwise-producer))" --split-input-file %s | FileCheck %s

util.func public @batchnorm_training(%10 : tensor<12xf32>, %11 : tensor<12x12x12x12x12xf32>, %12 : tensor<12xf32>) -> (tensor<12xf32>, tensor<12xf32>, tensor<12xf32>) {
  %cst = arith.constant 1.42 : f32
  %cst_1 = arith.constant 1.45 : f32
  %cst_0 = arith.constant 1.3 : f32
  %cst_2 = arith.constant 0.0 : f32
  %13 = tensor.empty() : tensor<12xf32>
  %14 = linalg.fill ins(%cst_2 : f32) outs(%13 : tensor<12xf32>) -> tensor<12xf32>
  %15 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4, d0)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction", "reduction", "reduction"]}
      ins(%11, %12 : tensor<12x12x12x12x12xf32>, tensor<12xf32>) outs(%14 : tensor<12xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %19 = arith.subf %arg1, %arg2 : f32
      %20 = arith.mulf %19, %19 : f32
      %21 = arith.addf %arg3, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<12xf32>
  %16 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%15: tensor<12xf32>) outs(%13 : tensor<12xf32>) {
    ^bb0(%arg1: f32, %arg2 : f32):
      %19 = arith.divf %arg1, %cst_1 : f32
      %20 = arith.addf %19, %cst_0 : f32
      linalg.yield %20 : f32
    } -> tensor<12xf32>
  %17 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%16 : tensor<12xf32>) outs(%13 : tensor<12xf32>) {
    ^bb0(%arg1: f32, %arg2 : f32):
      %19 = math.sqrt %arg1 : f32
      linalg.yield %19 : f32
    } -> tensor<12xf32>
  %18 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      {__internal_linalg_transform__ = "tensor_fuse_err"}
      ins(%10, %17 : tensor<12xf32>, tensor<12xf32>) outs(%13 : tensor<12xf32>)  {
    ^bb0(%arg1: f32, %arg2: f32, %arg3 : f32):
      %19 = arith.subf %arg1, %arg2 : f32
      %20 = arith.mulf %19, %cst : f32
      %21 = arith.subf %arg1, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<12xf32>
  util.return %16, %17, %18 : tensor<12xf32>, tensor<12xf32>, tensor<12xf32>
}
// CHECK-LABEL: util.func public @batchnorm_training(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<12x12x12x12x12xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<12xf32>
//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<12xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG1]], %[[ARG2]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   %[[GENERIC1:.+]]:3 = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[GENERIC0]] :
//  CHECK-SAME:       outs(%[[INIT]], %[[INIT]], %[[INIT]] :
//       CHECK:   util.return %[[GENERIC1]]#0, %[[GENERIC1]]#1, %[[GENERIC1]]#2

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @fuse_only_with_same_marker(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 3.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x5xf32>
  %1 = tensor.empty() : tensor<5x5xf32>
  %2 = tensor.empty() : tensor<5x5xf32>
  %3 = tensor.empty() : tensor<5x5xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%0 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %8 = arith.addf %arg2, %cst : f32
    linalg.yield %8 : f32
  } -> tensor<5x5xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<5x5xf32>) outs(%1 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %8 = arith.subf %arg2, %cst_0 : f32
    linalg.yield %8 : f32
  } -> tensor<5x5xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<5x5xf32>) outs(%2 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %8 = arith.addf %arg2, %cst_1 : f32
    linalg.yield %8 : f32
  } -> tensor<5x5xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : tensor<5x5xf32>, tensor<5x5xf32>) outs(%3 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %8 = arith.subf %arg2, %arg3 : f32
    linalg.yield %8 : f32
  } -> tensor<5x5xf32>
  util.return %4, %5, %6, %7 : tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>
}
// CHECK-LABEL: util.func public @fuse_only_with_same_marke
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic

// -----

util.func public @math_sin() {
  %cst = arith.constant 2.000000e+00 : f32
  %cst_0 = arith.constant dense<[0.000000e+00, 6.349640e-01, -6.349640e-01, 6.349640e-01]> : tensor<4xf32>
  %cst_1 = arith.constant dense<[0.000000e+00, 1.298460e+00, 1.298460e+00, -1.298460e+00]> : tensor<4xf32>
  %cst_2 = arith.constant dense<[0.000000e+00, 1.000000e+00, -1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %cst_3 = arith.constant dense<[0.000000e+00, 1.000000e+00, 1.000000e+00, -1.000000e+00]> : tensor<4xf32>
  %0 = util.optimization_barrier %cst_3 : tensor<4xf32>
  %1 = util.optimization_barrier %cst_2 : tensor<4xf32>
  %2 = tensor.empty() : tensor<4xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1 : tensor<4xf32>) outs(%2 : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = math.exp %in : f32
    linalg.yield %6 : f32
  } -> tensor<4xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %3, %1 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%2, %2 : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32, %out_6: f32):
    %6 = arith.negf %in_5 : f32
    %7 = math.exp %6 : f32
    %8 = arith.addf %in_4, %7 : f32
    %9 = math.sin %in : f32
    %10 = arith.mulf %9, %8 : f32
    %11 = arith.divf %10, %cst : f32
    linalg.yield %7, %11 : f32, f32
  } -> (tensor<4xf32>, tensor<4xf32>)
  %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %3, %4#0 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%2 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
    %6 = arith.subf %in_4, %in_5 : f32
    %7 = math.cos %in : f32
    %8 = arith.mulf %7, %6 : f32
    %9 = arith.divf %8, %cst : f32
    linalg.yield %9 : f32
  } -> tensor<4xf32>
  check.expect_almost_eq(%4#1, %cst_1) : tensor<4xf32>
  check.expect_almost_eq(%5, %cst_0) : tensor<4xf32>
  util.return
}
// CHECK-LABEL: util.func public @math_sin()
//       CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
//   CHECK-DAG:   check.expect_almost_eq(%[[GENERIC]]#0,
//   CHECK-DAG:   check.expect_almost_eq(%[[GENERIC]]#1,

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @fuse_by_moving_consumer(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<25xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 3.000000e+00 : f32
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %8 = arith.addf %arg2, %cst : f32
    linalg.yield %8 : f32
  } -> tensor<5x5xf32>
  // expected-note @below {{prior use here}}
  %collapsed = tensor.collapse_shape %4 [[0, 1]] : tensor<5x5xf32> into tensor<25xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %8 = arith.subf %arg2, %cst_0 : f32
    linalg.yield %8 : f32
  } -> tensor<5x5xf32>
  util.return %5, %collapsed: tensor<5x5xf32>, tensor<25xf32>
}
// CHECK-LABEL: util.func public @fuse_by_moving_consumer
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic


// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @dont_fuse_use_from_above(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<25xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 3.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<5x5xf32> into tensor<25xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c2 = arith.constant 2 : index
    %extracted = tensor.extract %collapsed[%c2] : tensor<25xf32>
    %2 = arith.addf %extracted, %extracted : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  util.return %1, %collapsed : tensor<5x5xf32>, tensor<25xf32>
}

// CHECK-LABEL: util.func public @dont_fuse_use_from_above
// CHECK:         linalg.generic
// CHECK:         linalg.generic


// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @do_fuse_use_from_above(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<25xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 3.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<5x5xf32> into tensor<25xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c2 = arith.constant 2 : index
    %extracted = tensor.extract %arg0[%c2, %c2] : tensor<5x5xf32>
    %2 = arith.addf %extracted, %extracted : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  util.return %1, %collapsed : tensor<5x5xf32>, tensor<25xf32>
}

// CHECK-LABEL: util.func public @do_fuse_use_from_above
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic
