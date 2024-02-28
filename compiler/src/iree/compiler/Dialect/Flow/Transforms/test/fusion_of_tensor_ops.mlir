// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-flow-fusion-of-tensor-ops{fuse-multi-use=true}))" %s | FileCheck %s

util.func public @softmax(%arg0 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %1 = tensor.empty() : tensor<12x128xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<12x128x128xf32>) outs(%2 : tensor<12x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = arith.maximumf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %4 = tensor.empty() : tensor<12x128x128xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %3 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%b0: f32, %b1: f32, %arg2: f32):
    %11 = arith.subf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<12x128x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = math.exp %b0 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  %7 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<12x128x128xf32>) outs(%7 : tensor<12x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = arith.addf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<12x128xf32>) outs(%1 : tensor<12x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = arith.divf %cst, %b0 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %9 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%b0: f32, %b1: f32, %arg2: f32):
    %11 = arith.mulf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  util.return %10 : tensor<12x128x128xf32>
}
// CHECK-LABEL: util.func public @softmax
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12x128x128xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty()
//       CHECK:   %[[FILL0:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:       ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[ARG0]] :
//  CHECK-SAME:       outs(%[[FILL0]] :
//       CHECK:   %[[INIT1:.+]] = tensor.empty()
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[GENERIC0]] :
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[FILL1:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[GENERIC1]] :
//  CHECK-SAME:       outs(%[[FILL1]] :
//       CHECK:   %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[GENERIC1]], %[[GENERIC2]] :
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   util.return %[[GENERIC3]]

// -----

util.func public @batchnorm_training(%10 : tensor<12xf32>, %11 : tensor<12x12x12x12x12xf32>, %12 : tensor<12xf32>) -> (tensor<12xf32>, tensor<12xf32>, tensor<12xf32>)
{
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
module {
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
}
// CHECK-LABEL: util.func public @fuse_only_with_same_marke
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic


// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d2 + d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
module {
  util.func public @fuse_only_projected_perm(%arg0: tensor<16x1082x1922xi8>, %arg1: tensor<32x16x3x3xf32>, %arg2: tensor<32x1080x1920xi32>) -> tensor<32x1080x1920xi32> {
    %0 = tensor.empty() : tensor<32x16x3x3xi8>
    %eltwise = linalg.generic {
             indexing_maps = [#map0, #map0],
             iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
             ins(%arg1 : tensor<32x16x3x3xf32>)
             outs(%0 : tensor<32x16x3x3xi8>) {
    ^bb0(%in: f32, %out: i8):
      %1 = arith.fptosi %in : f32 to i8
      linalg.yield %1 : i8
    } -> tensor<32x16x3x3xi8>

    %conv = linalg.generic {
          indexing_maps = [#map1, #map2, #map3],
          iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"] }
          ins(%arg0, %eltwise : tensor<16x1082x1922xi8>, tensor<32x16x3x3xi8>)
          outs(%arg2 : tensor<32x1080x1920xi32>) {
    ^bb0(%in: i8, %in_108: i8, %out: i32):
      %232 = arith.extui %in : i8 to i32
      %233 = arith.extsi %in_108 : i8 to i32
      %234 = arith.muli %232, %233 : i32
      %235 = arith.addi %234, %out : i32
      linalg.yield %235 : i32
    } -> tensor<32x1080x1920xi32>

    util.return %conv : tensor<32x1080x1920xi32>
  }
}
// CHECK-LABEL: util.func public @fuse_only_projected_perm
// CHECK:         linalg.generic
// CHECK:         linalg.generic

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  util.func public @nofuse_broadcast_compute(%arg0: tensor<702x702x128xf32>, %arg1: tensor<702x702x128xf32>,
      %arg2: tensor<702x702x128xf32>, %arg3: tensor<702x702x128xf32>) -> tensor<128x702x702xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<702x702x128xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<702x702x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3 : tensor<702x702x128xf32>) outs(%0 : tensor<702x702x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.exp %in : f32
      linalg.yield %9 : f32
    } -> tensor<702x702x128xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %cst : tensor<702x702x128xf32>, tensor<702x702x128xf32>) outs(%0 : tensor<702x702x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.addf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<702x702x128xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst, %2 : tensor<702x702x128xf32>, tensor<702x702x128xf32>) outs(%0 : tensor<702x702x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.divf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<702x702x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg2 : tensor<702x702x128xf32>, tensor<702x702x128xf32>) outs(%0 : tensor<702x702x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.mulf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<702x702x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1, %3 : tensor<702x702x128xf32>, tensor<702x702x128xf32>) outs(%0 : tensor<702x702x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.mulf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<702x702x128xf32>
    %6 = tensor.empty() : tensor<128x702x702xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<128x702x702xf32>) -> tensor<128x702x702xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %4 : tensor<702x702x128xf32>, tensor<702x702x128xf32>) outs(%7 : tensor<128x702x702xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.mulf %in, %in_1 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<128x702x702xf32>
    util.return %8 : tensor<128x702x702xf32>
  }
}
// CHECK-LABEL: util.func public @nofuse_broadcast_compute(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<702x702x128xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<702x702x128xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<702x702x128xf32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<702x702x128xf32>)
//       CHECK:   %[[EMPTY0:.+]] = tensor.empty() : tensor<702x702x128xf32>
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]] :
//  CHECK-SAME:       outs(%[[EMPTY0]] :
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG1]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[EMPTY0]] :
//       CHECK:   %[[EMPTY1:.+]] = tensor.empty() : tensor<128x702x702xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[EMPTY1]] :
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[GENERIC1]], %[[GENERIC0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   util.return %[[GENERIC2]]

// -----

util.func public @fuse_iota_ops(%arg0: tensor<10x20xi32>) -> (tensor<10x20xi32>, tensor<10x20xi32>) {
  %c20 = arith.constant 20 : index
  %0 = tensor.empty() : tensor<10x20xi32>
  %1 = tensor.empty() : tensor<10x20xindex>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%1 : tensor<10x20xindex>) {
    ^bb0(%b0 : index):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = arith.muli %4, %c20 : index
      %6 = arith.addi %3, %5 : index
      linalg.yield %6 : index
    } -> tensor<10x20xindex>
  %7 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %2: tensor<10x20xi32>, tensor<10x20xindex>) outs(%0 : tensor<10x20xi32>) {
    ^bb0(%b0 : i32, %b1 : index, %b2 : i32):
      %8 = arith.index_cast %b1 : index to i32
      %9 = arith.addi %8, %b0 : i32
      linalg.yield %9 : i32
    } -> tensor<10x20xi32>
  %8 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %2: tensor<10x20xi32>, tensor<10x20xindex>) outs(%0 : tensor<10x20xi32>) {
    ^bb0(%b0 : i32, %b1 : index, %b2 : i32):
      %8 = arith.index_cast %b1 : index to i32
      %9 = arith.muli %8, %b0 : i32
      linalg.yield %9 : i32
    } -> tensor<10x20xi32>
    util.return %7, %8 : tensor<10x20xi32>, tensor<10x20xi32>
}
// CHECK-LABEL: util.func public @fuse_iota_ops(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi32>)
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<10x20xi32>
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<10x20xi32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<10x20xi32>)
//       CHECK:     linalg.index
//       CHECK:     linalg.index
//       CHECK:     arith.addi
//       CHECK:     linalg.yield
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<10x20xi32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<10x20xi32>)
//       CHECK:     linalg.index
//       CHECK:     linalg.index
//       CHECK:     arith.muli
//       CHECK:     linalg.yield
//       CHECK:   util.return %[[GENERIC1]], %[[GENERIC2]]

// -----

util.func public @no_fuse_within_dispatch(%arg0 : tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = flow.dispatch.region[] -> (tensor<10x20xf32>) {
    %1 = tensor.empty() : tensor<10x20xf32>
    %2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<10x20xf32>) outs(%1 : tensor<10x20xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %3 = arith.addf %b0, %b0 : f32
      linalg.yield %3 : f32
    } -> tensor<10x20xf32>
    %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%2, %arg0 : tensor<10x20xf32>, tensor<10x20xf32>) outs(%1 : tensor<10x20xf32>) {
    ^bb0(%b0: f32, %b1: f32, %b2 : f32):
      %4 = arith.mulf %b0, %b1 : f32
      linalg.yield %4 : f32
    } -> tensor<10x20xf32>
    flow.return %3 : tensor<10x20xf32>
  }
  util.return %0 : tensor<10x20xf32>
}
// CHECK-LABEL: util.func public @no_fuse_within_dispatch
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RETURN]]

// -----

util.func public @nofuse_by_expand_dequant(%arg0 : tensor<11008x4096xi4>, %arg1 : tensor<11008x32x1xf16>, %arg2 : tensor<11008x32x1xf16>, %arg3 : tensor<1x1x32x128xf16>) -> (tensor<11008xf16>) {
  %cst_1 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<11008x32x128xf16>
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<11008x4096xi4> into tensor<11008x32x128xi4>
  %collapsed = tensor.collapse_shape %arg2 [[0], [1, 2]] : tensor<11008x32x1xf16> into tensor<11008x32xf16>
  %collapsed_2 = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<11008x32x1xf16> into tensor<11008x32xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %collapsed, %collapsed_2 : tensor<11008x32x128xi4>, tensor<11008x32xf16>, tensor<11008x32xf16>) outs(%0 : tensor<11008x32x128xf16>) {
  ^bb0(%in: i4, %in_8: f16, %in_9: f16, %out: f16):
    %12 = arith.extui %in : i4 to i32
    %13 = arith.uitofp %12 : i32 to f16
    %14 = arith.subf %13, %in_9 : f16
    %15 = arith.mulf %14, %in_8 : f16
    linalg.yield %15 : f16
  } -> tensor<11008x32x128xf16>
  %collapsed_3 = tensor.collapse_shape %arg3 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
  %3 = tensor.empty() : tensor<11008xf16>
  %4 = linalg.fill ins(%cst_1 : f16) outs(%3 : tensor<11008xf16>) -> tensor<11008xf16>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%collapsed_3, %2 : tensor<32x128xf16>, tensor<11008x32x128xf16>) outs(%4 : tensor<11008xf16>) {
  ^bb0(%in: f16, %in_8: f16, %out: f16):
    %12 = arith.mulf %in, %in_8 : f16
    %13 = arith.addf %12, %out : f16
    linalg.yield %13 : f16
  } -> tensor<11008xf16>
  util.return %5 : tensor<11008xf16>
}
//     CHECK-LABEL: util.func public @nofuse_by_expand_dequant
//   CHECK-COUNT-2:   tensor.collapse_shape
//           CHECK:     %[[DEQUANT:.+]] = linalg.generic
//      CHECK-SAME:         iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK-NOT:   tensor.collapse_shape %[[DEQUANT]]
//           CHECK:     %[[MATVEC:.+]] = linalg.generic
//      CHECK-SAME:         iterator_types = ["parallel", "reduction", "reduction"]
//      CHECK-SAME:         ins({{.+}}%[[DEQUANT]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
util.func public @nofuse_by_collapse_matmul(%arg0: tensor<1x1xi64>, %arg1: tensor<4096x32x128xi4>, %arg2: tensor<4096x32x1xf16>, %arg3: tensor<4096x32x1xf16>) -> tensor<1x1x4096xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %c32000 = arith.constant 32000 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = util.unfoldable_constant dense<0.000000e+00> : tensor<32000x4096xf16>
  %1 = util.unfoldable_constant dense<0.000000e+00> : tensor<1x1x1xf16>
  %2 = tensor.empty() : tensor<1x1x4096xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1xi64>) outs(%2 : tensor<1x1x4096xf16>) {
  ^bb0(%in: i64, %out: f16):
    %11 = arith.index_cast %in : i64 to index
    %12 = linalg.index 2 : index
    %13 = arith.cmpi slt, %11, %c32000 : index
    cf.assert %13, "index must be smaller than dim size"
    %14 = arith.cmpi sge, %in, %c0_i64 : i64
    cf.assert %14, "index must be larger or equal to 0"
    %extracted = tensor.extract %0[%11, %12] : tensor<32000x4096xf16>
    linalg.yield %extracted : f16
  } -> tensor<1x1x4096xf16>
  %4 = tensor.empty() : tensor<1x1x4096xf16>
  %5 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %1 : tensor<1x1x4096xf16>, tensor<1x1x1xf16>) outs(%4 : tensor<1x1x4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %11 = arith.mulf %in, %in_0 : f16
    linalg.yield %11 : f16
  } -> tensor<1x1x4096xf16>
  %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<1x1x4096xf16>, tensor<1x1x4096xf16>) outs(%4 : tensor<1x1x4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %11 = arith.addf %in, %in_0 : f16
    linalg.yield %11 : f16
  } -> tensor<1x1x4096xf16>
  %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] : tensor<1x1x4096xf16> into tensor<1x1x32x128xf16>
  %7 = tensor.empty() : tensor<4096x32x128xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%2 : tensor<1x1x4096xf16>) -> tensor<1x1x4096xf16>
  %9 = linalg.generic {indexing_maps = [#map1, #map3, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1, %arg2, %arg3 : tensor<4096x32x128xi4>, tensor<4096x32x1xf16>, tensor<4096x32x1xf16>) outs(%7 : tensor<4096x32x128xf16>) {
  ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
    %11 = arith.extui %in : i4 to i32
    %12 = arith.uitofp %11 : i32 to f16
    %13 = arith.subf %12, %in_1 : f16
    %14 = arith.mulf %13, %in_0 : f16
    linalg.yield %14 : f16
  } -> tensor<4096x32x128xf16>
  %10 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%expanded, %9 : tensor<1x1x32x128xf16>, tensor<4096x32x128xf16>) outs(%8 : tensor<1x1x4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %11 = arith.mulf %in, %in_0 : f16
    %12 = arith.addf %11, %out : f16
    linalg.yield %12 : f16
  } -> tensor<1x1x4096xf16>
  util.return %10 : tensor<1x1x4096xf16>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//      CHECK: util.func public @nofuse_by_collapse_matmul
//      CHECK:   %[[DEQUANT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]],  #[[MAP]]],
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-NOT:   tensor.collapse_shape %[[DEQUANT]]
//      CHECK:   %[[MATVEC:.+]] = linalg.generic  {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]],
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:     ins(%{{.+}}, %[[DEQUANT]] : tensor<1x1x32x128xf16>, tensor<4096x32x128xf16>)
//  CHECK-NOT:   tensor.expand_shape %[[MATVEC]]
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

// Check for fix for https://github.com/openxla/iree/issues/14953
util.func public @fix_issue_14953(%arg0: tensor<11008x32x1xf16>, %arg1: tensor<11008x32x1xf16>, %arg2: tensor<1x1x32x128xf16>) -> tensor<1x1x11008xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant dense<0> : tensor<11008x32x128xi4>
  %3 = util.optimization_barrier %cst_0 : tensor<11008x32x128xi4>
  %4 = tensor.empty() : tensor<11008x32x128xf16>
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<11008x32x1xf16> into tensor<11008x32xf16>
  %collapsed_1 = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<11008x32x1xf16> into tensor<11008x32xf16>
  %collapsed_2 = tensor.collapse_shape %arg2 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
  %5 = tensor.empty() : tensor<11008xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<11008xf16>) -> tensor<11008xf16>
  %7 = flow.dispatch.region -> (tensor<11008xf16>) {
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %collapsed, %collapsed_1 : tensor<11008x32x128xi4>, tensor<11008x32xf16>, tensor<11008x32xf16>) outs(%4 : tensor<11008x32x128xf16>) {
    ^bb0(%in: i4, %in_3: f16, %in_4: f16, %out: f16):
      %11 = arith.extui %in : i4 to i32
      %12 = arith.uitofp %11 : i32 to f16
      %13 = arith.subf %12, %in_4 : f16
      %14 = arith.mulf %13, %in_3 : f16
      linalg.yield %14 : f16
    } -> tensor<11008x32x128xf16>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%collapsed_2, %9 : tensor<32x128xf16>, tensor<11008x32x128xf16>) outs(%6 : tensor<11008xf16>) {
    ^bb0(%in: f16, %in_3: f16, %out: f16):
      %11 = arith.mulf %in, %in_3 : f16
      %12 = arith.addf %11, %out : f16
      linalg.yield %12 : f16
    } -> tensor<11008xf16>
    flow.return %10 : tensor<11008xf16>
  }
  %expanded = tensor.expand_shape %7 [[0, 1, 2]] : tensor<11008xf16> into tensor<1x1x11008xf16>
  util.return %expanded : tensor<1x1x11008xf16>
}
// CHECK-LABEL: util.func public @fix_issue_14953
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%{{.+}}, %[[GENERIC0]] :
//       CHECK:     flow.return %[[GENERIC1]]
