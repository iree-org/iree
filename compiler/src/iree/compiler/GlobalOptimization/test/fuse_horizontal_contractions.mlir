// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-fuse-horizontal-contractions))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @test_horizontal_fuse(%arg0 : tensor<2x4096x640xf16>, %arg1: tensor<10x64x640xf16>, %arg2: tensor<10x64x640xf16>, %arg3: tensor<10x64x640xf16>) -> (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>) {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<2x10x4096x64xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x10x4096x64xf32>) -> tensor<2x10x4096x64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x4096x640xf16>, tensor<10x64x640xf16>) outs(%3 : tensor<2x10x4096x64xf32>) {
  ^bb0(%in: f16, %in_5: f16, %out: f32):
    %11 = arith.extf %in : f16 to f32
    %12 = arith.extf %in_5 : f16 to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<2x10x4096x64xf32>
  %5 = tensor.empty() : tensor<2x10x4096x64xf16>
  %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<2x10x4096x64xf32>) outs(%5 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %11 = arith.truncf %in : f32 to f16
    linalg.yield %11 : f16
  } -> tensor<2x10x4096x64xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg2 : tensor<2x4096x640xf16>, tensor<10x64x640xf16>) outs(%3 : tensor<2x10x4096x64xf32>) {
  ^bb0(%in: f16, %in_5: f16, %out: f32):
    %11 = arith.extf %in : f16 to f32
    %12 = arith.extf %in_5 : f16 to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<2x10x4096x64xf32>
  %8 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<2x10x4096x64xf32>) outs(%5 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %11 = arith.truncf %in : f32 to f16
    linalg.yield %11 : f16
  } -> tensor<2x10x4096x64xf16>
  %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg3 : tensor<2x4096x640xf16>, tensor<10x64x640xf16>) outs(%3 : tensor<2x10x4096x64xf32>) {
  ^bb0(%in: f16, %in_5: f16, %out: f32):
    %11 = arith.extf %in : f16 to f32
    %12 = arith.extf %in_5 : f16 to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<2x10x4096x64xf32>
  %10 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<2x10x4096x64xf32>) outs(%5 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %11 = arith.truncf %in : f32 to f16
    linalg.yield %11 : f16
  } -> tensor<2x10x4096x64xf16>
  util.return %6, %8, %10 : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//      CHECK: util.func public @test_horizontal_fuse
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4096x640xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
//  CHECK-DAG:    %[[CST:.+]] = arith.constant 0.0
//  CHECK-DAG:    %[[EXP:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]] output_shape {{.*}} : tensor<10x64x640xf16> into tensor<1x10x64x640xf16>
//  CHECK-DAG:    %[[EXP1:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]] output_shape {{.*}} : tensor<10x64x640xf16> into tensor<1x10x64x640xf16>
//  CHECK-DAG:    %[[EXP2:.+]] = tensor.expand_shape %[[ARG3]] {{\[}}[0, 1], [2], [3]] output_shape {{.*}} : tensor<10x64x640xf16> into tensor<1x10x64x640xf16>
//      CHECK:    %[[INP:.+]] = tensor.empty() : tensor<3x10x64x640xf16>
//      CHECK:    %[[SLC:.+]] = tensor.insert_slice %[[EXP]] into %[[INP]][0, 0, 0, 0] [1, 10, 64, 640] [1, 1, 1, 1] : tensor<1x10x64x640xf16> into tensor<3x10x64x640xf16>
//      CHECK:    %[[SLC1:.+]] = tensor.insert_slice %[[EXP1]] into %[[SLC]][1, 0, 0, 0] [1, 10, 64, 640] [1, 1, 1, 1] : tensor<1x10x64x640xf16> into tensor<3x10x64x640xf16>
//      CHECK:    %[[SLC2:.+]] = tensor.insert_slice %[[EXP2]] into %[[SLC1]][2, 0, 0, 0] [1, 10, 64, 640] [1, 1, 1, 1] : tensor<1x10x64x640xf16> into tensor<3x10x64x640xf16>
//      CHECK:    %[[OUT:.+]] = tensor.empty() : tensor<3x2x10x4096x64xf32>
//      CHECK:    %[[FILL:.+]] = linalg.fill
// CHECK-SAME:        ins(%[[CST]]
// CHECK-SAME:        outs(%[[OUT]]
//      CHECK:    %[[GEN1:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[SLC2]] : tensor<2x4096x640xf16>, tensor<3x10x64x640xf16>) outs(%[[FILL]] : tensor<3x2x10x4096x64xf32>)
//      CHECK:    %[[EMPTY:.+]] = tensor.empty() : tensor<3x2x10x4096x64xf16>
//      CHECK:    %[[GEN2:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[GEN1]] : tensor<3x2x10x4096x64xf32>) outs(%[[EMPTY]] : tensor<3x2x10x4096x64xf16>) {
//      CHECK:    %[[R1:.+]] = tensor.extract_slice %[[GEN2]][0, 0, 0, 0, 0] [1, 2, 10, 4096, 64] [1, 1, 1, 1, 1] : tensor<3x2x10x4096x64xf16> to tensor<2x10x4096x64xf16>
//      CHECK:    %[[R2:.+]] = tensor.extract_slice %[[GEN2]][1, 0, 0, 0, 0] [1, 2, 10, 4096, 64] [1, 1, 1, 1, 1] : tensor<3x2x10x4096x64xf16> to tensor<2x10x4096x64xf16>
//      CHECK:    %[[R3:.+]] = tensor.extract_slice %[[GEN2]][2, 0, 0, 0, 0] [1, 2, 10, 4096, 64] [1, 1, 1, 1, 1] : tensor<3x2x10x4096x64xf16> to tensor<2x10x4096x64xf16>
//      CHECK:    util.return %[[R1]], %[[R2]], %[[R3]]

// -----

util.func public @test_horizontal_fuse(%arg0 : tensor<4096x640xf32>, %arg1: tensor<640x640xf32>, %arg2: tensor<640x640xf32>, %arg3: tensor<640x640xf32>) -> (tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<4096x640xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  %4 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x640xf32>, tensor<640x640xf32>) outs(%3 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  %7 = linalg.matmul ins(%arg0, %arg2 : tensor<4096x640xf32>, tensor<640x640xf32>) outs(%3 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  %9 = linalg.matmul ins(%arg0, %arg3 : tensor<4096x640xf32>, tensor<640x640xf32>) outs(%3 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  util.return %4, %7, %9 : tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//     CHECK: util.func public @test_horizontal_fuse
// CHECK-SAME:     %[[ARG0:.+]]: tensor<4096x640xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<640x640xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<640x640xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<640x640xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.0
// CHECK-DAG:    %[[EXP:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2]] output_shape {{.*}} : tensor<640x640xf32> into tensor<1x640x640xf32>
// CHECK-DAG:    %[[EXP1:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2]] output_shape {{.*}} : tensor<640x640xf32> into tensor<1x640x640xf32>
// CHECK-DAG:    %[[EXP2:.+]] = tensor.expand_shape %[[ARG3]] {{\[}}[0, 1], [2]] output_shape {{.*}} : tensor<640x640xf32> into tensor<1x640x640xf32>
//     CHECK:    %[[INP:.+]] = tensor.empty() : tensor<3x640x640xf32>
//     CHECK:    %[[SLC:.+]] = tensor.insert_slice %[[EXP]] into %[[INP]][0, 0, 0] [1, 640, 640] [1, 1, 1] : tensor<1x640x640xf32> into tensor<3x640x640xf32>
//     CHECK:    %[[SLC1:.+]] = tensor.insert_slice %[[EXP1]] into %[[SLC]][1, 0, 0] [1, 640, 640] [1, 1, 1] : tensor<1x640x640xf32> into tensor<3x640x640xf32>
//     CHECK:    %[[SLC2:.+]] = tensor.insert_slice %[[EXP2]] into %[[SLC1]][2, 0, 0] [1, 640, 640] [1, 1, 1] : tensor<1x640x640xf32> into tensor<3x640x640xf32>
//     CHECK:    %[[OUT:.+]] = tensor.empty() : tensor<3x4096x640xf32>
//     CHECK:    %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       ins(%[[CST]] :
// CHECK-SAME:       outs(%[[OUT]] :
//     CHECK:    %[[GEN1:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[SLC2]] : tensor<4096x640xf32>, tensor<3x640x640xf32>) outs(%[[FILL]] : tensor<3x4096x640xf32>)
//     CHECK:    %[[R1:.+]] = tensor.extract_slice %[[GEN1]][0, 0, 0] [1, 4096, 640] [1, 1, 1] : tensor<3x4096x640xf32> to tensor<4096x640xf32>
//     CHECK:    %[[R2:.+]] = tensor.extract_slice %[[GEN1]][1, 0, 0] [1, 4096, 640] [1, 1, 1] : tensor<3x4096x640xf32> to tensor<4096x640xf32>
//     CHECK:    %[[R3:.+]] = tensor.extract_slice %[[GEN1]][2, 0, 0] [1, 4096, 640] [1, 1, 1] : tensor<3x4096x640xf32> to tensor<4096x640xf32>
//     CHECK:    util.return %[[R1]], %[[R2]], %[[R3]]
