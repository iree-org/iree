// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-flow-fuse-horizontal-contractions))" --split-input-file %s | FileCheck %s

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

// -----

util.func @horizontal_fusion_i8(%arg0: tensor<2x4096x640xi8>,
    %arg1: tensor<640x640xi8>, %arg2: tensor<640xi8>, %arg3: tensor<f32>,
    %arg6: tensor<640x640xi8>, %arg7: tensor<640xi8>, %arg8: tensor<f32>)
    -> (tensor<2x4096x640xf16>, tensor<2x4096x640xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<2x4096x640xf16>
  %1 = tensor.empty() : tensor<2x4096x640xi32>
  %2 = linalg.fill ins(%c0_i32 : i32)
      outs(%1 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %3 = tensor.empty() : tensor<2x4096xi32>
  %4 = linalg.fill ins(%c0_i32 : i32)
      outs(%3 : tensor<2x4096xi32>) -> tensor<2x4096xi32>
  %empty_rhs0 = tensor.empty() : tensor<2x640x640xi8>
  %rhs0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<640x640xi8>) outs(%empty_rhs0 : tensor<2x640x640xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
  } -> tensor<2x640x640xi8>
  %5 = linalg.batch_matmul_transpose_b
      ins(%arg0, %rhs0 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>)
      outs(%2 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0 : tensor<2x4096x640xi8>) outs(%4 : tensor<2x4096xi32>) {
    ^bb0(%in: i8, %out: i32):
      %12 = arith.extsi %in : i8 to i32
      %13 = arith.addi %12, %out : i32
      linalg.yield %13 : i32
  } -> tensor<2x4096xi32>
  %7 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> ()>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%5, %6, %arg2, %arg3 : tensor<2x4096x640xi32>, tensor<2x4096xi32>, tensor<640xi8>, tensor<f32>)
      outs(%0 : tensor<2x4096x640xf16>) {
    ^bb0(%in: i32, %in_0: i32, %in_1: i8, %in_2: f32, %out: f16):
      %13 = arith.extsi %in_1 : i8 to i32
      %14 = arith.muli %in_0, %13 : i32
      %15 = arith.subi %in, %14 : i32
      %16 = arith.sitofp %15 : i32 to f32
      %17 = arith.mulf %16, %in_2 : f32
      %18 = arith.truncf %17 : f32 to f16
      linalg.yield %18 : f16
  } -> tensor<2x4096x640xf16>
  %empty_rhs1 = tensor.empty() : tensor<2x640x640xi8>
  %rhs1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg6 : tensor<640x640xi8>) outs(%empty_rhs1 : tensor<2x640x640xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
  } -> tensor<2x640x640xi8>
  %8 = linalg.batch_matmul_transpose_b
      ins(%arg0, %rhs1 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>)
      outs(%2 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %9 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> ()>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%8, %6, %arg7, %arg8 : tensor<2x4096x640xi32>, tensor<2x4096xi32>, tensor<640xi8>, tensor<f32>)
      outs(%0 : tensor<2x4096x640xf16>) {
  ^bb0(%in: i32, %in_0: i32, %in_1: i8, %in_2: f32, %out: f16):
    %13 = arith.extsi %in_1 : i8 to i32
    %14 = arith.muli %in_0, %13 : i32
    %15 = arith.subi %in, %14 : i32
    %16 = arith.sitofp %15 : i32 to f32
    %17 = arith.mulf %16, %in_2 : f32
    %18 = arith.truncf %17 : f32 to f16
    linalg.yield %18 : f16
  } -> tensor<2x4096x640xf16>
  util.return %7, %9 : tensor<2x4096x640xf16>, tensor<2x4096x640xf16>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
//  CHECK-DAG: #[[MAP8:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
//  CHECK-DAG: #[[MAP9:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
//      CHECK: util.func public @horizontal_fusion_i8
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x4096x640xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<640x640xi8>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<640xi8>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<f32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<640x640xi8>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: tensor<640xi8>
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9]+]]: tensor<f32>

// First check the hoisted operand definitions.
//      CHECK:   %[[RHS0:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[ARG1]] :
//      CHECK:   %[[RHS1:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[ARG4]] :

// Concatanetion of the RHS
//  CHECK-DAG:   %[[EXPANDED_RHS0:.+]] = tensor.expand_shape %[[RHS0]] {{\[}}[0, 1], [2], [3]{{\]}}
//  CHECK-DAG:   %[[EXPANDED_RHS1:.+]] = tensor.expand_shape %[[RHS1]] {{\[}}[0, 1], [2], [3]{{\]}}
//      CHECK:   %[[INSERT_RHS0:.+]] = tensor.insert_slice %[[EXPANDED_RHS0]]
//      CHECK:   %[[INSERT_RHS1:.+]] = tensor.insert_slice %[[EXPANDED_RHS1]] into %[[INSERT_RHS0]]

// Check empty and fill for the concatenated contraction
//      CHECK:   %[[CONTRACT_EMPTY:.+]] = tensor.empty()
//      CHECK:   %[[CONTRACT_FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[CONTRACT_EMPTY]] :
//      CHECK:   %[[CONTRACT:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       ins(%[[ARG0]], %[[INSERT_RHS1]] :
// CHECK-SAME:       outs(%[[CONTRACT_FILL]] :

// The reduction that is adjacent to matmuls does not need to be hoisted. Is kept as is
//      CHECK:   %[[UNTOUCHED_REDUCTION:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[ARG0]] :

// Concatenation of non-common truncation operation operand
//  CHECK-DAG:   %[[EXPANDED_OPERAND1_1:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]{{\]}}
//  CHECK-DAG:   %[[EXPANDED_OPERAND1_2:.+]] = tensor.expand_shape %[[ARG5]] {{\[}}[0, 1]{{\]}}
//      CHECK:   %[[INSERT_OPERAND1_1:.+]] = tensor.insert_slice %[[EXPANDED_OPERAND1_1]]
//      CHECK:   %[[INSERT_OPERAND1_2:.+]] = tensor.insert_slice %[[EXPANDED_OPERAND1_2]] into %[[INSERT_OPERAND1_1]]

// Concatenation of non-common truncation operation zero-rank tensor operand
//  CHECK-DAG:   %[[EXPANDED_OPERAND2_1:.+]] = tensor.expand_shape %[[ARG3]] []
//  CHECK-DAG:   %[[EXPANDED_OPERAND2_2:.+]] = tensor.expand_shape %[[ARG6]] []
//      CHECK:   %[[INSERT_OPERAND2_1:.+]] = tensor.insert_slice %[[EXPANDED_OPERAND2_1]]
//      CHECK:   %[[INSERT_OPERAND2_2:.+]] = tensor.insert_slice %[[EXPANDED_OPERAND2_2]] into %[[INSERT_OPERAND2_1]]

// Concatanated truncate operation
//      CHECK:   %[[TRUNC_EMPTY:.+]] = tensor.empty()
//      CHECK:   %[[TRUNCATE:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP6]], #[[MAP7]], #[[MAP8]], #[[MAP9]], #[[MAP6]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:       ins(%[[CONTRACT]], %[[UNTOUCHED_REDUCTION]], %[[INSERT_OPERAND1_2]], %[[INSERT_OPERAND2_2]] :
// CHECK-SAME:       outs(%[[TRUNC_EMPTY]] :

// Extract the slices for replacement
//  CHECK-DAG:   %[[SLICE1:.+]] = tensor.extract_slice %[[TRUNCATE]][0, 0, 0, 0]
//  CHECK-DAG:   %[[SLICE2:.+]] = tensor.extract_slice %[[TRUNCATE]][1, 0, 0, 0]
//      CHECK:   util.return %[[SLICE1]], %[[SLICE2]]
