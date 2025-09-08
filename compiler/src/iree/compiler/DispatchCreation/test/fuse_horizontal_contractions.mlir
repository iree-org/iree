// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-horizontal-contractions, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck %s

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

//      CHECK: util.func public @test_horizontal_fuse
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4096x640xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
//      CHECK:   %[[FILL:.+]] = linalg.fill
//      CHECK:   %[[FUSED_OP:.+]]:3 = linalg.generic
// CHECK-SAME:       indexing_maps
// CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
// CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-SAME:       iterator_types =
// CHECK-SAME:           ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[FILL]], %[[FILL]], %[[FILL]] :
// CHECK-NEXT:     ^bb0(
// CHECK-SAME:         %[[IN0:[a-zA-Z0-9_]+]]: f16
// CHECK-SAME:         %[[IN1:[a-zA-Z0-9_]+]]: f16
// CHECK-SAME:         %[[IN2:[a-zA-Z0-9_]+]]: f16
// CHECK-SAME:         %[[IN3:[a-zA-Z0-9_]+]]: f16
// CHECK-SAME:         %[[OUT0:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME:         %[[OUT1:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME:         %[[OUT2:[a-zA-Z0-9_]+]]: f32
//  CHECK-DAG:       %[[EXT_IN0:.+]] = arith.extf %[[IN0]]
//  CHECK-DAG:       %[[EXT_IN1:.+]] = arith.extf %[[IN1]]
//  CHECK-DAG:       %[[MULF0:.+]] = arith.mulf %[[EXT_IN0]], %[[EXT_IN1]]
//  CHECK-DAG:       %[[ADDF0:.+]] = arith.addf %[[OUT0]], %[[MULF0]]
//  CHECK-DAG:       %[[EXT_IN2:.+]] = arith.extf %[[IN2]]
//  CHECK-DAG:       %[[MULF1:.+]] = arith.mulf %[[EXT_IN0]], %[[EXT_IN2]]
//  CHECK-DAG:       %[[ADDF1:.+]] = arith.addf %[[OUT1]], %[[MULF1]]
//  CHECK-DAG:       %[[EXT_IN3:.+]] = arith.extf %[[IN3]]
//  CHECK-DAG:       %[[MULF2:.+]] = arith.mulf %[[EXT_IN0]], %[[EXT_IN3]]
//  CHECK-DAG:       %[[ADDF2:.+]] = arith.addf %[[OUT2]], %[[MULF2]]
//      CHECK:       linalg.yield %[[ADDF0]], %[[ADDF1]], %[[ADDF2]]
//      CHECK:   %[[TRUNCF1:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[FUSED_OP]]#0 :
//      CHECK:   %[[TRUNCF2:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[FUSED_OP]]#1 :
//      CHECK:   %[[TRUNCF3:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[FUSED_OP]]#2 :
//      CHECK:   util.return %[[TRUNCF1]], %[[TRUNCF2]], %[[TRUNCF3]]

// -----

util.func public @test_horizontal_fuse1(%arg0 : tensor<4096x640xf32>, %arg1: tensor<640x640xf32>, %arg2: tensor<640x640xf32>, %arg3: tensor<640x640xf32>) -> (tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<4096x640xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  %4 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x640xf32>, tensor<640x640xf32>) outs(%3 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  %7 = linalg.matmul ins(%arg0, %arg2 : tensor<4096x640xf32>, tensor<640x640xf32>) outs(%3 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  %9 = linalg.matmul ins(%arg0, %arg3 : tensor<4096x640xf32>, tensor<640x640xf32>) outs(%3 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  util.return %4, %7, %9 : tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>
}
//      CHECK: util.func public @test_horizontal_fuse1
// CHECK-SAME:     %[[ARG0:.+]]: tensor<4096x640xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<640x640xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<640x640xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<640x640xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
//      CHECK:   %[[FUSED_OP:.+]]:3 = linalg.generic
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[FILL]], %[[FILL]], %[[FILL]] :
//      CHECK:   util.return %[[FUSED_OP]]#0, %[[FUSED_OP]]#1, %[[FUSED_OP]]#2

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
  %5 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
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
  %8 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
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
// CHECK-LABEL: func public @horizontal_fusion_i8
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4096x640xi8>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<640x640xi8>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<640xi8>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<640x640xi8>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: tensor<640xi8>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9]+]]: tensor<f32>
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG1]] :
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG4]] :
//       CHECK:   %[[FUSED_OP:.+]]:2 = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[GENERIC1]], %[[GENERIC2]] :
//       CHECK:   %[[REDUCTION:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] :
//       CHECK:   %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[FUSED_OP]]#0, %[[REDUCTION]], %[[ARG2]], %[[ARG3]] :
//       CHECK:   %[[GENERIC4:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[FUSED_OP]]#1, %[[REDUCTION]], %[[ARG5]], %[[ARG6]] :
//       CHECK:   util.return %[[GENERIC3]], %[[GENERIC4]]

// -----

util.func public @fusion_same_trunc_op(%arg0: tensor<2x4096x640xi8>, %arg1: tensor<2x640x640xi8>, %arg2: tensor<640xi8>, %arg3: tensor<f32>, %arg4: tensor<2x640x640xi8>, %arg5: tensor<640xi8>, %arg6: tensor<f32>) -> tensor<2x4096x640xf16> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<2x4096x640xf16>
  %1 = tensor.empty() : tensor<2x4096x640xi32>
  %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %3 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg1 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%2 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %4 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg4 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%2 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %3 : tensor<2x4096x640xi32>, tensor<2x4096x640xi32>) outs(%0 : tensor<2x4096x640xf16>) {
  ^bb0(%in: i32, %in_0: i32, %out: f16):
    %6 = arith.sitofp %in : i32 to f32
    %7 = arith.truncf %6 : f32 to f16
    %8 = arith.sitofp %in_0 : i32 to f32
    %9 = arith.truncf %8 : f32 to f16
    %10 = arith.addf %7, %9 : f16
    linalg.yield %10 : f16
  } -> tensor<2x4096x640xf16>
  util.return %5 : tensor<2x4096x640xf16>
}
// CHECK-LABEL: func public @fusion_same_trunc_op
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4096x640xi8>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2x640x640xi8>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<640xi8>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<2x640x640xi8>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: tensor<640xi8>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9]+]]: tensor<f32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   %[[FUSED_OP:.+]]:2 = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG4]] :
//       CHECK:       outs(%[[FILL]], %[[FILL]] :
//       CHECK:   %[[TRUNCF:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[FUSED_OP]]#1, %[[FUSED_OP]]#0 :

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>
util.func public @test_horizontal_fuse_with_transpose(%arg0 : tensor<2x4096x640xf16>, %arg1: tensor<10x64x640xf16>, %arg2: tensor<10x64x640xf16>, %arg3: tensor<10x64x640xf16>) -> (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x64x4096xf16>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x10x64x4096xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x10x64x4096xf32>) -> tensor<2x10x64x4096xf32>
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
  %9 = linalg.generic {indexing_maps = [#map, #map1, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg3 : tensor<2x4096x640xf16>, tensor<10x64x640xf16>) outs(%1 : tensor<2x10x64x4096xf32>) {
  ^bb0(%in: f16, %in_5: f16, %out: f32):
    %11 = arith.extf %in : f16 to f32
    %12 = arith.extf %in_5 : f16 to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<2x10x64x4096xf32>
  %11 = tensor.empty() : tensor<2x10x64x4096xf16>
  %10 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<2x10x64x4096xf32>) outs(%11 : tensor<2x10x64x4096xf16>) {
  ^bb0(%in: f32, %out: f16):
    %12 = arith.truncf %in : f32 to f16
    linalg.yield %12 : f16
  } -> tensor<2x10x64x4096xf16>
  util.return %6, %8, %10 : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x64x4096xf16>
}
//      CHECK: util.func public @test_horizontal_fuse_with_transpose
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4096x640xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
//      CHECK:   %[[EMPTY0:.+]] = tensor.empty() : tensor<2x10x64x4096xf32>
//      CHECK:   %[[FILL0:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY0]] :
//      CHECK:   %[[EMPTY1:.+]] = tensor.empty() : tensor<2x10x4096x64xf32>
//      CHECK:   %[[FILL1:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY1]] :
//      CHECK:   %[[FUSED_OP:.+]]:3 = linalg.generic
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[FILL1]], %[[FILL1]], %[[FILL0]] :
//      CHECK:   %[[TRUNCF1:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[FUSED_OP]]#0 :
//      CHECK:   %[[TRUNCF2:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[FUSED_OP]]#1 :
//      CHECK:   %[[TRUNCF3:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[FUSED_OP]]#2 :
//      CHECK:   util.return %[[TRUNCF1]], %[[TRUNCF2]], %[[TRUNCF3]]

// -----

util.func @dont_fuse_contractions_with_different_n(%lhs : tensor<10x20xf32>,
    %rhs0 : tensor<20x40xf32>, %rhs1 : tensor<20x80xf32>)
    -> (tensor<10x40xf32>, tensor<10x80xf32>) {
  %0 = tensor.empty() : tensor<10x40xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %2 = linalg.matmul ins(%lhs, %rhs0 : tensor<10x20xf32>, tensor<20x40xf32>)
      outs(%1 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %3 = tensor.empty() : tensor<10x80xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<10x80xf32>) -> tensor<10x80xf32>
  %5 = linalg.matmul ins(%lhs, %rhs1 : tensor<10x20xf32>, tensor<20x80xf32>)
      outs(%4 : tensor<10x80xf32>) -> tensor<10x80xf32>
  util.return %2, %5 : tensor<10x40xf32>, tensor<10x80xf32>
}
// CHECK-LABEL: func public @dont_fuse_contractions_with_different_n(
//  CHECK-SAME:     %[[LHS:.+]]: tensor<10x20xf32>,
//  CHECK-SAME:     %[[RHS0:.+]]: tensor<20x40xf32>,
//  CHECK-SAME:     %[[RHS1:.+]]: tensor<20x80xf32>)
//       CHECK:   %[[MATMUL0:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS0]] :
//       CHECK:   %[[MATMUL1:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS1]] :
//       CHECK:   util.return %[[MATMUL0]], %[[MATMUL1]]

// -----

util.func public @check_horizontal_independence(%arg0: tensor<640x640xf32>,
    %arg1: tensor<640x640xf32>, %arg2: tensor<640x640xf32>,
    %arg3: tensor<640x640xf32>)
    -> (tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<640x640xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<640x640xf32>) -> tensor<640x640xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<640x640xf32>, tensor<640x640xf32>)
      outs(%1 : tensor<640x640xf32>) -> tensor<640x640xf32>
  %3 = linalg.matmul ins(%arg0, %arg2 : tensor<640x640xf32>, tensor<640x640xf32>)
      outs(%1 : tensor<640x640xf32>) -> tensor<640x640xf32>
  %4 = linalg.matmul ins(%arg0, %3 : tensor<640x640xf32>, tensor<640x640xf32>)
      outs(%1 : tensor<640x640xf32>) -> tensor<640x640xf32>
  util.return %2, %3, %4 : tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>
}
// CHECK-LABEL: func public @check_horizontal_independence
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<640x640xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<640x640xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: tensor<640x640xf32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: tensor<640x640xf32>
//       CHECK:   %[[FUSED_OP:.+]]:2 = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
//       CHECK:   %[[OP:.+]] = linalg.matmul
//       CHECK:       ins(%[[ARG0]], %[[FUSED_OP]]#1 :
//       CHECK:   util.return %[[FUSED_OP]]#0, %[[FUSED_OP]]#1, %[[OP]]

// -----

util.func public @fuse_horizontal_with_transpose(%arg0: tensor<2x4096x640xi8>,
    %arg1: tensor<10x64x640xi8>, %arg2: tensor<10x64x640xi8>, %arg3: tensor<10x64x640xi8>)
    -> (tensor<2x10x4096x64xi32>, tensor<2x10x4096x64xi32>, tensor<2x10x64x4096xi32>) {
  %0 = tensor.empty() : tensor<2x10x4096x64xi32>
  %c0_i32 = arith.constant 0 : i32
  %1 = linalg.fill ins(%c0_i32 : i32)
      outs(%0 : tensor<2x10x4096x64xi32>) -> tensor<2x10x4096x64xi32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<2x4096x640xi8>, tensor<10x64x640xi8>)
      outs(%1 : tensor<2x10x4096x64xi32>) {
  ^bb0(%in: i8, %in_0: i8, %out: i32):
    %8 = arith.extsi %in : i8 to i32
    %9 = arith.extsi %in_0 : i8 to i32
    %10 = arith.muli %8, %9 : i32
    %11 = arith.addi %out, %10 : i32
    linalg.yield %11 : i32
  } -> tensor<2x10x4096x64xi32>
  %3 = tensor.empty() : tensor<2x10x4096x64xf16>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg2 : tensor<2x4096x640xi8>, tensor<10x64x640xi8>)
      outs(%1 : tensor<2x10x4096x64xi32>) {
  ^bb0(%in: i8, %in_0: i8, %out: i32):
    %8 = arith.extsi %in : i8 to i32
    %9 = arith.extsi %in_0 : i8 to i32
    %10 = arith.muli %8, %9 : i32
    %11 = arith.addi %out, %10 : i32
    linalg.yield %11 : i32
  } -> tensor<2x10x4096x64xi32>
  %5 = tensor.empty() : tensor<2x10x64x4096xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<2x10x64x4096xi32>) -> tensor<2x10x64x4096xi32>
  %7 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg3 : tensor<2x4096x640xi8>, tensor<10x64x640xi8>)
      outs(%6 : tensor<2x10x64x4096xi32>) {
  ^bb0(%in: i8, %in_0: i8, %out: i32):
    %8 = arith.extsi %in : i8 to i32
    %9 = arith.extsi %in_0 : i8 to i32
    %10 = arith.muli %8, %9 : i32
    %11 = arith.addi %out, %10 : i32
    linalg.yield %11 : i32
  } -> tensor<2x10x64x4096xi32>
  util.return %2, %4, %7 : tensor<2x10x4096x64xi32>, tensor<2x10x4096x64xi32>, tensor<2x10x64x4096xi32>
}
// CHECK-LABEL: func public @fuse_horizontal_with_transpose
//  CHECK-SAME:     %[[LHS:.[a-zA-Z0-9]+]]: tensor<2x4096x640xi8>
//  CHECK-SAME:     %[[RHS0:[a-zA-Z0-9]+]]: tensor<10x64x640xi8>
//  CHECK-SAME:     %[[RHS1:[a-zA-Z0-9]+]]: tensor<10x64x640xi8>
//  CHECK-SAME:     %[[RHS2:[1-zA-Z0-9]+]]: tensor<10x64x640xi8>
//       CHECK:   %[[RESULT:.+]]:3 = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//  CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
//  CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
//  CHECK-SAME:           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS0]], %[[RHS1]], %[[RHS2]] :
//       CHECK:   util.return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2
