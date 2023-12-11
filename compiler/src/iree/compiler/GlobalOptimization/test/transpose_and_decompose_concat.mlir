// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-decompose-concat{enable-concat-transposition=true}, cse))" %s | FileCheck %s

func.func @test_inner_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<32x?x128xf16> {
  %concat = tensor.concat dim(2) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<32x?x128xf16>
  return %concat : tensor<32x?x128xf16>
}
// CHECK-LABEL: func.func @test_inner_dim_concat
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//       CHECK:   %[[T0:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       CHECK:   %[[T1:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       CHECK:   %[[SLICE0:.+]] = tensor.insert_slice %[[T0]] {{.*}}[0, 0, 0] [64, 32, %{{.*}}] [1, 1, 1]
//       CHECK:   %[[SLICE1:.+]] = tensor.insert_slice %[[T1]] into %[[SLICE0]][64, 0, 0] [64, 32, %{{.*}}] [1, 1, 1]
//       CHECK:   %[[T2:.+]] = linalg.transpose ins(%[[SLICE1]] : tensor<128x32x?xf16>) {{.*}} permutation = [1, 2, 0]
//       CHECK:   return %[[T2]] : tensor<32x?x128xf16>

// -----

func.func @test_outer_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<64x?x64xf16> {
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<64x?x64xf16>
  return %concat : tensor<64x?x64xf16>
}
// CHECK-LABEL: func.func @test_outer_dim_concat
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//       CHECK:   %[[SLICE0:.+]] = tensor.insert_slice %[[ARG0]] {{.*}}[0, 0, 0] [32, %{{.*}}, 64] [1, 1, 1]
//       CHECK:   %[[SLICE1:.+]] = tensor.insert_slice %[[ARG1]] into %[[SLICE0]][32, 0, 0] [32, %{{.*}}, 64] [1, 1, 1]
//       CHECK:   return %[[SLICE1]] : tensor<64x?x64xf16>
