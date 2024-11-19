// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-global-opt-decompose-concat, cse))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-global-opt-decompose-concat{enable-concat-transposition=true}, cse))" %s | FileCheck %s --check-prefix=MAKEOUTER

util.func public @test_inner_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<32x?x128xf16> {
  %concat = tensor.concat dim(2) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<32x?x128xf16>
  util.return %concat : tensor<32x?x128xf16>
}
// CHECK-LABEL: util.func public @test_inner_dim_concat
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D0]])
//       CHECK:   %[[SLICE0:.+]] = tensor.insert_slice %[[ARG0]] into %[[EMPTY]][0, 0, 0] [32, %[[D0]], 64] [1, 1, 1]
//       CHECK:   %[[SLICE1:.+]] = tensor.insert_slice %[[ARG1]] into %[[SLICE0]][0, 0, 64] [32, %[[D1]], 64] [1, 1, 1]
//       CHECK:   util.return %[[SLICE1]] : tensor<32x?x128xf16>

// MAKEOUTER-LABEL: util.func public @test_inner_dim_concat
//  MAKEOUTER-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//  MAKEOUTER-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//       MAKEOUTER:   %[[T0:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       MAKEOUTER:   %[[T1:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       MAKEOUTER:   %[[CONCAT:.+]] = tensor.concat dim(0) %[[T0]], %[[T1]] :
//       MAKEOUTER:   %[[T2:.+]] = linalg.transpose ins(%[[CONCAT]] : tensor<128x32x?xf16>) {{.*}} permutation = [1, 2, 0]
//       MAKEOUTER:   util.return %[[T2]] : tensor<32x?x128xf16>

// -----

// Do not decompose outer dim concats.
util.func public @test_outer_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<64x?x64xf16> {
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<64x?x64xf16>
  util.return %concat : tensor<64x?x64xf16>
}
// CHECK-LABEL: util.func public @test_outer_dim_concat
//       CHECK:   %[[CONCAT:.+]] = tensor.concat
//       CHECK:   util.return %[[CONCAT]] : tensor<64x?x64xf16>

// MAKEOUTER-LABEL: util.func public @test_outer_dim_concat
//       MAKEOUTER:   %[[CONCAT:.+]] = tensor.concat
//       MAKEOUTER:   util.return %[[CONCAT]] : tensor<64x?x64xf16>
