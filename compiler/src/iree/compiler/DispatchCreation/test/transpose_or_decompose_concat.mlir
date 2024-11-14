// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-decompose-concat{enable-concat-transposition=true}, cse))" %s | FileCheck %s

util.func public @test_inner_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<32x?x128xf16> {
  %concat = tensor.concat dim(2) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<32x?x128xf16>
  util.return %concat : tensor<32x?x128xf16>
}
// CHECK-LABEL: util.func public @test_inner_dim_concat
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//       CHECK:   %[[T0:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       CHECK:   %[[T1:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       CHECK:   %[[CONCAT:.+]] = tensor.concat dim(0) %[[T0]], %[[T1]] :
//       CHECK:   %[[T2:.+]] = linalg.transpose ins(%[[CONCAT]] : tensor<128x32x?xf16>) {{.*}} permutation = [1, 2, 0]
//       CHECK:   util.return %[[T2]] : tensor<32x?x128xf16>

// -----

// Do not decompose outer dim concats.
util.func public @test_outer_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<64x?x64xf16> {
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<64x?x64xf16>
  util.return %concat : tensor<64x?x64xf16>
}
// CHECK-LABEL: util.func public @test_outer_dim_concat
//       CHECK:   %[[CONCAT:.+]] = tensor.concat
//       CHECK:   util.return %[[CONCAT]] : tensor<64x?x64xf16>
