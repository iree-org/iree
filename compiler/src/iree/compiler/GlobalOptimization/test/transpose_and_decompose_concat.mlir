// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-global-opt-decompose-tensor-ops{enable-transposition=true}, cse))" %s | FileCheck %s

util.func public @test_inner_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<32x?x128xf16> {
  %concat = tensor.concat dim(2) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<32x?x128xf16>
  util.return %concat : tensor<32x?x128xf16>
}
// CHECK-LABEL: util.func public @test_inner_dim_concat
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//       CHECK:   %[[T0:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       CHECK:   %[[T1:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<32x?x64xf16>) {{.*}} permutation = [2, 0, 1]
//       CHECK:   %[[SLICE0:.+]] = tensor.insert_slice %[[T0]] {{.*}}[0, 0, 0] [64, 32, %{{.*}}] [1, 1, 1]
//       CHECK:   %[[SLICE1:.+]] = tensor.insert_slice %[[T1]] into %[[SLICE0]][64, 0, 0] [64, 32, %{{.*}}] [1, 1, 1]
//       CHECK:   %[[T2:.+]] = linalg.transpose ins(%[[SLICE1]] : tensor<128x32x?xf16>) {{.*}} permutation = [1, 2, 0]
//       CHECK:   util.return %[[T2]] : tensor<32x?x128xf16>

// -----

util.func public @test_outer_dim_concat(%arg0: tensor<32x?x64xf16>, %arg1: tensor<32x?x64xf16>) -> tensor<64x?x64xf16> {
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<64x?x64xf16>
  util.return %concat : tensor<64x?x64xf16>
}
// CHECK-LABEL: util.func public @test_outer_dim_concat
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x?x64xf16>
//       CHECK:   %[[SLICE0:.+]] = tensor.insert_slice %[[ARG0]] {{.*}}[0, 0, 0] [32, %{{.*}}, 64] [1, 1, 1]
//       CHECK:   %[[SLICE1:.+]] = tensor.insert_slice %[[ARG1]] into %[[SLICE0]][32, 0, 0] [32, %{{.*}}, 64] [1, 1, 1]
//       CHECK:   util.return %[[SLICE1]] : tensor<64x?x64xf16>

// -----

util.func public @test_inner_dim_slice(%arg0: tensor<1024x7x7x2xi8>) -> tensor<1024x7x7xi8> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xi8> to tensor<1024x7x7xi8>
  util.return %extracted_slice : tensor<1024x7x7xi8>
}
// CHECK-LABEL:   util.func public @test_inner_dim_slice
// CHECK:         %[[TPOS:[a-zA-Z0-9_]+]] = linalg.transpose
// CHECK:         %[[EXTR:[a-zA-Z0-9_]+]] = tensor.extract_slice %[[TPOS]]
// CHECK-SAME:      tensor<2x1024x7x7xi8> to tensor<1024x7x7xi8>

// -----

util.func public @test_inner_dim_slice(%arg0: tensor<1024x7x7x2xi8>) -> tensor<1024x7x7x1xi8> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xi8> to tensor<1024x7x7x1xi8>
  util.return %extracted_slice : tensor<1024x7x7x1xi8>
}
// CHECK-LABEL:   util.func public @test_inner_dim_slice
// CHECK:         %[[TPOS:[a-zA-Z0-9_]+]] = linalg.transpose
// CHECK:         %[[EXTR:[a-zA-Z0-9_]+]] = tensor.extract_slice %[[TPOS]]
// CHECK-SAME:      tensor<2x1024x7x7xi8> to tensor<1x1024x7x7xi8>
// CHECK:         %[[TPOS2:[a-zA-Z0-9_]+]] = linalg.transpose

// -----

util.func public @test_inner_dim_slice2(%arg0: tensor<1024x7x7x2xi8>) -> tensor<1024x7x7xi8> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xi8> to tensor<1024x7x7xi8>
  util.return %extracted_slice : tensor<1024x7x7xi8>
}
// CHECK-LABEL:   util.func public @test_inner_dim_slice2
// CHECK:         %[[TPOS:[a-zA-Z0-9_]+]] = linalg.transpose
// CHECK:         %[[EXTR:[a-zA-Z0-9_]+]] = tensor.extract_slice %[[TPOS]]
// CHECK-SAME:      tensor<2x1024x7x7xi8> to tensor<1024x7x7xi8>
