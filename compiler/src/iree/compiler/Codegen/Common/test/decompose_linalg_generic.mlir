// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-linalg-generic))" %s | FileCheck %s

// -----
// CHECK-LABEL: @parallel_with_broadcast_dynamic
func.func @parallel_with_broadcast_dynamic(
    %arg0 : tensor<64xf32>, %arg1 : tensor<?x64xf32>) -> tensor<?x64xf32> {
  // Just verify that the upstream patterns are working. They should produce
  // two generics: one with an add and the next with a mul
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  %12 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%arg1 : tensor<?x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %13 = arith.addf %arg2, %arg2 : f32
    %14 = arith.mulf %13, %arg2 : f32
    linalg.yield %14 : f32
  } -> tensor<?x64xf32>
  func.return %12 : tensor<?x64xf32>
}
