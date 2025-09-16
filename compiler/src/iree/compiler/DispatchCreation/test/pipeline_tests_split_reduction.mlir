// RUN: iree-opt %s --iree-dispatch-creation-pipeline='split-reduction=true' --split-input-file | FileCheck %s

// CHECK-LABEL: @basic_reduction(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<4096xf32>
util.func public @basic_reduction(%arg0: tensor<4096xf32>) -> tensor<f32> {
  // Check that the split reduction transformation gets applied.
  %1 = arith.constant dense<0.0> : tensor<f32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
  } ins(%arg0 : tensor<4096xf32>) outs(%1 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<f32>
  // CHECK: %[[SPLIT:.+]] = flow.dispatch.workgroups(%[[ARG0]])
  // CHECK:   scf.forall (%{{.+}}) = (0) to (4096) step (1024)
  // CHECK:     linalg.generic {{.+}} ins({{.+}} : tensor<1024xf32>) outs({{.+}} : tensor<f32>)
  // CHECK: %[[RESULT:.+]] = flow.dispatch.workgroups(%[[SPLIT]])
  // CHECK:   linalg.reduce ins({{.+}} : tensor<4xf32>) outs({{.+}} : tensor<f32>)
  // CHECK: return %[[RESULT]]
  util.return %2 : tensor<f32>
}
