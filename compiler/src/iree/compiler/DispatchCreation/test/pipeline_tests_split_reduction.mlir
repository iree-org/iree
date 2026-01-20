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

// -----

// CHECK-LABEL: @basic_arg_compare(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<4096xf32>
util.func public @basic_arg_compare(%arg0: tensor<4096xf32>)
    -> (tensor<f32>, tensor<i32>) {
  %c0f = arith.constant 0.0 : f32
  %c0i = arith.constant 0 : i32

  %init_val = tensor.empty() : tensor<f32>
  %init_idx = tensor.empty() : tensor<i32>
  %filled_val = linalg.fill ins(%c0f : f32)
                 outs(%init_val : tensor<f32>) -> tensor<f32>
  %filled_idx = linalg.fill ins(%c0i : i32)
                 outs(%init_idx : tensor<i32>) -> tensor<i32>

  %res_val, %res_idx = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%arg0 : tensor<4096xf32>)
    outs(%filled_val, %filled_idx : tensor<f32>, tensor<i32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  // First level: tile reduction in 1024-size chunks.
  // CHECK: %[[SPLIT:.+]]:2 = flow.dispatch.workgroups(%[[ARG0]]) : (tensor<4096xf32>) -> (tensor<4xf32>, tensor<4xi32>)
  // CHECK:   scf.forall ({{.*}}) = (0) to (4096) step (1024)
  // CHECK:     iree_linalg_ext.arg_compare
  // CHECK-SAME: ins({{.*}} : tensor<1024xf32>)

  // Second level: merge 4 partials via arg_compare with explicit-index mode.
  // CHECK: %[[RESULT:.+]]:2 = flow.dispatch.workgroups(%[[SPLIT]]#0, %[[SPLIT]]#1)
  // CHECK:   iree_linalg_ext.arg_compare
  // CHECK-SAME: ins({{.*}} : tensor<4xf32>, tensor<4xi32>)
  // CHECK-SAME: outs({{.*}} : tensor<f32>, tensor<i32>)

  // CHECK: util.return %[[RESULT]]#0, %[[RESULT]]#1

  util.return %res_val, %res_idx : tensor<f32>, tensor<i32>
}
