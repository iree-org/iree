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
  // CHECK:   scf.forall (%{{.+}}) = (0) to (4096) step (64)
  // CHECK:     linalg.generic {{.+}} ins({{.+}} : tensor<64xf32>) outs({{.+}} : tensor<f32>)
  // CHECK: %[[RESULT:.+]] = flow.dispatch.workgroups(%[[SPLIT]])
  // CHECK:   linalg.reduce ins({{.+}} : tensor<64xf32>) outs({{.+}} : tensor<f32>)
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

  // First level: tile reduction in 64-size chunks.
  // CHECK: %[[SPLIT:.+]]:2 = flow.dispatch.workgroups(%[[ARG0]]) : (tensor<4096xf32>) -> (tensor<64xf32>, tensor<64xi32>)
  // CHECK:   scf.forall ({{.*}}) = (0) to (4096) step (64)
  // CHECK:     iree_linalg_ext.arg_compare
  // CHECK-SAME: ins({{.*}} : tensor<64xf32>)

  // Second level: merge 64 partials via arg_compare with explicit-index mode.
  // CHECK: %[[RESULT:.+]]:2 = flow.dispatch.workgroups(%[[SPLIT]]#0, %[[SPLIT]]#1)
  // CHECK:   iree_linalg_ext.arg_compare
  // CHECK-SAME: ins({{.*}} : tensor<64xf32>, tensor<64xi32>)
  // CHECK-SAME: outs({{.*}} : tensor<f32>, tensor<i32>)

  // CHECK: util.return %[[RESULT]]#0, %[[RESULT]]#1

  util.return %res_val, %res_idx : tensor<f32>, tensor<i32>
}

// -----

// CHECK-LABEL: @coupled_argmax(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1025x8xi32>
util.func public @coupled_argmax(%arg0: tensor<1025x8xi32>)
    -> (tensor<8xi32>, tensor<8xi64>) {
  // Check that dispatch formation succeeds for a coupled value/index reduction
  // that cannot be split.
  // CHECK: flow.dispatch.workgroups
  // CHECK: util.return
  %min = arith.constant -2147483648 : i32
  %last = arith.constant 9223372036854775807 : i64
  %empty_values = tensor.empty() : tensor<8xi32>
  %values = linalg.fill ins(%min : i32)
      outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
  %empty_indices = tensor.empty() : tensor<8xi64>
  %indices = linalg.fill ins(%last : i64)
      outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
  %result:2 = linalg.generic {
      indexing_maps = [affine_map<(r, c) -> (r, c)>,
                       affine_map<(r, c) -> (c)>,
                       affine_map<(r, c) -> (c)>],
      iterator_types = ["reduction", "parallel"]
  } ins(%arg0 : tensor<1025x8xi32>)
    outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
  ^bb0(%incoming: i32, %best: i32, %best_index: i64):
    %r = linalg.index 0 : index
    %index = arith.index_cast %r : index to i64
    %maximum = arith.maxsi %incoming, %best : i32
    %greater = arith.cmpi sgt, %incoming, %best : i32
    %equal = arith.cmpi eq, %incoming, %best : i32
    %earlier = arith.cmpi slt, %index, %best_index : i64
    %tie = arith.andi %equal, %earlier : i1
    %take = arith.ori %greater, %tie : i1
    %selected = arith.select %take, %index, %best_index : i64
    linalg.yield %maximum, %selected : i32, i64
  } -> (tensor<8xi32>, tensor<8xi64>)
  util.return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
}
