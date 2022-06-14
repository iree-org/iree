// RUN: iree-opt --split-input-file --iree-mhlo-flatten-tuples-in-cfg --canonicalize %s | FileCheck %s
// We rely on canonicalization to cancel out tuple/get_element operations, so
// we test this followed by the canonicalizer rather than just the pass in
// isolation.
// TODO: It would be better if the pass was standalone.

// CHECK-LABEL: @flatten_func
module @flatten_func {
  // CHECK: func.func @caller(%arg0: i1, %arg1: tensor<f32>) -> tensor<f32>
  func.func @caller(%arg0 : i1, %arg1: tensor<f32>) -> tensor<f32> {
    // CHECK: %[[RESULT:.*]]:2 = call @callee(%arg0, %arg1, %arg1, %arg1) : (i1, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %0 = "mhlo.tuple"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
    %1 = "mhlo.tuple"(%arg1, %0) : (tensor<f32>, tuple<tensor<f32>, tensor<f32>>) -> tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>
    %2 = call @callee(%arg0, %1) : (i1, tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>>
    %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    // CHECK: return %[[RESULT]]#0 : tensor<f32>
    return %3 : tensor<f32>
  }

  // CHECK: func.func private @callee(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.func private @callee(%arg0: i1, %arg1: tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>> {
    // CHECK-DAG: %[[RESULT0:.*]] = arith.select %arg0, %arg2, %arg1 : tensor<f32>
    // CHECK-DAG: %[[RESULT1:.*]] = arith.select %arg0, %arg3, %arg1 : tensor<f32>
    // CHECK: return %[[RESULT0]], %[[RESULT1]] : tensor<f32>, tensor<f32>
    %0 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tensor<f32>
    %1 = "mhlo.get_tuple_element"(%arg1) {index = 1 : i32} : (tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>>
    cf.cond_br %arg0, ^bb1(%1 : tuple<tensor<f32>, tensor<f32>>), ^bb2(%0 : tensor<f32>)
  ^bb1(%phi0 : tuple<tensor<f32>, tensor<f32>>):
    return %phi0 : tuple<tensor<f32>, tensor<f32>>
  ^bb2(%phi1 : tensor<f32>):
    %2 = "mhlo.tuple"(%phi1, %phi1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
    cf.br ^bb1(%2 : tuple<tensor<f32>, tensor<f32>>)
  }
}
