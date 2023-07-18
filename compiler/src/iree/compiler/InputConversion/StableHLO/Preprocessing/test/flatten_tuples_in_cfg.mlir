// RUN: iree-opt --iree-stablehlo-preprocessing-flatten-tuples %s | FileCheck %s

// CHECK-LABEL: @flatten_func
module @flatten_func {
  // CHECK: func.func @caller(%arg0: i1, %arg1: tensor<f32>) -> tensor<f32>
  func.func @caller(%arg0 : i1, %arg1: tensor<f32>) -> tensor<f32> {
    // CHECK: %[[RESULT:.*]]:2 = call @callee(%arg0, %arg1, %arg1, %arg1) : (i1, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %0 = stablehlo.tuple %arg1, %arg1 : tuple<tensor<f32>, tensor<f32>>
    %1 = stablehlo.tuple %arg1, %0 : tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>
    %2 = call @callee(%arg0, %1) : (i1, tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>>
    %3 = stablehlo.get_tuple_element %2[0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    // CHECK: return %[[RESULT]]#0 : tensor<f32>
    return %3 : tensor<f32>
  }

  // CHECK: func.func private @callee(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.func private @callee(%arg0: i1, %arg1: tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>> {
    // CHECK-NEXT: cf.cond_br %arg0, ^[[BB:.+]](%arg2, %arg3 : tensor<f32>, tensor<f32>), ^bb2(%arg1 : tensor<f32>)
    // CHECK:      ^[[BB]](%[[V0:[^:]+]]: tensor<f32>, %[[V1:[^:]+]]: tensor<f32>)
    // CHECK-NEXT:   return %[[V0]], %[[V1]] : tensor<f32>, tensor<f32>
    %0 = stablehlo.get_tuple_element %arg1[0] : (tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tensor<f32>
    %1 = stablehlo.get_tuple_element %arg1[1] : (tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>>
    cf.cond_br %arg0, ^bb1(%1 : tuple<tensor<f32>, tensor<f32>>), ^bb2(%0 : tensor<f32>)
  ^bb1(%phi0 : tuple<tensor<f32>, tensor<f32>>):
    return %phi0 : tuple<tensor<f32>, tensor<f32>>
  ^bb2(%phi1 : tensor<f32>):
    %2 = stablehlo.tuple %phi1, %phi1 : tuple<tensor<f32>, tensor<f32>>
    cf.br ^bb1(%2 : tuple<tensor<f32>, tensor<f32>>)
  }
}
