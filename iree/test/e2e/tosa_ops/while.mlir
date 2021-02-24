func @while_test() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<0> : tensor<i32>
  %1 = "tosa.while_loop"(%0) ( {
  ^bb0(%arg2: tensor<i32>):  // no predecessors
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.greater_equal"(%2, %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<i32>):  // no predecessors
    %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg2, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  check.expect_eq_const(%1, dense<4> : tensor<i32>) : tensor<i32>
  return
}
