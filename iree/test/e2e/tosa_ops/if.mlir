func @if_true_test() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<true> : tensor<i1>
  %1 = iree.unfoldable_constant dense<5> : tensor<i32>
  %2 = "tosa.cond_if"(%0, %1) ( {
  ^bb0(%arg2: tensor<i32>):  // no predecessors
    %c1 = constant dense<1> : tensor<i32>
    %2 = "tosa.add"(%arg2, %c1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%2) : (tensor<i32>) -> ()
  },  {
  ^bb0(%arg2: tensor<i32>):  // no predecessors
    %c1 = constant dense<2> : tensor<i32>
    %3 = "tosa.add"(%arg2, %c1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i1>, tensor<i32>) -> (tensor<i32>)
  check.expect_eq_const(%2, dense<6> : tensor<i32>) : tensor<i32>
  return
}

func @if_false_test() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<false> : tensor<i1>
  %1 = iree.unfoldable_constant dense<5> : tensor<i32>
  %2 = "tosa.cond_if"(%0, %1) ( {
  ^bb0(%arg2: tensor<i32>):  // no predecessors
    %c1 = constant dense<1> : tensor<i32>
    %2 = "tosa.add"(%arg2, %c1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%2) : (tensor<i32>) -> ()
  },  {
  ^bb0(%arg2: tensor<i32>):  // no predecessors
    %c1 = constant dense<2> : tensor<i32>
    %3 = "tosa.add"(%arg2, %c1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i1>, tensor<i32>) -> (tensor<i32>)
  check.expect_eq_const(%2, dense<7> : tensor<i32>) : tensor<i32>
  return
}
