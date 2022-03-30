// These following tests executes the following pseudo code with different
// constant values:
//
// i = ?, n = ?
// while (i <= 3) {
//    i += n
// }

// i = 4, n = 2
func.func @while_test_iter0() {
  %0 = util.unfoldable_constant dense<4> : tensor<i32>
  %1 = "tosa.while_loop"(%0) ( {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.greater_equal"(%2, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  check.expect_eq_const(%1, dense<4> : tensor<i32>) : tensor<i32>
  return
}

// i = 2, n = 2
func.func @while_test_iter1() {
  %0 = util.unfoldable_constant dense<2> : tensor<i32>
  %1 = "tosa.while_loop"(%0) ( {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.greater_equal"(%2, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  check.expect_eq_const(%1, dense<4> : tensor<i32>) : tensor<i32>
  return
}

// i = 0, n = 2
func.func @while_test_iter2() {
  %0 = util.unfoldable_constant dense<0> : tensor<i32>
  %1 = "tosa.while_loop"(%0) ( {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.greater_equal"(%2, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  check.expect_eq_const(%1, dense<4> : tensor<i32>) : tensor<i32>
  return
}

// i = 0, n = 1
func.func @while_test_iter4() {
  %0 = util.unfoldable_constant dense<0> : tensor<i32>
  %1 = "tosa.while_loop"(%0) ( {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.greater_equal"(%2, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg0: tensor<i32>):
    %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  check.expect_eq_const(%1, dense<4> : tensor<i32>) : tensor<i32>
  return
}
