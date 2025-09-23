func.func @if_true_test() {
  %0 = util.unfoldable_constant dense<true> : tensor<i1>
  %1 = util.unfoldable_constant dense<10> : tensor<i32>
  %path = util.unfoldable_constant 1 : i32
  %2 = tosa.cond_if %0 : tensor<i1> -> tensor<i32> {
    check.expect_true(%path) : i32
    %3 = util.unfoldable_constant dense<10> : tensor<i32>
    %4 = tosa.add %1, %3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tosa.yield %4 : tensor<i32>
  } else {
    check.expect_false(%path) : i32
    tosa.yield %1 : tensor<i32>
  }
  check.expect_eq_const(%2, dense<20> : tensor<i32>) : tensor<i32>
  return
}

func.func @if_false_test() {
  %0 = util.unfoldable_constant dense<false> : tensor<i1>
  %1 = util.unfoldable_constant dense<10> : tensor<i32>
  %path = util.unfoldable_constant 0 : i32
  %2 = tosa.cond_if %0 : tensor<i1> -> tensor<i32> {
    check.expect_true(%path) : i32
    tosa.yield %1 : tensor<i32>
  } else {
    check.expect_false(%path) : i32
    %3 = util.unfoldable_constant dense<10> : tensor<i32>
    %4 = tosa.add %1, %3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tosa.yield %4 : tensor<i32>
  }
  check.expect_eq_const(%2, dense<20> : tensor<i32>) : tensor<i32>
  return
}
