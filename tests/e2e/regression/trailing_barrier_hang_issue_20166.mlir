func.func @trailing_barrier_hang_1_iters_issue_20166(){
  %iters = util.unfoldable_constant 1 : i64
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %init = flow.tensor.dynamic_constant dense<> : tensor<0xi64> -> tensor<?xi64>
  %result = scf.for %i = %c0 to %iters step %c1 iter_args(%iter_arg = %init) -> (tensor<?xi64>) : i64 {
    %empty = tensor.empty() : tensor<1xi64>
    %fill = linalg.fill ins(%i : i64) outs(%empty : tensor<1xi64>) -> tensor<1xi64>
    %concat = tensor.concat dim(0) %iter_arg, %fill : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
    scf.yield %concat : tensor<?xi64>
  }
  %static = tensor.cast %result : tensor<?xi64> to tensor<1xi64>
  check.expect_eq_const(%static, dense<[0]> : tensor<1xi64>) : tensor<1xi64>
  return
}

func.func @trailing_barrier_hang_10_iters_issue_20166(){
  %iters = util.unfoldable_constant 10 : i64
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %init = flow.tensor.dynamic_constant dense<> : tensor<0xi64> -> tensor<?xi64>
  %result = scf.for %i = %c0 to %iters step %c1 iter_args(%iter_arg = %init) -> (tensor<?xi64>) : i64 {
    %empty = tensor.empty() : tensor<1xi64>
    %fill = linalg.fill ins(%i : i64) outs(%empty : tensor<1xi64>) -> tensor<1xi64>
    %concat = tensor.concat dim(0) %iter_arg, %fill : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
    scf.yield %concat : tensor<?xi64>
  }
  %static = tensor.cast %result : tensor<?xi64> to tensor<10xi64>
  check.expect_eq_const(%static, dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi64>) : tensor<10xi64>
  return
}

func.func @trailing_barrier_hang_0_iters_issue_20166(){
  %iters = util.unfoldable_constant 0 : i64
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %init = flow.tensor.dynamic_constant dense<> : tensor<0xi64> -> tensor<?xi64>
  %result = scf.for %i = %c0 to %iters step %c1 iter_args(%iter_arg = %init) -> (tensor<?xi64>) : i64 {
    %empty = tensor.empty() : tensor<1xi64>
    %fill = linalg.fill ins(%i : i64) outs(%empty : tensor<1xi64>) -> tensor<1xi64>
    %concat = tensor.concat dim(0) %iter_arg, %fill : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
    scf.yield %concat : tensor<?xi64>
  }
  %static = tensor.cast %result : tensor<?xi64> to tensor<0xi64>
  check.expect_eq_const(%static, dense<> : tensor<0xi64>) : tensor<0xi64>
  return
}
