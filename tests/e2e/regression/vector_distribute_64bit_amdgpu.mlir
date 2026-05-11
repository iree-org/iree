// 64-bit reduction and arg_compare on AMDGPU through VectorDistribute.

// f64 sum reduction. 8 rows of 256 ones each: each row sums to 256.0.
func.func @reduction_f64_sum() {
  %in = util.unfoldable_constant dense<1.0> : tensor<8x256xf64>
  %cst = arith.constant 0.0 : f64
  %init = tensor.empty() : tensor<8xf64>
  %fill = linalg.fill ins(%cst : f64) outs(%init : tensor<8xf64>) -> tensor<8xf64>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<8x256xf64>) outs(%fill : tensor<8xf64>) {
    ^bb0(%a: f64, %b: f64):
      %2 = arith.addf %a, %b : f64
      linalg.yield %2 : f64
    } -> tensor<8xf64>
  check.expect_eq_const(%result, dense<256.0> : tensor<8xf64>) : tensor<8xf64>
  return
}

// i64 arg_compare (argmax). Length-256 input is a subgroup-compatible size
// that selects VectorDistribute (small lengths land on TileAndFuse and skip
// the wide-shuffle path entirely). All elements are 1 except index 100 = 7,
// so the maximum is 7 at index 100. Putting the max at a non-last index
// catches a buggy "always returns the last lane" reducer.
func.func @argcompare_i64_argmax() {
  %ones = util.unfoldable_constant dense<1> : tensor<256xi64>
  %c7_i64 = arith.constant 7 : i64
  %c100 = arith.constant 100 : index
  %in_i64 = tensor.insert %c7_i64 into %ones[%c100] : tensor<256xi64>
  %int_min = arith.constant -9223372036854775808 : i64
  %c0 = arith.constant 0 : i32
  %init_v_empty = tensor.empty() : tensor<i64>
  %init_i_empty = tensor.empty() : tensor<i32>
  %init_v = linalg.fill ins(%int_min : i64) outs(%init_v_empty : tensor<i64>) -> tensor<i64>
  %init_i = linalg.fill ins(%c0 : i32) outs(%init_i_empty : tensor<i32>) -> tensor<i32>
  %res:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%in_i64 : tensor<256xi64>)
    outs(%init_v, %init_i : tensor<i64>, tensor<i32>) {
    ^bb0(%a: i64, %b: i64):
      %cmp = arith.cmpi sgt, %a, %b : i64
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<i64>, tensor<i32>
  check.expect_eq_const(%res#0, dense<7> : tensor<i64>) : tensor<i64>
  check.expect_eq_const(%res#1, dense<100> : tensor<i32>) : tensor<i32>
  return
}

// f64 arg_compare (argmax). Same shape and rationale as @argcompare_i64_argmax
// on the floating-point path: 256 elements of 1.0 with 7.0 at index 100. The
// value seed (-1.0) is smaller than every input so the first comparison
// establishes the running max.
func.func @argcompare_f64_argmax() {
  %ones = util.unfoldable_constant dense<1.0> : tensor<256xf64>
  %c7_f64 = arith.constant 7.0 : f64
  %c100 = arith.constant 100 : index
  %in_f64 = tensor.insert %c7_f64 into %ones[%c100] : tensor<256xf64>
  %seed = arith.constant -1.0 : f64
  %c0 = arith.constant 0 : i32
  %init_v_empty = tensor.empty() : tensor<f64>
  %init_i_empty = tensor.empty() : tensor<i32>
  %init_v = linalg.fill ins(%seed : f64) outs(%init_v_empty : tensor<f64>) -> tensor<f64>
  %init_i = linalg.fill ins(%c0 : i32) outs(%init_i_empty : tensor<i32>) -> tensor<i32>
  %res:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%in_f64 : tensor<256xf64>)
    outs(%init_v, %init_i : tensor<f64>, tensor<i32>) {
    ^bb0(%a: f64, %b: f64):
      %cmp = arith.cmpf ogt, %a, %b : f64
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f64>, tensor<i32>
  check.expect_eq_const(%res#0, dense<7.0> : tensor<f64>) : tensor<f64>
  check.expect_eq_const(%res#1, dense<100> : tensor<i32>) : tensor<i32>
  return
}
