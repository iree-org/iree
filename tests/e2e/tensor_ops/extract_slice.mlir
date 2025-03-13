func.func public @extract_slice_i32_offset1_size2_stride1() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %2 = util.optimization_barrier %1 : tensor<4xi32>
  %extracted_slice = tensor.extract_slice %2[1] [2] [1] : tensor<4xi32> to tensor<2xi32>
  check.expect_eq_const(%extracted_slice, dense<[2, 3]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func public @extract_slice_i64_offset1_size2_stride1() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %2 = util.optimization_barrier %1 : tensor<4xi64>
  %extracted_slice = tensor.extract_slice %2[1] [2] [1] : tensor<4xi64> to tensor<2xi64>
  check.expect_eq_const(%extracted_slice, dense<[2, 3]> : tensor<2xi64>) : tensor<2xi64>
  return
}

func.func public @extract_slice_i32_offset1_size2_stride2() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %2 = util.optimization_barrier %1 : tensor<4xi32>
  %extracted_slice = tensor.extract_slice %2[1] [2] [2] : tensor<4xi32> to tensor<2xi32>
  check.expect_eq_const(%extracted_slice, dense<[2, 4]> : tensor<2xi32>) : tensor<2xi32>
  return
}
