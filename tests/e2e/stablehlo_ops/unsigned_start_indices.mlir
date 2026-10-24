// Start indices are clamped before narrowing or losing unsignedness.

func.func @dynamic_slice_unsigned() {
  %operand = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]> : tensor<5x3xi32>
  %u8_bits = util.unfoldable_constant dense<-1> : tensor<i8>
  %u8 = stablehlo.bitcast_convert %u8_bits : (tensor<i8>) -> tensor<ui8>
  %u64_max_bits = util.unfoldable_constant dense<-1> : tensor<i64>
  %u64_max = stablehlo.bitcast_convert %u64_max_bits : (tensor<i64>) -> tensor<ui64>
  %u64_sign_bits = util.unfoldable_constant dense<-9223372036854775808> : tensor<i64>
  %u64_sign = stablehlo.bitcast_convert %u64_sign_bits : (tensor<i64>) -> tensor<ui64>
  %zero8_bits = util.unfoldable_constant dense<0> : tensor<i8>
  %zero8 = stablehlo.bitcast_convert %zero8_bits : (tensor<i8>) -> tensor<ui8>
  %zero64_bits = util.unfoldable_constant dense<0> : tensor<i64>
  %zero64 = stablehlo.bitcast_convert %zero64_bits : (tensor<i64>) -> tensor<ui64>
  %r8 = stablehlo.dynamic_slice %operand, %u8, %zero8, sizes = [2, 3] : (tensor<5x3xi32>, tensor<ui8>, tensor<ui8>) -> tensor<2x3xi32>
  %r64 = stablehlo.dynamic_slice %operand, %u64_max, %zero64, sizes = [2, 3] : (tensor<5x3xi32>, tensor<ui64>, tensor<ui64>) -> tensor<2x3xi32>
  %rsign = stablehlo.dynamic_slice %operand, %u64_sign, %zero64, sizes = [2, 3] : (tensor<5x3xi32>, tensor<ui64>, tensor<ui64>) -> tensor<2x3xi32>
  check.expect_eq_const(%r8, dense<[[10, 11, 12], [13, 14, 15]]> : tensor<2x3xi32>) : tensor<2x3xi32>
  check.expect_eq_const(%r64, dense<[[10, 11, 12], [13, 14, 15]]> : tensor<2x3xi32>) : tensor<2x3xi32>
  check.expect_eq_const(%rsign, dense<[[10, 11, 12], [13, 14, 15]]> : tensor<2x3xi32>) : tensor<2x3xi32>
  return
}
func.func @dynamic_update_slice_unsigned() {
  %operand = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]> : tensor<5x3xi32>
  %update = util.unfoldable_constant dense<[[100, 101, 102], [200, 201, 202]]> : tensor<2x3xi32>
  %u64_max_bits = util.unfoldable_constant dense<-1> : tensor<i64>
  %u64_max = stablehlo.bitcast_convert %u64_max_bits : (tensor<i64>) -> tensor<ui64>
  %zero64_bits = util.unfoldable_constant dense<0> : tensor<i64>
  %zero64 = stablehlo.bitcast_convert %zero64_bits : (tensor<i64>) -> tensor<ui64>
  %result = stablehlo.dynamic_update_slice %operand, %update, %u64_max, %zero64 : (tensor<5x3xi32>, tensor<2x3xi32>, tensor<ui64>, tensor<ui64>) -> tensor<5x3xi32>
  check.expect_eq_const(%result, dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [100, 101, 102], [200, 201, 202]]> : tensor<5x3xi32>) : tensor<5x3xi32>
  return
}
func.func @dynamic_slice_wide_signed() {
  %operand = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]> : tensor<5x3xi32>
  %row = util.unfoldable_constant dense<4294967297> : tensor<i64>
  %col = util.unfoldable_constant dense<0> : tensor<i64>
  %r = stablehlo.dynamic_slice %operand, %row, %col, sizes = [2, 3] : (tensor<5x3xi32>, tensor<i64>, tensor<i64>) -> tensor<2x3xi32>
  check.expect_eq_const(%r, dense<[[10, 11, 12], [13, 14, 15]]> : tensor<2x3xi32>) : tensor<2x3xi32>
  return
}
