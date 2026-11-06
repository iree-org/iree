// Unsigned start indices are non-negative values: large ones clamp to the last
// valid start, and scatter skips updates whose index is out of bounds.
// A signed i64 start clamps before it narrows to a 32-bit index.

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

func.func @gather_unsigned() {
  %operand = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]> : tensor<5x3xi32>
  %indices_bits = util.unfoldable_constant dense<[[-1], [1]]> : tensor<2x1xi64>
  %indices = stablehlo.bitcast_convert %indices_bits : (tensor<2x1xi64>) -> tensor<2x1xui64>
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 2, 3>, indices_are_sorted = false
  } : (tensor<5x3xi32>, tensor<2x1xui64>) -> tensor<2x2x3xi32>
  check.expect_eq_const(%result, dense<[[[10, 11, 12], [13, 14, 15]], [[4, 5, 6], [7, 8, 9]]]> : tensor<2x2x3xi32>) : tensor<2x2x3xi32>
  return
}

func.func @scatter_unsigned_out_of_bounds() {
  %operand = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]> : tensor<5x3xi32>
  %indices_bits = util.unfoldable_constant dense<[[-1], [255]]> : tensor<2x1xi64>
  %indices = stablehlo.bitcast_convert %indices_bits : (tensor<2x1xi64>) -> tensor<2x1xui64>
  %updates = util.unfoldable_constant dense<[[[100, 101, 102], [200, 201, 202]], [[300, 301, 302], [400, 401, 402]]]> : tensor<2x2x3xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>,
    indices_are_sorted = false, unique_indices = false
  } : (tensor<5x3xi32>, tensor<2x1xui64>, tensor<2x2x3xi32>) -> tensor<5x3xi32>
  check.expect_eq_const(%result, dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]> : tensor<5x3xi32>) : tensor<5x3xi32>
  return
}

// Element (r, c) of the 200x3 operand is 3 * r + c.
func.func @real_dynamic_slice_unsigned() {
  %rows = stablehlo.iota dim = 0 : tensor<200x3xi32>
  %columns = stablehlo.iota dim = 1 : tensor<200x3xi32>
  %three = stablehlo.constant dense<3> : tensor<200x3xi32>
  %scaled = stablehlo.multiply %rows, %three : tensor<200x3xi32>
  %iota = stablehlo.add %scaled, %columns : tensor<200x3xi32>
  %operand = util.optimization_barrier %iota : tensor<200x3xi32>
  %start_bits = util.unfoldable_constant dense<[-106, 1]> : tensor<2xi8>
  %start = stablehlo.bitcast_convert %start_bits : (tensor<2xi8>) -> tensor<2xui8>
  %limit_bits = util.unfoldable_constant dense<[-66, 3]> : tensor<2xi8>
  %limit = stablehlo.bitcast_convert %limit_bits : (tensor<2xi8>) -> tensor<2xui8>
  %strides_bits = util.unfoldable_constant dense<[10, 1]> : tensor<2xi8>
  %strides = stablehlo.bitcast_convert %strides_bits : (tensor<2xi8>) -> tensor<2xui8>
  // Rows 150, 160, 170 and 180 of columns 1 and 2.
  %result = stablehlo.real_dynamic_slice %operand, %start, %limit, %strides : (tensor<200x3xi32>, tensor<2xui8>, tensor<2xui8>, tensor<2xui8>) -> tensor<?x?xi32>
  %static = tensor.cast %result : tensor<?x?xi32> to tensor<4x2xi32>
  check.expect_eq_const(%static, dense<[[451, 452], [481, 482], [511, 512], [541, 542]]> : tensor<4x2xi32>) : tensor<4x2xi32>

  // A size of 190 rows is above the signed range of ui8.
  %long_start_bits = util.unfoldable_constant dense<[10, 0]> : tensor<2xi8>
  %long_start = stablehlo.bitcast_convert %long_start_bits : (tensor<2xi8>) -> tensor<2xui8>
  %long_limit_bits = util.unfoldable_constant dense<[-56, 3]> : tensor<2xi8>
  %long_limit = stablehlo.bitcast_convert %long_limit_bits : (tensor<2xi8>) -> tensor<2xui8>
  %unit_bits = util.unfoldable_constant dense<1> : tensor<2xi8>
  %unit = stablehlo.bitcast_convert %unit_bits : (tensor<2xi8>) -> tensor<2xui8>
  %long = stablehlo.real_dynamic_slice %operand, %long_start, %long_limit, %unit : (tensor<200x3xi32>, tensor<2xui8>, tensor<2xui8>, tensor<2xui8>) -> tensor<?x?xi32>
  %long_static = tensor.cast %long : tensor<?x?xi32> to tensor<190x3xi32>
  %tail = stablehlo.slice %long_static [188:190, 0:3] : (tensor<190x3xi32>) -> tensor<2x3xi32>
  check.expect_eq_const(%tail, dense<[[594, 595, 596], [597, 598, 599]]> : tensor<2x3xi32>) : tensor<2x3xi32>
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
