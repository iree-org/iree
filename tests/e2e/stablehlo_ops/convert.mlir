func.func @narrow_int_i32_i8() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi8>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi8>) : tensor<3xi8>
  return
}

func.func @widen_int_i8_i32() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi8>
  %res = "stablehlo.convert"(%input) : (tensor<3xi8>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @narrow_int_i32_i16() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi16>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi16>) : tensor<3xi16>
  return
}

func.func @widen_int_i16_i32() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi16>
  %res = "stablehlo.convert"(%input) : (tensor<3xi16>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @narrow_int_i64_i32() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi64>
  %res = "stablehlo.convert"(%input) : (tensor<3xi64>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @widen_int_i32_i64() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi64>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi64>) : tensor<3xi64>
  return
}

func.func @int_to_float() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[-42.0, 0.0, 42.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

// TODO(#6160): XLA does not specify the rounding behavior, meaning that we
// can't test something like -10.5 as that could be -11 (roundf) or -10 (rint
// with round-to-even mode).
//
// For casting rules, see
// https://www.tensorflow.org/xla/operation_semantics#convertelementtype
// func.func @float_to_int() {
//   %input = util.unfoldable_constant dense<[-10.5, -4.4, 4.4, 10.5]> : tensor<4xf32>
//   %res = "stablehlo.convert"(%input) : (tensor<4xf32>) -> tensor<4xi32>
//   check.expect_eq_const(%res, dense<[-10, -4, 4, 10]> : tensor<4xi32>) : tensor<4xi32>
//   return
// }


// Ties round to even, including underflow and the subnormal/normal boundary.
// Compare storage bits so signed zeros and overflow cannot pass by tolerance.
func.func @f32_to_f16_rounding_boundaries() {
  %input = util.unfoldable_constant dense<[0x00000000, 0x80000000, 0x33000000, 0x33000001, 0xB3000000, 0xB3000001, 0x33800000, 0x387FDFFF, 0x387FE000, 0x38800000, 0x3F801000, 0x3F801001, 0x477FE000, 0x477FF000, 0x7F800000, 0xFF800000]> : tensor<16xf32>
  %half = stablehlo.convert %input : (tensor<16xf32>) -> tensor<16xf16>
  %bits = stablehlo.bitcast_convert %half : (tensor<16xf16>) -> tensor<16xi16>
  check.expect_eq_const(%bits, dense<[0, -32768, 0, 1, -32768, -32767, 1, 1023, 1024, 1024, 15360, 15361, 31743, 31744, 31744, -1024]> : tensor<16xi16>) : tensor<16xi16>
  return
}

func.func @f16_to_f32_exact_boundaries() {
  %input_bits = util.unfoldable_constant dense<[0, -32768, 1, -32767, 1023, 1024, 15360, 31743, 31744, -1024]> : tensor<10xi16>
  %half = stablehlo.bitcast_convert %input_bits : (tensor<10xi16>) -> tensor<10xf16>
  %float = stablehlo.convert %half : (tensor<10xf16>) -> tensor<10xf32>
  %bits = stablehlo.bitcast_convert %float : (tensor<10xf32>) -> tensor<10xi32>
  check.expect_eq_const(%bits, dense<[0x00000000, 0x80000000, 0x33800000, 0xB3800000, 0x387FC000, 0x38800000, 0x3F800000, 0x477FE000, 0x7F800000, 0xFF800000]> : tensor<10xi32>) : tensor<10xi32>
  return
}

// The carry from the rounded subnormal becomes the smallest normal value.
func.func @f32_to_f8e5m2_subnormal_carry() {
  %input = util.unfoldable_constant dense<[0x385FFFFF, 0x38600000, 0x38600001]> : tensor<3xf32>
  %narrow = stablehlo.convert %input : (tensor<3xf32>) -> tensor<3xf8E5M2>
  %bits = stablehlo.bitcast_convert %narrow : (tensor<3xf8E5M2>) -> tensor<3xi8>
  check.expect_eq_const(%bits, dense<[3, 4, 4]> : tensor<3xi8>) : tensor<3xi8>
  return
}

// Widening all arithmetic without retaining the intermediate rounding would
// produce one instead of zero in these cancellation cases.
func.func @f16_intermediate_rounding() {
  %large = util.unfoldable_constant dense<2048.0> : tensor<f16>
  %one = util.unfoldable_constant dense<1.0> : tensor<f16>
  %sum = stablehlo.add %large, %one : tensor<f16>
  %result = stablehlo.subtract %sum, %large : tensor<f16>
  check.expect_eq_const(%result, dense<0.0> : tensor<f16>) : tensor<f16>
  return
}

func.func @bf16_intermediate_rounding() {
  %large = util.unfoldable_constant dense<256.0> : tensor<bf16>
  %one = util.unfoldable_constant dense<1.0> : tensor<bf16>
  %sum = stablehlo.add %large, %one : tensor<bf16>
  %result = stablehlo.subtract %sum, %large : tensor<bf16>
  check.expect_eq_const(%result, dense<0.0> : tensor<bf16>) : tensor<bf16>
  return
}

func.func @f16_conversion_preserves_nan() {
  %input = util.unfoldable_constant dense<[0x7FC00000, 0xFFC00000]> : tensor<2xf32>
  %half = stablehlo.convert %input : (tensor<2xf32>) -> tensor<2xf16>
  %nan = "stablehlo.compare"(%half, %half) {comparison_direction = #stablehlo<comparison_direction NE>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<2xf16>, tensor<2xf16>) -> tensor<2xi1>
  check.expect_eq_const(%nan, dense<true> : tensor<2xi1>) : tensor<2xi1>
  %float = stablehlo.convert %half : (tensor<2xf16>) -> tensor<2xf32>
  %wide_nan = "stablehlo.compare"(%float, %float) {comparison_direction = #stablehlo<comparison_direction NE>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  check.expect_eq_const(%wide_nan, dense<true> : tensor<2xi1>) : tensor<2xi1>
  return
}


func.func @f32_bf16_round_trip() {
  %input = util.unfoldable_constant dense<257.0> : tensor<f32>
  %narrow = stablehlo.convert %input : (tensor<f32>) -> tensor<bf16>
  %wide = stablehlo.convert %narrow : (tensor<bf16>) -> tensor<f32>
  check.expect_eq_const(%wide, dense<256.0> : tensor<f32>) : tensor<f32>
  return
}
