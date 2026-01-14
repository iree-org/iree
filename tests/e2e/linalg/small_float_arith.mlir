func.func @add_f8E5M2FNUZ() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf8E5M2FNUZ>
  %init = tensor.empty() : tensor<4xf8E5M2FNUZ>
  %add = linalg.add
    ins(%input, %input : tensor<4xf8E5M2FNUZ>, tensor<4xf8E5M2FNUZ>)
    outs(%init : tensor<4xf8E5M2FNUZ>) -> tensor<4xf8E5M2FNUZ>
  check.expect_almost_eq_const(%add, dense<[0.0, 2.0, 4.0, 8.0]> : tensor<4xf8E5M2FNUZ>) : tensor<4xf8E5M2FNUZ>
  return
}

func.func @sub_f8E5M2FNUZ() {
  %lhs = util.unfoldable_constant dense<[4.0, 8.0, 16.0, 32.0]> : tensor<4xf8E5M2FNUZ>
  %rhs = util.unfoldable_constant dense<[1.0, 2.0, 4.0, 8.0]> : tensor<4xf8E5M2FNUZ>
  %init = tensor.empty() : tensor<4xf8E5M2FNUZ>
  %sub = linalg.sub
    ins(%lhs, %rhs : tensor<4xf8E5M2FNUZ>, tensor<4xf8E5M2FNUZ>)
    outs(%init : tensor<4xf8E5M2FNUZ>) -> tensor<4xf8E5M2FNUZ>
  check.expect_almost_eq_const(%sub, dense<[3.0, 6.0, 12.0, 24.0]> : tensor<4xf8E5M2FNUZ>) : tensor<4xf8E5M2FNUZ>
  return
}

func.func @mul_f8E5M2FNUZ() {
  %lhs = util.unfoldable_constant dense<[1.0, 2.0, 4.0, 8.0]> : tensor<4xf8E5M2FNUZ>
  %rhs = util.unfoldable_constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf8E5M2FNUZ>
  %init = tensor.empty() : tensor<4xf8E5M2FNUZ>
  %mul = linalg.mul
    ins(%lhs, %rhs : tensor<4xf8E5M2FNUZ>, tensor<4xf8E5M2FNUZ>)
    outs(%init : tensor<4xf8E5M2FNUZ>) -> tensor<4xf8E5M2FNUZ>
  check.expect_almost_eq_const(%mul, dense<[2.0, 4.0, 8.0, 16.0]> : tensor<4xf8E5M2FNUZ>) : tensor<4xf8E5M2FNUZ>
  return
}

func.func @negf_f8E5M2FNUZ() {
  %input = util.unfoldable_constant dense<[1.0, -2.0, 4.0, -8.0]> : tensor<4xf8E5M2FNUZ>
  %init = tensor.empty() : tensor<4xf8E5M2FNUZ>
  %neg = linalg.negf
    ins(%input : tensor<4xf8E5M2FNUZ>)
    outs(%init : tensor<4xf8E5M2FNUZ>) -> tensor<4xf8E5M2FNUZ>
  check.expect_almost_eq_const(%neg, dense<[-1.0, 2.0, -4.0, 8.0]> : tensor<4xf8E5M2FNUZ>) : tensor<4xf8E5M2FNUZ>
  return
}

func.func @add_f8E4M3FNUZ() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf8E4M3FNUZ>
  %init = tensor.empty() : tensor<4xf8E4M3FNUZ>
  %add = linalg.add
    ins(%input, %input : tensor<4xf8E4M3FNUZ>, tensor<4xf8E4M3FNUZ>)
    outs(%init : tensor<4xf8E4M3FNUZ>) -> tensor<4xf8E4M3FNUZ>
  check.expect_almost_eq_const(%add, dense<[0.0, 2.0, 4.0, 8.0]> : tensor<4xf8E4M3FNUZ>) : tensor<4xf8E4M3FNUZ>
  return
}

func.func @mul_f8E4M3FNUZ() {
  %lhs = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf8E4M3FNUZ>
  %rhs = util.unfoldable_constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf8E4M3FNUZ>
  %init = tensor.empty() : tensor<4xf8E4M3FNUZ>
  %mul = linalg.mul
    ins(%lhs, %rhs : tensor<4xf8E4M3FNUZ>, tensor<4xf8E4M3FNUZ>)
    outs(%init : tensor<4xf8E4M3FNUZ>) -> tensor<4xf8E4M3FNUZ>
  check.expect_almost_eq_const(%mul, dense<[2.0, 4.0, 6.0, 8.0]> : tensor<4xf8E4M3FNUZ>) : tensor<4xf8E4M3FNUZ>
  return
}

func.func @add_f8E5M2() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf8E5M2>
  %init = tensor.empty() : tensor<4xf8E5M2>
  %add = linalg.add
    ins(%input, %input : tensor<4xf8E5M2>, tensor<4xf8E5M2>)
    outs(%init : tensor<4xf8E5M2>) -> tensor<4xf8E5M2>
  check.expect_almost_eq_const(%add, dense<[0.0, 2.0, 4.0, 8.0]> : tensor<4xf8E5M2>) : tensor<4xf8E5M2>
  return
}

func.func @add_f8E4M3FN() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf8E4M3FN>
  %init = tensor.empty() : tensor<4xf8E4M3FN>
  %add = linalg.add
    ins(%input, %input : tensor<4xf8E4M3FN>, tensor<4xf8E4M3FN>)
    outs(%init : tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN>
  check.expect_almost_eq_const(%add, dense<[0.0, 2.0, 4.0, 8.0]> : tensor<4xf8E4M3FN>) : tensor<4xf8E4M3FN>
  return
}

//===----------------------------------------------------------------------===//
// Special value tests for each fp8 type.
//===----------------------------------------------------------------------===//

// f8E5M2FNUZ: No Inf, No -0, NaN = 0x80. Max finite = 57344.0.
// Test vector:
//   [0]: Zero + Zero = Zero.
//   [1]: Normal + Normal = Normal.
//   [2]: Positive overflow → NaN.
//   [3]: Near-max (no overflow).
//   [4]: NaN + x = NaN (NaN propagation).
//   [5]: Negative overflow → NaN.
//   [6]: Small + Small = Small.
//   [7]: Negative normal.
func.func @special_f8E5M2FNUZ() {
  %lhs = util.unfoldable_constant dense<[0.0, 4.0, 32768.0, 49152.0, 0x80, -32768.0, 0.125, -4.0]> : tensor<8xf8E5M2FNUZ>
  %rhs = util.unfoldable_constant dense<[0.0, 4.0, 32768.0, 1.0, 1.0, -32768.0, 0.125, -4.0]> : tensor<8xf8E5M2FNUZ>
  %init = tensor.empty() : tensor<8xf8E5M2FNUZ>
  %add = linalg.add
    ins(%lhs, %rhs : tensor<8xf8E5M2FNUZ>, tensor<8xf8E5M2FNUZ>)
    outs(%init : tensor<8xf8E5M2FNUZ>) -> tensor<8xf8E5M2FNUZ>
  check.expect_almost_eq_const(%add, dense<[0.0, 8.0, 0x80, 49152.0, 0x80, 0x80, 0.25, -8.0]> : tensor<8xf8E5M2FNUZ>) : tensor<8xf8E5M2FNUZ>
  return
}

// f8E4M3FNUZ: No Inf, No -0, NaN = 0x80. Max finite = 240.0.
// Test vector:
//   [0]: Zero + Zero = Zero.
//   [1]: Normal + Normal = Normal.
//   [2]: Positive overflow → NaN.
//   [3]: Near-max (no overflow).
//   [4]: NaN + x = NaN (NaN propagation).
//   [5]: Negative overflow → NaN.
//   [6]: Small + Small = Small.
//   [7]: Negative normal.
func.func @special_f8E4M3FNUZ() {
  %lhs = util.unfoldable_constant dense<[0.0, 4.0, 192.0, 224.0, 0x80, -192.0, 0.125, -4.0]> : tensor<8xf8E4M3FNUZ>
  %rhs = util.unfoldable_constant dense<[0.0, 4.0, 192.0, 1.0, 1.0, -192.0, 0.125, -4.0]> : tensor<8xf8E4M3FNUZ>
  %init = tensor.empty() : tensor<8xf8E4M3FNUZ>
  %add = linalg.add
    ins(%lhs, %rhs : tensor<8xf8E4M3FNUZ>, tensor<8xf8E4M3FNUZ>)
    outs(%init : tensor<8xf8E4M3FNUZ>) -> tensor<8xf8E4M3FNUZ>
  check.expect_almost_eq_const(%add, dense<[0.0, 8.0, 0x80, 224.0, 0x80, 0x80, 0.25, -8.0]> : tensor<8xf8E4M3FNUZ>) : tensor<8xf8E4M3FNUZ>
  return
}

// f8E5M2: Has Inf, Has -0, IEEE-like. Max finite = 57344.0, +Inf = 0x7C, -Inf = 0xFC.
// Test vector:
//   [0]: Zero + Zero = Zero.
//   [1]: -0 + 0 = +0.
//   [2]: Positive overflow → +Inf.
//   [3]: +Inf + x = +Inf.
//   [4]: NaN + x = NaN (NaN propagation).
//   [5]: Negative overflow → -Inf.
//   [6]: -Inf + x = -Inf.
//   [7]: +Inf + -Inf = NaN.
func.func @special_f8E5M2() {
  %lhs = util.unfoldable_constant dense<[0.0, -0.0, 32768.0, 0x7C, 0x7F, -32768.0, 0xFC, 0x7C]> : tensor<8xf8E5M2>
  %rhs = util.unfoldable_constant dense<[0.0, 0.0, 32768.0, 1.0, 1.0, -32768.0, -1.0, 0xFC]> : tensor<8xf8E5M2>
  %init = tensor.empty() : tensor<8xf8E5M2>
  %add = linalg.add
    ins(%lhs, %rhs : tensor<8xf8E5M2>, tensor<8xf8E5M2>)
    outs(%init : tensor<8xf8E5M2>) -> tensor<8xf8E5M2>
  check.expect_almost_eq_const(%add, dense<[0.0, 0.0, 0x7C, 0x7C, 0x7F, 0xFC, 0xFC, 0x7F]> : tensor<8xf8E5M2>) : tensor<8xf8E5M2>
  return
}

// f8E4M3FN: No Inf, Has -0, NaN = 0x7F. Max finite = 448.0.
// Test vector:
//   [0]: Zero + Zero = Zero.
//   [1]: -0 + 0 = +0.
//   [2]: Positive overflow → NaN.
//   [3]: Near-max (no overflow).
//   [4]: NaN + x = NaN (NaN propagation).
//   [5]: Negative overflow → NaN.
//   [6]: Small + Small = Small.
//   [7]: Negative normal.
func.func @special_f8E4M3FN() {
  %lhs = util.unfoldable_constant dense<[0.0, -0.0, 384.0, 416.0, 0x7F, -384.0, 0.125, -4.0]> : tensor<8xf8E4M3FN>
  %rhs = util.unfoldable_constant dense<[0.0, 0.0, 384.0, 1.0, 1.0, -384.0, 0.125, -4.0]> : tensor<8xf8E4M3FN>
  %init = tensor.empty() : tensor<8xf8E4M3FN>
  %add = linalg.add
    ins(%lhs, %rhs : tensor<8xf8E4M3FN>, tensor<8xf8E4M3FN>)
    outs(%init : tensor<8xf8E4M3FN>) -> tensor<8xf8E4M3FN>
  check.expect_almost_eq_const(%add, dense<[0.0, 0.0, 0x7F, 416.0, 0x7F, 0x7F, 0.25, -8.0]> : tensor<8xf8E4M3FN>) : tensor<8xf8E4M3FN>
  return
}

//===----------------------------------------------------------------------===//
// Negative zero preservation tests for IEEE fp8 types.
// IEEE types (f8E5M2, f8E4M3FN) support negative zero and should preserve it.
// We use bitcast to i8 and check.expect_eq_const to verify bit patterns,
// since check.expect_almost_eq_const treats -0.0 == +0.0.
//
// Bit patterns:
//   f8E5M2:   +0.0 = 0x00, -0.0 = 0x80
//   f8E4M3FN: +0.0 = 0x00, -0.0 = 0x80
//===----------------------------------------------------------------------===//

// Test: negf(+0.0) should produce -0.0 for f8E5M2.
// Expected bit pattern: 0x80 (-128 as signed i8).
func.func @negzero_negf_f8E5M2() {
  %input = util.unfoldable_constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf8E5M2>
  %init = tensor.empty() : tensor<4xf8E5M2>
  %neg = linalg.negf ins(%input : tensor<4xf8E5M2>) outs(%init : tensor<4xf8E5M2>) -> tensor<4xf8E5M2>
  %result_i8 = arith.bitcast %neg : tensor<4xf8E5M2> to tensor<4xi8>
  // -128 = 0x80 = -0.0 in f8E5M2
  check.expect_eq_const(%result_i8, dense<[-128, -128, -128, -128]> : tensor<4xi8>) : tensor<4xi8>
  return
}

// Test: negf(+0.0) should produce -0.0 for f8E4M3FN.
// Expected bit pattern: 0x80 (-128 as signed i8).
func.func @negzero_negf_f8E4M3FN() {
  %input = util.unfoldable_constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf8E4M3FN>
  %init = tensor.empty() : tensor<4xf8E4M3FN>
  %neg = linalg.negf ins(%input : tensor<4xf8E4M3FN>) outs(%init : tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN>
  %result_i8 = arith.bitcast %neg : tensor<4xf8E4M3FN> to tensor<4xi8>
  // -128 = 0x80 = -0.0 in f8E4M3FN
  check.expect_eq_const(%result_i8, dense<[-128, -128, -128, -128]> : tensor<4xi8>) : tensor<4xi8>
  return
}

// Test: -0.0 * 1.0 should preserve -0.0 for f8E5M2.
// IEEE semantics: -0.0 * positive = -0.0
func.func @negzero_mul_f8E5M2() {
  %input = util.unfoldable_constant dense<[-0.0, -0.0, -0.0, -0.0]> : tensor<4xf8E5M2>
  %one = util.unfoldable_constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf8E5M2>
  %init = tensor.empty() : tensor<4xf8E5M2>
  %mul = linalg.mul ins(%input, %one : tensor<4xf8E5M2>, tensor<4xf8E5M2>) outs(%init : tensor<4xf8E5M2>) -> tensor<4xf8E5M2>
  %result_i8 = arith.bitcast %mul : tensor<4xf8E5M2> to tensor<4xi8>
  // -128 = 0x80 = -0.0 in f8E5M2
  check.expect_eq_const(%result_i8, dense<[-128, -128, -128, -128]> : tensor<4xi8>) : tensor<4xi8>
  return
}

// Test: -0.0 * 1.0 should preserve -0.0 for f8E4M3FN.
func.func @negzero_mul_f8E4M3FN() {
  %input = util.unfoldable_constant dense<[-0.0, -0.0, -0.0, -0.0]> : tensor<4xf8E4M3FN>
  %one = util.unfoldable_constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf8E4M3FN>
  %init = tensor.empty() : tensor<4xf8E4M3FN>
  %mul = linalg.mul ins(%input, %one : tensor<4xf8E4M3FN>, tensor<4xf8E4M3FN>) outs(%init : tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN>
  %result_i8 = arith.bitcast %mul : tensor<4xf8E4M3FN> to tensor<4xi8>
  // -128 = 0x80 = -0.0 in f8E4M3FN
  check.expect_eq_const(%result_i8, dense<[-128, -128, -128, -128]> : tensor<4xi8>) : tensor<4xi8>
  return
}

//===----------------------------------------------------------------------===//
// Round-to-nearest-even tests.
// These tests verify IEEE 754 round-to-nearest-even behavior at tie points.
// At a tie (exactly halfway between two representable values), we round to
// the value with an even mantissa (LSB = 0).
//
// For f8E5M2FNUZ with exp=17 (values 2.0, 2.5, 3.0, 3.5):
//   2.0 = 0x44 (mantissa=00, even)
//   2.5 = 0x45 (mantissa=01, odd)
//   3.0 = 0x46 (mantissa=10, even)
//   3.5 = 0x47 (mantissa=11, odd)
//
// Tie points:
//   2.25 (halfway between 2.0 and 2.5) → 2.0 (even)
//   3.25 (halfway between 3.0 and 3.5) → 3.0 (even)
//
// Without round-to-nearest-even (e.g., round-half-away-from-zero):
//   2.25 → 2.5 (wrong!)
//   3.25 → 3.5 (wrong!)
//===----------------------------------------------------------------------===//

// Test: Round-to-nearest-even at tie points for f8E5M2FNUZ.
// Input f32 values 2.25 and 3.25 are exact tie points.
func.func @round_to_nearest_even_f8E5M2FNUZ() {
  // Use unfoldable f32 constant to test the runtime rounding behavior
  %tie_points = util.unfoldable_constant dense<[2.25, 3.25]> : tensor<2xf32>
  %truncated = arith.truncf %tie_points : tensor<2xf32> to tensor<2xf8E5M2FNUZ>
  %result_i8 = arith.bitcast %truncated : tensor<2xf8E5M2FNUZ> to tensor<2xi8>
  // 2.25 → 2.0 = 0x44 = 68
  // 3.25 → 3.0 = 0x46 = 70
  check.expect_eq_const(%result_i8, dense<[68, 70]> : tensor<2xi8>) : tensor<2xi8>
  return
}

//===----------------------------------------------------------------------===//
// Denormal (subnormal) handling tests.
// These tests verify proper handling of denormal values in fp8 formats.
//
// For f8E5M2FNUZ (bias=16, 2 mantissa bits):
//   Denormal values have exp=0, and value = mantissa * 2^(1-16-2) = mantissa * 2^-17
//   Denormal encodings: 0x01 (mantissa=1), 0x02 (mantissa=2), 0x03 (mantissa=3)
//   Values: ~7.63e-6, ~1.53e-5, ~2.29e-5
//   Min normal: 2^-15 = ~3.05e-5
//
// Without proper denormal handling:
//   ExtF: fp8 denormal → 0 (wrong, should be small nonzero f32)
//   TruncF: small f32 → 0 (wrong, should be fp8 denormal)
//===----------------------------------------------------------------------===//

// Test: ExtF properly converts fp8 denormals to f32.
// fp8 denormal 0x01 (mantissa=1) should become f32 7.629e-6, not 0.
func.func @extf_denormal_f8E5M2FNUZ() {
  // 0x01 = denormal with mantissa=1 → 1 * 2^-17 = 7.629394531e-6
  // 0x03 = denormal with mantissa=3 → 3 * 2^-17 = 2.288818359e-5
  %denormals = util.unfoldable_constant dense<[0x01, 0x03]> : tensor<2xf8E5M2FNUZ>
  %extended = arith.extf %denormals : tensor<2xf8E5M2FNUZ> to tensor<2xf32>
  // Check that values are nonzero and approximately correct.
  check.expect_almost_eq_const(%extended, dense<[7.629394531e-6, 2.288818359e-5]> : tensor<2xf32>) : tensor<2xf32>
  return
}

// Test: TruncF properly generates fp8 denormals from small f32 values.
// Small f32 values below min normal should become fp8 denormals, not 0.
// which would use APFloat (correct) instead of our emulation (being tested).
func.func @truncf_denormal_f8E5M2FNUZ() {
  // Values that should become denormals:
  // 7.629e-6 → 0x01 (mantissa=1)
  // 2.289e-5 → 0x03 (mantissa=3)
  %small_f32 = util.unfoldable_constant dense<[7.629394531e-6, 2.288818359e-5]> : tensor<2xf32>
  %truncated = arith.truncf %small_f32 : tensor<2xf32> to tensor<2xf8E5M2FNUZ>
  %result_i8 = arith.bitcast %truncated : tensor<2xf8E5M2FNUZ> to tensor<2xi8>
  // Should be denormal encodings 0x01 and 0x03, not 0.
  check.expect_eq_const(%result_i8, dense<[1, 3]> : tensor<2xi8>) : tensor<2xi8>
  return
}
