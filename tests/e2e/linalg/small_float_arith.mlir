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
