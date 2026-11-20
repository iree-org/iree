// Addition wraps at the logical i4 width.
func.func @unpacked_i4_arithmetic() {
  %input = util.unfoldable_constant dense<[7, -8, -1, 0]> : tensor<4xi4, #iree_encoding.unpacked_storage>
  %empty = tensor.empty() : tensor<4xi4, #iree_encoding.unpacked_storage>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<4xi4, #iree_encoding.unpacked_storage>)
      outs(%empty : tensor<4xi4, #iree_encoding.unpacked_storage>) {
    ^bb0(%in: i4, %out: i4):
      %one = arith.constant 1 : i4
      %sum = arith.addi %in, %one : i4
      linalg.yield %sum : i4
  } -> tensor<4xi4, #iree_encoding.unpacked_storage>
  %stored = util.optimization_barrier %result : tensor<4xi4, #iree_encoding.unpacked_storage>
  %wide_empty = tensor.empty() : tensor<4xi8>
  %signed = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%stored : tensor<4xi4, #iree_encoding.unpacked_storage>)
      outs(%wide_empty : tensor<4xi8>) {
    ^bb0(%in: i4, %out: i8):
      %extended = arith.extsi %in : i4 to i8
      linalg.yield %extended : i8
  } -> tensor<4xi8>
  check.expect_eq_const(%signed, dense<[-8, -7, 0, 1]> : tensor<4xi8>) : tensor<4xi8>

  // Check that stores clear the unused high bits.
  %buffer = hal.tensor.export %stored : tensor<4xi4, #iree_encoding.unpacked_storage> -> !hal.buffer
  %bytes = hal.tensor.import %buffer : !hal.buffer -> tensor<4xi8>
  check.expect_eq_const(%bytes, dense<[8, 9, 0, 1]> : tensor<4xi8>) : tensor<4xi8>
  return
}

// Loads ignore unused high bits in imported storage.
func.func @unpacked_i5_import() {
  %bytes = util.unfoldable_constant dense<[255, 16, 15, 32]> : tensor<4xi8>
  %buffer = hal.tensor.export %bytes : tensor<4xi8> -> !hal.buffer
  %input = hal.tensor.import %buffer : !hal.buffer -> tensor<4xi5, #iree_encoding.unpacked_storage>
  %empty = tensor.empty() : tensor<4xi8>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<4xi5, #iree_encoding.unpacked_storage>)
      outs(%empty : tensor<4xi8>) {
    ^bb0(%in: i5, %out: i8):
      %extended = arith.extsi %in : i5 to i8
      linalg.yield %extended : i8
  } -> tensor<4xi8>
  check.expect_eq_const(%result, dense<[-1, -16, 15, 0]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func.func @unpacked_signed_dot() {
  %lhs = util.unfoldable_constant dense<[-1, -8, 7, 0]> : tensor<4xi4, #iree_encoding.unpacked_storage>
  %rhs = util.unfoldable_constant dense<[2, -1, 3, 7]> : tensor<4xi4, #iree_encoding.unpacked_storage>
  %init = arith.constant dense<0> : tensor<i32>
  %result = linalg.dot ins(%lhs, %rhs : tensor<4xi4, #iree_encoding.unpacked_storage>, tensor<4xi4, #iree_encoding.unpacked_storage>) outs(%init : tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<27> : tensor<i32>) : tensor<i32>
  return
}

// Exercise every FP4 bit pattern, including both zeros. Loads ignore the high
// nibble; negation flips the sign bit, and stores clear the high nibble.
func.func @unpacked_fp4_negate() {
  %bytes = util.unfoldable_constant dense<[-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]> : tensor<16xi8>
  %buffer = hal.tensor.export %bytes : tensor<16xi8> -> !hal.buffer
  %input = hal.tensor.import %buffer : !hal.buffer -> tensor<16xf4E2M1FN, #iree_encoding.unpacked_storage>
  %empty = tensor.empty() : tensor<16xf4E2M1FN, #iree_encoding.unpacked_storage>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<16xf4E2M1FN, #iree_encoding.unpacked_storage>)
      outs(%empty : tensor<16xf4E2M1FN, #iree_encoding.unpacked_storage>) {
    ^bb0(%in: f4E2M1FN, %out: f4E2M1FN):
      %neg = arith.negf %in : f4E2M1FN
      linalg.yield %neg : f4E2M1FN
  } -> tensor<16xf4E2M1FN, #iree_encoding.unpacked_storage>
  %stored = util.optimization_barrier %result : tensor<16xf4E2M1FN, #iree_encoding.unpacked_storage>
  %output_buffer = hal.tensor.export %stored : tensor<16xf4E2M1FN, #iree_encoding.unpacked_storage> -> !hal.buffer
  %output = hal.tensor.import %output_buffer : !hal.buffer -> tensor<16xi8>
  check.expect_eq_const(%output, dense<[8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]> : tensor<16xi8>) : tensor<16xi8>
  return
}

// Byte-aligned packed and unpacked storage share the ordinary float layout.
func.func @byte_aligned_float_storage() {
  %input = util.unfoldable_constant dense<[1.0, -2.0]> : tensor<2xf16, #iree_encoding.packed_storage>
  %empty = tensor.empty() : tensor<2xf16, #iree_encoding.unpacked_storage>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<2xf16, #iree_encoding.packed_storage>)
      outs(%empty : tensor<2xf16, #iree_encoding.unpacked_storage>) {
    ^bb0(%in: f16, %out: f16):
      %neg = arith.negf %in : f16
      linalg.yield %neg : f16
  } -> tensor<2xf16, #iree_encoding.unpacked_storage>
  %buffer = hal.tensor.export %result : tensor<2xf16, #iree_encoding.unpacked_storage> -> !hal.buffer
  %output = hal.tensor.import %buffer : !hal.buffer -> tensor<2xf16>
  check.expect_eq_const(%output, dense<[-1.0, 2.0]> : tensor<2xf16>) : tensor<2xf16>
  return
}
