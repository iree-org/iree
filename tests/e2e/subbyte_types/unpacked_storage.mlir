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
