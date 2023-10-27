// TODO: The following generates tosa.ApplyScale ops that leaks to backends.
// Sizes like tensor<4xi32> will trigger vectorization on the SPIR-V backend.
// But we cannot vectorize tosa.ApplyScale ops.

func.func @tensor_int_shifted() {
  %0 = util.unfoldable_constant dense<[1, 0, 3, 4, 4]> : tensor<5xi32>
  %1 = util.unfoldable_constant dense<[5, 6, -3, 8, 8]> : tensor<5xi32>
  %result = tosa.mul %0, %1 {shift = 1 : i8} : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
  check.expect_eq_const(%result, dense<[3, 0, -4, 16, 16]> : tensor<5xi32>) : tensor<5xi32>
  return
}
