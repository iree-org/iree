// Complex overlap checks each result component.

func.func @scatter_complex_overlap() {
  %zeros = util.unfoldable_constant dense<0.0> : tensor<5xf32>
  %input = stablehlo.complex %zeros, %zeros : tensor<5xcomplex<f32>>
  %indices = util.unfoldable_constant dense<[[-1], [3]]> : tensor<2x1xi64>
  %real_updates = util.unfoldable_constant dense<[[1.0, 3.0], [5.0, 7.0]]> : tensor<2x2xf32>
  %imag_updates = util.unfoldable_constant dense<[[2.0, 4.0], [6.0, 8.0]]> : tensor<2x2xf32>
  %updates = stablehlo.complex %real_updates, %imag_updates : tensor<2x2xcomplex<f32>>
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<complex<f32>>, %update: tensor<complex<f32>>):
    %sum = stablehlo.add %current, %update : tensor<complex<f32>>
    stablehlo.return %sum : tensor<complex<f32>>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<5xcomplex<f32>>, tensor<2x1xi64>, tensor<2x2xcomplex<f32>>) -> tensor<5xcomplex<f32>>
  %real = stablehlo.real %result : (tensor<5xcomplex<f32>>) -> tensor<5xf32>
  %imag = stablehlo.imag %result : (tensor<5xcomplex<f32>>) -> tensor<5xf32>
  check.expect_eq_const(%real, dense<[3.0, 0.0, 0.0, 5.0, 7.0]> : tensor<5xf32>) : tensor<5xf32>
  check.expect_eq_const(%imag, dense<[4.0, 0.0, 0.0, 6.0, 8.0]> : tensor<5xf32>) : tensor<5xf32>
  return
}
