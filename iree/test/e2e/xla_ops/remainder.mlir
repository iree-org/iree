func @scalar() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<16.0> : tensor<f32>
  %input2 = iree.unfoldable_constant dense<7.0> : tensor<f32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<2.0> : tensor<f32>) : tensor<f32>
  return
}

func @tensor() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<[16.0, 17.0, 18.0]> : tensor<3xf32>
  %input2 = iree.unfoldable_constant dense<[7.0, 8.0, 9.0]> : tensor<3xf32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%result, dense<[2.0, 1.0, 0.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

func @negative_den() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<16.0> : tensor<f32>
  %input2 = iree.unfoldable_constant dense<-7.0> : tensor<f32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<2.0> : tensor<f32>) : tensor<f32>
  return
}

func @negative_num() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<-16.0> : tensor<f32>
  %input2 = iree.unfoldable_constant dense<7.0> : tensor<f32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-2.0> : tensor<f32>) : tensor<f32>
  return
}

func @scalar_int() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<16> : tensor<i32>
  %input2 = iree.unfoldable_constant dense<7> : tensor<i32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<2> : tensor<i32>) : tensor<i32>
  return
}

func @tensor_int() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<[16, 17, 18]> : tensor<3xi32>
  %input2 = iree.unfoldable_constant dense<[7, 8, 9]> : tensor<3xi32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  check.expect_eq_const(%result, dense<[2, 1, 0]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @negative_den_int() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<16> : tensor<i32>
  %input2 = iree.unfoldable_constant dense<-7> : tensor<i32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<2> : tensor<i32>) : tensor<i32>
  return
}

func @negative_num_int() attributes { iree.module.export } {
  %input1 = iree.unfoldable_constant dense<-16> : tensor<i32>
  %input2 = iree.unfoldable_constant dense<7> : tensor<i32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<-2> : tensor<i32>) : tensor<i32>
  return
}
