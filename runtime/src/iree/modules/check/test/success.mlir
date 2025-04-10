// RUN: iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx --iree-input-demote-f64-to-f32=false %s | iree-check-module --device=local-task --module=-

func.func @expect_true() {
  %true = util.unfoldable_constant 1 : i32
  check.expect_true(%true) : i32
  return
}

func.func @expect_false() {
  %false = util.unfoldable_constant 0 : i32
  check.expect_false(%false) : i32
  return
}

func.func @expect_all_true() {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %all_true = util.unfoldable_constant dense<1> : tensor<2x2xi32>
  %all_true_view = hal.tensor.export %all_true : tensor<2x2xi32> -> !hal.buffer_view
  check.expect_all_true<%device>(%all_true_view) : !hal.buffer_view
  return
}

func.func @expect_all_true_tensor() {
  %all_true = util.unfoldable_constant dense<1> : tensor<2x2xi32>
  check.expect_all_true(%all_true) : tensor<2x2xi32>
  return
}

func.func @expect_eq() {
  %const0 = util.unfoldable_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %const1 = util.unfoldable_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  check.expect_eq(%const0, %const1) : tensor<5xi32>
  return
}

func.func @expect_eq_const() {
  %const0 = util.unfoldable_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  check.expect_eq_const(%const0, dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func.func @expect_almost_eq() {
  %const0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  %const1 = util.unfoldable_constant dense<[0.999999, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  check.expect_almost_eq(%const0, %const1) : tensor<5xf32>
  return
}

func.func @expect_almost_eq_const_f32() {
  %const0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  check.expect_almost_eq_const(%const0, dense<[0.999999, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>) : tensor<5xf32>
  return
}

func.func @expect_almost_eq_const_f64() {
  %const0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>
  check.expect_almost_eq_const(%const0, dense<[0.999, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>, atol 1.0e-2) : tensor<5xf64>
  return
}

func.func @expect_almost_eq_const_f16() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xf16>
  check.expect_almost_eq_const(%const0, dense<[0.009, 99.0]> : tensor<2xf16>, atol 1.5e-2, rtol 1.5e-2) : tensor<2xf16>
  return
}

func.func @expect_almost_eq_const_bf16() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xbf16>
  check.expect_almost_eq_const(%const0, dense<[0.009, 99.0]> : tensor<2xbf16>, atol 1.5e-2, rtol 1.5e-2) : tensor<2xbf16>
  return
}

func.func @expect_almost_eq_const_f8E5M2() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xf8E5M2>
  check.expect_almost_eq_const(%const0, dense<[0.2, 80.0]> : tensor<2xf8E5M2>, atol 0.3, rtol 0.3) : tensor<2xf8E5M2>
  return
}

func.func @expect_almost_eq_const_f8E5M2FNUZ() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xf8E5M2FNUZ>
  check.expect_almost_eq_const(%const0, dense<[0.2, 80.0]> : tensor<2xf8E5M2FNUZ>, atol 0.3, rtol 0.3) : tensor<2xf8E5M2FNUZ>
  return
}

func.func @expect_almost_eq_const_f8E4M3FN() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xf8E4M3FN>
  check.expect_almost_eq_const(%const0, dense<[0.2, 80.0]> : tensor<2xf8E4M3FN>, atol 0.3, rtol 0.3) : tensor<2xf8E4M3FN>
  return
}

func.func @expect_almost_eq_const_f8E4M3FNUZ() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xf8E4M3FNUZ>
  check.expect_almost_eq_const(%const0, dense<[0.2, 80.0]> : tensor<2xf8E4M3FNUZ>, atol 0.3, rtol 0.3) : tensor<2xf8E4M3FNUZ>
  return
}

func.func @add() {
  %c5 = util.unfoldable_constant dense<5> : tensor<i32>
  %result = arith.addi %c5, %c5 : tensor<i32>
  %c10 = util.unfoldable_constant dense<10> : tensor<i32>
  check.expect_eq(%result, %c10) : tensor<i32>
  return
}

func.func @floats() {
  %cp1 = util.unfoldable_constant dense<0.1> : tensor<f32>
  %c1 = util.unfoldable_constant dense<1.0> : tensor<f32>
  %p2 = arith.addf %cp1, %cp1 : tensor<f32>
  %p3 = arith.addf %p2, %cp1 : tensor<f32>
  %p4 = arith.addf %p3, %cp1 : tensor<f32>
  %p5 = arith.addf %p4, %cp1 : tensor<f32>
  %p6 = arith.addf %p5, %cp1 : tensor<f32>
  %p7 = arith.addf %p6, %cp1 : tensor<f32>
  %p8 = arith.addf %p7, %cp1 : tensor<f32>
  %p9 = arith.addf %p8, %cp1 : tensor<f32>
  %approximately_1 = arith.addf %p9, %cp1 : tensor<f32>
  check.expect_almost_eq(%approximately_1, %c1) : tensor<f32>
  return
}
