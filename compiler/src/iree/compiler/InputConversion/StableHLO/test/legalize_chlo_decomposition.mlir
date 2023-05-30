// RUN: iree-opt --iree-stablehlo-legalize-chlo \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @asin_bf16(
func.func @asin_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  %result = "chlo.asin"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %result : tensor<bf16>
}

// -----

// CHECK-LABEL: func.func @asin_f16(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<f16>
func.func @asin_f16(%arg : tensor<f16>) -> tensor<f16> {
  %result = "chlo.asin"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %result : tensor<f16>
}

// -----

// CHECK-LABEL: func.func @asin_f32(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<f32>) -> tensor<f32>
func.func @asin_f32(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.asin"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL:  func.func @asin_f64(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<f64>) -> tensor<f64>
func.func @asin_f64(%arg : tensor<f64>) -> tensor<f64> {
  %result = "chlo.asin"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %result : tensor<f64>
}

// -----

// CHECK-LABEL:  func.func @asin_complex_f32(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<complex<f32>>) -> tensor<complex<f32>>
func.func @asin_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.asin"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL:  func.func @asin_complex_f64_dynamic(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>>
func.func @asin_complex_f64_dynamic(%arg : tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>> {
  %result = "chlo.asin"(%arg) : (tensor<?xcomplex<f64>>) -> tensor<?xcomplex<f64>>
  func.return %result : tensor<?xcomplex<f64>>
}

// -----

// CHECK-LABEL: @asinh_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @asinh_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  %result = "chlo.asinh"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %result : tensor<bf16>
}

// -----

// CHECK-LABEL: @asinh_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @asinh_f16(%arg : tensor<f16>) -> tensor<f16> {
  %result = "chlo.asinh"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %result : tensor<f16>
}

// -----

// CHECK-LABEL: @asinh_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @asinh_f32(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.asinh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL: @asinh_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @asinh_f64(%arg : tensor<f64>) -> tensor<f64> {
  %result = "chlo.asinh"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %result : tensor<f64>
}

// -----

// CHECK-LABEL: @asinh_complex_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<complex<f32>>
func.func @asinh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.asinh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// Lower statically shaped `constant_like` to constant.
// CHECK-LABEL: @constant_like_static_shape
func.func @constant_like_static_shape(%arg : tensor<1x2xi64>) -> tensor<1x2xf32> {
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<1x2xi64>) -> tensor<1x2xf32>
  func.return %result : tensor<1x2xf32>
}

// -----

// Lower dynamically shaped `constant_like` to broadcasted constant.
// CHECK-LABEL: constant_like_dynamic_shape
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x?xi64>)
func.func @constant_like_dynamic_shape(%arg : tensor<?x?xi64>) -> tensor<?x?xf32> {
  %result = "chlo.constant_like"(%arg) { value = 3.2 : f32 }
      : (tensor<?x?xi64>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @conj
func.func @conj(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
  // CHECK-SAME: ([[INPUT:%.*]]: tensor
  %1 = "chlo.conj"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  func.return %1 : tensor<3xcomplex<f32>>
}

// -----

// CHECK-LABEL: @erf_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @erf_f64(%arg : tensor<f64>) -> tensor<f64> {
  %1 = "chlo.erf"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @erf_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @erf_f32(%arg : tensor<f32>) -> tensor<f32> {
  %1 = "chlo.erf"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @erf_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @erf_f16(%arg : tensor<f16>) -> tensor<f16> {
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @erf_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @erf_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  %1 = "chlo.erf"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// -----

// CHECK-LABEL: @acosh
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @acosh(%arg: tensor<f16>) -> tensor<f16> {
  %1 = "chlo.acosh"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @acosh_complex_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<complex<f32>>
func.func @acosh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.acosh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @erfc_f64
// CHECK-SAME: %[[ARG:.*]]: tensor<f64>
func.func @erfc_f64(%arg : tensor<f64>) -> tensor<f64> {
  %1 = "chlo.erfc"(%arg) : (tensor<f64>) -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @erfc_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @erfc_f32(%arg : tensor<f32>) -> tensor<f32> {
  %1 = "chlo.erfc"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @erfc_f16
// CHECK-SAME: %[[ARG:.*]]: tensor<f16>
func.func @erfc_f16(%arg : tensor<f16>) -> tensor<f16> {
  %1 = "chlo.erfc"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @erfc_bf16
// CHECK-SAME: %[[ARG:.*]]: tensor<bf16>
func.func @erfc_bf16(%arg : tensor<bf16>) -> tensor<bf16> {
  %1 = "chlo.erfc"(%arg) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// -----

// CHECK-LABEL: @is_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  %1 = chlo.is_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @is_pos_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_pos_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  %1 = chlo.is_pos_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @is_neg_inf_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @is_neg_inf_f32(%arg : tensor<f32>) -> tensor<i1> {
  %1 = chlo.is_neg_inf %arg : tensor<f32> -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: @lgamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func.func @lgamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  %1 = chlo.lgamma %arg : tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @lgamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @lgamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  %1 = chlo.lgamma %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @lgamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func.func @lgamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  %1 = chlo.lgamma %arg : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @digamma_f64
// CHECK-SAME: (%[[ARG:.*]]: tensor<f64>)
func.func @digamma_f64(%arg : tensor<f64>) -> tensor<f64> {
  %1 = chlo.digamma %arg : tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @digamma_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @digamma_f32(%arg : tensor<f32>) -> tensor<f32> {
  %1 = chlo.digamma %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @digamma_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func.func @digamma_f16(%arg : tensor<f16>) -> tensor<f16> {
  %1 = chlo.digamma %arg : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @zeta_f16
// CHECK-SAME:  (%[[X:.*]]: tensor<f16>, %[[Q:.*]]: tensor<f16>) -> tensor<f16>
func.func @zeta_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  %0 = chlo.zeta %arg0, %arg1 : tensor<f16>, tensor<f16> -> tensor<f16>
  func.return %0 : tensor<f16>
}

// -----


// CHECK-LABEL: @polygamma_f32
func.func @polygamma_f32(%lhs : tensor<f32>, %rhs : tensor<f32>) -> tensor<f32> {
  %1 = chlo.polygamma %lhs, %rhs : tensor<f32>, tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----


// CHECK-LABEL: @polygamma_f64
func.func @polygamma_f64(%lhs : tensor<f64>, %rhs : tensor<f64>) -> tensor<f64> {
  %1 = chlo.polygamma %lhs, %rhs : tensor<f64>, tensor<f64> -> tensor<f64>
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: @polygamma_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>, %[[ARG1:.*]]: tensor<f16>)
func.func @polygamma_f16(%lhs : tensor<f16>, %rhs : tensor<f16>) -> tensor<f16> {
  %1 = chlo.polygamma %lhs, %rhs : tensor<f16>, tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @sinh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func.func @sinh_f32(%x : tensor<f32>) -> tensor<f32> {
  %1 = chlo.sinh %x : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @sinh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func.func @sinh_f16(%x : tensor<f16>) -> tensor<f16> {
  %1 = chlo.sinh %x : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @sinh_complex
// CHECK-SAME: (%[[X:.*]]: tensor<2xcomplex<f32>>)
func.func @sinh_complex(%x : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  %1 = chlo.sinh %x : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
  func.return %1 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @cosh_f32
// CHECK-SAME: (%[[X:.*]]: tensor<f32>)
func.func @cosh_f32(%x : tensor<f32>) -> tensor<f32> {
  %1 = chlo.cosh %x : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @cosh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f16>)
func.func @cosh_f16(%x : tensor<f16>) -> tensor<f16> {
  %1 = chlo.cosh %x : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @cosh_complex_f32
// CHECK-SAME: (%[[X:.*]]: tensor<complex<f32>>)
func.func @cosh_complex_f32(%x : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %1 = chlo.cosh %x : tensor<complex<f32>> -> tensor<complex<f32>>
  func.return %1 : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @atanh_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>
func.func @atanh_f32(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.atanh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL: @atanh_complex_f32
// CHECK-SAME: %[[ARG:.*]]: tensor<complex<f32>>
func.func @atanh_complex_f32(%arg : tensor<complex<f32>>) -> tensor<complex<f32>> {
  %result = "chlo.atanh"(%arg) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %result : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: @next_after_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: tensor<2xf32>)
func.func @next_after_f32(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
  %1 = chlo.broadcast_next_after %x, %y : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @tan_f16
// CHECK-SAME: (%[[ARG:.*]]: tensor<f16>)
func.func @tan_f16(%arg : tensor<f16>) -> tensor<f16> {
 %1 = chlo.tan %arg : tensor<f16> -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

// CHECK-LABEL: @tan_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @tan_f32(%arg : tensor<f32>) -> tensor<f32> {
  %1 = chlo.tan %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @top_k
// CHECK-SAME: (%[[ARG:.*]]: tensor<16x16xf32>)
func.func @top_k(%arg : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  func.return %1#0, %1#1 : tensor<16x8xf32>, tensor<16x8xi32>
}

// -----

// CHECK-LABEL: @dyn_top_k
// CHECK-SAME: ([[ARG:%.*]]: tensor<?x5x?xi1>
// CHECK-SAME: -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
func.func @dyn_top_k(%arg0: tensor<?x5x?xi1>) -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>) {
  %values, %indices = chlo.top_k(%arg0, k = 2) : tensor<?x5x?xi1> -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
  return %values, %indices : tensor<?x5x2xi1>, tensor<?x5x2xi32>
}

// -----

func.func @unranked_top_k(%arg : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>) {
  // expected-error@+1 {{failed to legalize operation 'chlo.top_k' that was explicitly marked illegal}}
  %1:2 = chlo.top_k(%arg, k=8) : tensor<*xf32> -> (tensor<*xf32>, tensor<*xi32>)
  func.return %1#0, %1#1 : tensor<*xf32>, tensor<*xi32>
}

// -----

// Verify bessel_i1e operator for f16, f32, f64 separately as they use
// different coefficients.

// CHECK-LABEL: @bessel_i1e_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf16>)
func.func @bessel_i1e_f16(%arg: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf16> -> tensor<16x16xf16>
  func.return %0 : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: @bessel_i1e_f32
// CHECK-SAME:   (%[[ARG0:.*]]: tensor<16x16xf32>)
func.func @bessel_i1e_f32(%arg : tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: @bessel_i1e_f64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf64>)
func.func @bessel_i1e_f64(%arg : tensor<16x16xf64>) -> tensor<16x16xf64> {
  %0 = chlo.bessel_i1e %arg : tensor<16x16xf64> -> tensor<16x16xf64>
  func.return %0 : tensor<16x16xf64>
}

// -----

// CHECK-LABEL: @erf_inv
func.func @erf_inv(%arg0 : tensor<16x16xf32>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
  return
}

// -----

// CHECK-LABEL: @erf_inv_wide
func.func @erf_inv_wide(%arg0 : tensor<16x16xf64>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf64> -> tensor<16x16xf64>
  return
}
