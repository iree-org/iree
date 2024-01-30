// RUN: iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=vmvx %s | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=llvm-cpu %s | FileCheck %s

// CHECK-LABEL: EXEC @entry
func.func @entry() -> tensor<5x5xf32> {
  %0 = util.unfoldable_constant dense<0.000000e+00> : tensor<f32>
  %1 = util.unfoldable_constant dense<[[1.000000e+00], [-2.000000e+00], [-3.000000e+00], [4.000000e+00], [-5.000000e+00]]> : tensor<5x1xf32>
  %2 = util.unfoldable_constant dense<1.000000e+00> : tensor<f32>
  %3 = util.unfoldable_constant dense<[[3.464990e+00, -7.643890e+00, -5.722490e+00, 5.98052978, 1.768920e+01], [2.970700e+00, -6.207340e+00, -4.259620e+00, 4.760550e+00, 1.387840e+01], [2.476410e+00, -4.770790e+00, -2.796750e+00, 3.540560e+00, 1.006750e+01], [1.982120e+00, -3.334240e+00, -1.333880e+00, 2.320580e+00, 6.256660e+00], [1.487830e+00, -1.897700e+00, 1.289900e-01, 1.100600e+00, 2.445800e+00]]> : tensor<5x5xf32>
  %4 = util.unfoldable_constant dense<0.000000e+00> : tensor<5xf32>
  %5 = call @_entry(%0, %1, %2, %3, %4) : (tensor<f32>, tensor<5x1xf32>, tensor<f32>, tensor<5x5xf32>, tensor<5xf32>) -> tensor<5x5xf32>
  return %5 : tensor<5x5xf32>
}
func.func private @_entry(%arg0: tensor<f32>, %arg1: tensor<5x1xf32>, %arg2: tensor<f32>, %arg3: tensor<5x5xf32>, %arg4: tensor<5xf32>) -> tensor<5x5xf32> {
  %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] {name = "broadcast.44"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %1 = stablehlo.broadcast_in_dim %arg2, dims = [] {name = "broadcast.9"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<5x1x5xf32>
  %3 = stablehlo.broadcast_in_dim %arg0, dims = [] {name = "broadcast.47"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %4 = stablehlo.compare  GT, %2, %3 : (tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xi1>
  %5 = stablehlo.broadcast_in_dim %arg0, dims = [] {name = "broadcast.11"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %6 = stablehlo.broadcast_in_dim %arg0, dims = [] {name = "broadcast.67"} : (tensor<f32>) -> tensor<5x5xf32>
  %7 = stablehlo.broadcast_in_dim %arg4, dims = [1] {name = "broadcast.64"} : (tensor<5xf32>) -> tensor<5x5xf32>
  %8 = stablehlo.add %arg3, %7 : tensor<5x5xf32>
  %9 = stablehlo.maximum %6, %8 {name = "maximum.68"} : tensor<5x5xf32>
  %10 = stablehlo.reshape %9 {name = "reshape.70"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %11 = stablehlo.select %4, %5, %10 {name = "select.71"} : tensor<5x1x5xi1>, tensor<5x1x5xf32>
  %12 = stablehlo.reshape %11 {name = "reshape.72"} : (tensor<5x1x5xf32>) -> tensor<5x5xf32>
  return %12 : tensor<5x5xf32>
}

// On separate lines to avoid "[[" which FileCheck interprets as substitutions
// CHECK: 5x5xf32=
// CHECK-SAME: [0 0 0 0 0]
// CHECK-SAME: [2.97{{[0-9]+}} 0 0 4.76{{[0-9]+}} 13.87{{[0-9]+}}]
// CHECK-SAME: [2.47{{[0-9]+}} 0 0 3.54{{[0-9]+}} 10.06{{[0-9]+}}]
// CHECK-SAME: [0 0 0 0 0]
// CHECK-SAME: [1.48{{[0-9]+}} 0 0.12{{[0-9]+}} 1.10{{[0-9]+}} 2.44{{[0-9]+}}]
