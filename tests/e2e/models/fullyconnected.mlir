// RUN: iree-run-mlir --Xcompiler,iree-input-type=mhlo_legacy --Xcompiler,iree-hal-target-backends=llvm-cpu %s --input=1x5xf32=1,-2,-3,4,-5 --input=1x5x3x1xf32=15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-input-type=mhlo_legacy --Xcompiler,iree-hal-target-backends=vulkan-spirv %s --input=1x5xf32=1,-2,-3,4,-5 --input=1x5x3x1xf32=15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 | FileCheck %s)

// CHECK-LABEL: EXEC @main
func.func @main(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5x3x1xf32>) -> tensor<5x1x5xf32> {
  %0 = "mhlo.reshape"(%arg0) {name = "reshape.3"} : (tensor<1x5xf32>) -> tensor<1x5xf32>
  %1 = "mhlo.transpose"(%0) {name = "transpose.41", permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x5xf32>) -> tensor<5x1xf32>
  %2 = "mhlo.reshape"(%1) {name = "reshape.42"} : (tensor<5x1xf32>) -> tensor<5x1x1xf32>
  %3 = "mhlo.reshape"(%2) {name = "reshape.55"} : (tensor<5x1x1xf32>) -> tensor<5x1xf32>
  %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, name = "broadcast.56"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %cst = arith.constant  {name = "constant.22"} dense<1.000000e+00> : tensor<f32>
  %5 = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.23"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %6 = mhlo.multiply %4, %5 {name = "multiply.57"} : tensor<5x1x5xf32>
  %cst_0 = arith.constant  {name = "constant.58"} dense<0.000000e+00> : tensor<f32>
  %7 = "mhlo.broadcast_in_dim"(%cst_0) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.59"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %8 = "mhlo.compare"(%6, %7) {comparison_direction = #mhlo<comparison_direction GT>, name = "compare.60"} : (tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xi1>
  %cst_1 = arith.constant  {name = "constant.24"} dense<0.000000e+00> : tensor<f32>
  %9 = "mhlo.broadcast_in_dim"(%cst_1) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.25"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %cst_2 = arith.constant  {name = "constant.90"} dense<0.000000e+00> : tensor<f32>
  %10 = "mhlo.broadcast_in_dim"(%cst_2) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.91"} : (tensor<f32>) -> tensor<5x5xf32>
  %11 = "mhlo.reshape"(%2) {name = "reshape.49"} : (tensor<5x1x1xf32>) -> tensor<5x1xf32>
  %12 = "mhlo.broadcast_in_dim"(%11) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, name = "broadcast.50"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %cst_3 = arith.constant  {name = "constant.15"} dense<1.000000e+00> : tensor<f32>
  %13 = "mhlo.broadcast_in_dim"(%cst_3) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.16"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %14 = mhlo.multiply %12, %13 {name = "multiply.51"} : tensor<5x1x5xf32>
  %cst_4 = arith.constant  {name = "constant.52"} dense<0.000000e+00> : tensor<f32>
  %15 = "mhlo.broadcast_in_dim"(%cst_4) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.53"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %16 = "mhlo.compare"(%14, %15) {comparison_direction = #mhlo<comparison_direction GT>, name = "compare.54"} : (tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xi1>
  %cst_5 = arith.constant  {name = "constant.17"} dense<0.000000e+00> : tensor<f32>
  %17 = "mhlo.broadcast_in_dim"(%cst_5) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.18"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %cst_6 = arith.constant  {name = "constant.78"} dense<0.000000e+00> : tensor<f32>
  %18 = "mhlo.broadcast_in_dim"(%cst_6) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.79"} : (tensor<f32>) -> tensor<5x5xf32>
  %19 = "mhlo.reshape"(%2) {name = "reshape.43"} : (tensor<5x1x1xf32>) -> tensor<5x1xf32>
  %20 = "mhlo.broadcast_in_dim"(%19) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, name = "broadcast.44"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %cst_7 = arith.constant  {name = "constant.8"} dense<1.000000e+00> : tensor<f32>
  %21 = "mhlo.broadcast_in_dim"(%cst_7) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.9"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %22 = mhlo.multiply %20, %21 {name = "multiply.45"} : tensor<5x1x5xf32>
  %cst_8 = arith.constant  {name = "constant.46"} dense<0.000000e+00> : tensor<f32>
  %23 = "mhlo.broadcast_in_dim"(%cst_8) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.47"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %24 = "mhlo.compare"(%22, %23) {comparison_direction = #mhlo<comparison_direction GT>, name = "compare.48"} : (tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xi1>
  %cst_9 = arith.constant  {name = "constant.10"} dense<0.000000e+00> : tensor<f32>
  %25 = "mhlo.broadcast_in_dim"(%cst_9) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.11"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %cst_10 = arith.constant  {name = "constant.66"} dense<0.000000e+00> : tensor<f32>
  %26 = "mhlo.broadcast_in_dim"(%cst_10) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.67"} : (tensor<f32>) -> tensor<5x5xf32>
  %27 = "mhlo.copy"(%arg1) {name = "copy.3"} : (tensor<1x5x3x1xf32>) -> tensor<1x5x3x1xf32>
  %28 = "mhlo.reshape"(%27) {name = "reshape.4"} : (tensor<1x5x3x1xf32>) -> tensor<1x5x3x1xf32>
  %29 = "mhlo.reshape"(%28) {name = "reshape.38"} : (tensor<1x5x3x1xf32>) -> tensor<1x5x3xf32>
  %30 = "mhlo.transpose"(%29) {name = "transpose.39", permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<1x5x3xf32>) -> tensor<5x1x3xf32>
  %31 = "mhlo.reshape"(%30) {name = "reshape.40"} : (tensor<5x1x3xf32>) -> tensor<5x3xf32>
  %cst_11 = arith.constant  {name = "constant.61"} dense<[[0.706495285, -0.567672312, 0.483717591, 0.522725761, 0.7563259], [-0.0899272263, -0.283501834, -0.350822538, -0.351515919, -0.337136656], [-0.451804549, 0.372324884, -0.620518147, 0.235451385, 0.851095855]]> : tensor<3x5xf32>
  %32 = "mhlo.dot"(%31, %cst_11) {name = "dot.62", precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  %cst_12 = arith.constant  {name = "constant.63"} dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<5xf32>
  %33 = "mhlo.broadcast_in_dim"(%cst_12) {broadcast_dimensions = dense<[1]> : tensor<1xi64>, name = "broadcast.64"} : (tensor<5xf32>) -> tensor<5x5xf32>
  %34 = mhlo.add %32, %33 {name = "add.65"} : tensor<5x5xf32>
  %35 = mhlo.maximum %26, %34 {name = "maximum.68"} : tensor<5x5xf32>
  %36 = "mhlo.reshape"(%35) {name = "reshape.70"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %37 = "mhlo.select"(%24, %25, %36) {name = "select.71"} : (tensor<5x1x5xi1>, tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  %38 = "mhlo.copy"(%37) {name = "copy.4"} : (tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  %39 = "mhlo.reshape"(%38) {name = "reshape.72"} : (tensor<5x1x5xf32>) -> tensor<5x5xf32>
  %cst_13 = arith.constant  {name = "constant.73"} dense<[[-0.0118641369, -3.785000e-02, 0.489048243, 0.321015775, -0.702280283], [-0.280262798, -0.724645615, -0.00332254497, 0.392334729, 0.619746447], [-0.113318317, -0.180415511, -0.146743968, 0.250408649, -0.442881733], [0.115600757, 0.703136146, -0.00812680274, -0.225454301, -0.0835619792], [-0.136745885, -6.298570e-01, 0.43629986, -0.689790308, 0.230725273]]> : tensor<5x5xf32>
  %40 = "mhlo.dot"(%39, %cst_13) {name = "dot.74", precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %cst_14 = arith.constant  {name = "constant.75"} dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<5xf32>
  %41 = "mhlo.broadcast_in_dim"(%cst_14) {broadcast_dimensions = dense<[1]> : tensor<1xi64>, name = "broadcast.76"} : (tensor<5xf32>) -> tensor<5x5xf32>
  %42 = mhlo.add %40, %41 {name = "add.77"} : tensor<5x5xf32>
  %43 = mhlo.maximum %18, %42 {name = "maximum.80"} : tensor<5x5xf32>
  %44 = "mhlo.reshape"(%43) {name = "reshape.82"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %45 = "mhlo.select"(%16, %17, %44) {name = "select.83"} : (tensor<5x1x5xi1>, tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  %46 = "mhlo.copy"(%45) {name = "copy.5"} : (tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  %47 = "mhlo.reshape"(%46) {name = "reshape.84"} : (tensor<5x1x5xf32>) -> tensor<5x5xf32>
  %cst_15 = arith.constant  {name = "constant.85"} dense<[[-0.136191264, -0.0401721969, 0.38497138, -5.850760e-01, 0.370910525], [-0.391011149, 0.0266356133, 0.309115469, -0.205079094, -0.559861302], [0.497760415, 0.689488232, 0.0759292394, -0.33134672, -0.237128958], [-0.53243047, 0.476418108, -0.371978909, 0.283265263, 0.63842845], [0.101761498, -0.218626946, 0.475128263, 0.042601984, 0.0988005772]]> : tensor<5x5xf32>
  %48 = "mhlo.dot"(%47, %cst_15) {name = "dot.86", precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %cst_16 = arith.constant  {name = "constant.87"} dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<5xf32>
  %49 = "mhlo.broadcast_in_dim"(%cst_16) {broadcast_dimensions = dense<[1]> : tensor<1xi64>, name = "broadcast.88"} : (tensor<5xf32>) -> tensor<5x5xf32>
  %50 = mhlo.add %48, %49 {name = "add.89"} : tensor<5x5xf32>
  %51 = mhlo.maximum %10, %50 {name = "maximum.92"} : tensor<5x5xf32>
  %52 = "mhlo.reshape"(%51) {name = "reshape.94"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %53 = "mhlo.select"(%8, %9, %52) {name = "select.95"} : (tensor<5x1x5xi1>, tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  %54 = "mhlo.reshape"(%53) {name = "reshape.96"} : (tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  return %54 : tensor<5x1x5xf32>
}

// On separate lines to avoid "[[" which FileCheck interprets as substitutions
// CHECK: 5x1x5xf32=[
// CHECK-SAME:   [0 0 0 0 0]
// CHECK-SAME: ][
// CHECK-SAME:   [3.79{{[0-9]+}} 4.99{{[0-9]+}} 0.90{{[0-9]+}} 0 0]
// CHECK-SAME: ][
// CHECK-SAME:   [2.80{{[0-9]+}} 3.78{{[0-9]+}} 0.56{{[0-9]+}} 0 0]
// CHECK-SAME: ][
// CHECK-SAME:   [0 0 0 0 0]
// CHECK-SAME: ][
// CHECK-SAME:   [0.87{{[0-9]+}} 1.21{{[0-9]+}} 0.13{{[0-9]+}} 0 0]
// CHECK-SAME: ]
