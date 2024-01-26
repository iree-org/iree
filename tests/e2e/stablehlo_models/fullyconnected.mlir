func.func @main() {
  %input_0 = util.unfoldable_constant dense<[[1.0,-2.0,-3.0,4.0,-5.0]]> : tensor<1x5xf32>
  %input_1 = util.unfoldable_constant dense<[[[[15.],[14.],[13.]],
                                              [[12.],[11.],[10.]],
                                              [[9.],[8.],[7.]],
                                              [[6.],[5.],[4.]],
                                              [[3.],[2.],[1.]]]]> : tensor<1x5x3x1xf32>

  %0 = stablehlo.transpose %input_0, dims = [1, 0] {name = "transpose.41"} : (tensor<1x5xf32>) -> tensor<5x1xf32>
  %1 = stablehlo.reshape %0 {name = "reshape.42"} : (tensor<5x1xf32>) -> tensor<5x1x1xf32>
  %2 = stablehlo.reshape %0 {name = "reshape.55"} : (tensor<5x1xf32>) -> tensor<5x1xf32>
  %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] {name = "broadcast.56"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %cst = arith.constant {name = "constant.22"} dense<1.000000e+00> : tensor<f32>
  %4 = stablehlo.constant dense<1.000000e+00> : tensor<5x1x5xf32>
  %5 = stablehlo.multiply %3, %4 {name = "multiply.57"} : tensor<5x1x5xf32>
  %cst_0 = arith.constant {name = "constant.58"} dense<0.000000e+00> : tensor<f32>
  %6 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x5xf32>
  %7 = stablehlo.compare  GT, %5, %6 {name = "compare.60"} : (tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xi1>
  %cst_1 = arith.constant {name = "constant.24"} dense<0.000000e+00> : tensor<f32>
  %8 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x5xf32>
  %cst_2 = arith.constant {name = "constant.90"} dense<0.000000e+00> : tensor<f32>
  %9 = stablehlo.constant dense<0.000000e+00> : tensor<5x5xf32>
  %10 = stablehlo.reshape %0 {name = "reshape.49"} : (tensor<5x1xf32>) -> tensor<5x1xf32>
  %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] {name = "broadcast.50"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %cst_3 = arith.constant {name = "constant.15"} dense<1.000000e+00> : tensor<f32>
  %12 = stablehlo.constant dense<1.000000e+00> : tensor<5x1x5xf32>
  %13 = stablehlo.multiply %11, %12 {name = "multiply.51"} : tensor<5x1x5xf32>
  %cst_4 = arith.constant {name = "constant.52"} dense<0.000000e+00> : tensor<f32>
  %14 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x5xf32>
  %15 = stablehlo.compare  GT, %13, %14 {name = "compare.54"} : (tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xi1>
  %cst_5 = arith.constant {name = "constant.17"} dense<0.000000e+00> : tensor<f32>
  %16 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x5xf32>
  %cst_6 = arith.constant {name = "constant.78"} dense<0.000000e+00> : tensor<f32>
  %17 = stablehlo.constant dense<0.000000e+00> : tensor<5x5xf32>
  %18 = stablehlo.reshape %0 {name = "reshape.43"} : (tensor<5x1xf32>) -> tensor<5x1xf32>
  %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] {name = "broadcast.44"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %cst_7 = arith.constant {name = "constant.8"} dense<1.000000e+00> : tensor<f32>
  %20 = stablehlo.constant dense<1.000000e+00> : tensor<5x1x5xf32>
  %21 = stablehlo.multiply %19, %20 {name = "multiply.45"} : tensor<5x1x5xf32>
  %cst_8 = arith.constant {name = "constant.46"} dense<0.000000e+00> : tensor<f32>
  %22 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x5xf32>
  %23 = stablehlo.compare  GT, %21, %22 {name = "compare.48"} : (tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xi1>
  %cst_9 = arith.constant {name = "constant.10"} dense<0.000000e+00> : tensor<f32>
  %24 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x5xf32>
  %cst_10 = arith.constant {name = "constant.66"} dense<0.000000e+00> : tensor<f32>
  %25 = stablehlo.constant dense<0.000000e+00> : tensor<5x5xf32>

  %26 = stablehlo.reshape %input_1 {name = "reshape.38"} : (tensor<1x5x3x1xf32>) -> tensor<1x5x3xf32>
  %27 = stablehlo.transpose %26, dims = [1, 0, 2] {name = "transpose.39"} : (tensor<1x5x3xf32>) -> tensor<5x1x3xf32>
  %28 = stablehlo.reshape %27 {name = "reshape.40"} : (tensor<5x1x3xf32>) -> tensor<5x3xf32>
  %cst_11 = arith.constant {name = "constant.61"} dense<[[0.706495285, -0.567672312, 0.483717591, 0.522725761, 0.7563259], [-0.0899272263, -0.283501834, -0.350822538, -0.351515919, -0.337136656], [-0.451804549, 0.372324884, -0.620518147, 0.235451385, 0.851095855]]> : tensor<3x5xf32>
  %29 = stablehlo.dot %28, %cst_11, precision = [DEFAULT, DEFAULT] {name = "dot.62"} : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  %cst_12 = arith.constant {name = "constant.63"} dense<0.000000e+00> : tensor<5xf32>
  %30 = stablehlo.constant dense<0.000000e+00> : tensor<5x5xf32>
  %31 = stablehlo.add %29, %30 {name = "add.65"} : tensor<5x5xf32>
  %32 = stablehlo.maximum %25, %31 {name = "maximum.68"} : tensor<5x5xf32>
  %33 = stablehlo.reshape %32 {name = "reshape.70"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %34 = stablehlo.select %23, %24, %33 {name = "select.71"} : tensor<5x1x5xi1>, tensor<5x1x5xf32>
  %35 = stablehlo.reshape %34 {name = "reshape.72"} : (tensor<5x1x5xf32>) -> tensor<5x5xf32>
  %cst_13 = arith.constant {name = "constant.73"} dense<[[-0.0118641369, -3.785000e-02, 0.489048243, 0.321015775, -0.702280283], [-0.280262798, -0.724645615, -0.00332254497, 0.392334729, 0.619746447], [-0.113318317, -0.180415511, -0.146743968, 0.250408649, -0.442881733], [0.115600757, 0.703136146, -0.00812680274, -0.225454301, -0.0835619792], [-0.136745885, -6.298570e-01, 0.43629986, -0.689790308, 0.230725273]]> : tensor<5x5xf32>
  %36 = stablehlo.dot %35, %cst_13, precision = [DEFAULT, DEFAULT] {name = "dot.74"} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %cst_14 = arith.constant {name = "constant.75"} dense<0.000000e+00> : tensor<5xf32>
  %37 = stablehlo.constant dense<0.000000e+00> : tensor<5x5xf32>
  %38 = stablehlo.add %36, %37 {name = "add.77"} : tensor<5x5xf32>
  %39 = stablehlo.maximum %17, %38 {name = "maximum.80"} : tensor<5x5xf32>
  %40 = stablehlo.reshape %39 {name = "reshape.82"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %41 = stablehlo.select %15, %16, %40 {name = "select.83"} : tensor<5x1x5xi1>, tensor<5x1x5xf32>
  %42 = stablehlo.reshape %41 {name = "reshape.84"} : (tensor<5x1x5xf32>) -> tensor<5x5xf32>
  %cst_15 = arith.constant {name = "constant.85"} dense<[[-0.136191264, -0.0401721969, 0.38497138, -5.850760e-01, 0.370910525], [-0.391011149, 0.0266356133, 0.309115469, -0.205079094, -0.559861302], [0.497760415, 0.689488232, 0.0759292394, -0.33134672, -0.237128958], [-0.53243047, 0.476418108, -0.371978909, 0.283265263, 0.63842845], [0.101761498, -0.218626946, 0.475128263, 0.042601984, 0.0988005772]]> : tensor<5x5xf32>
  %43 = stablehlo.dot %42, %cst_15, precision = [DEFAULT, DEFAULT] {name = "dot.86"} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %cst_16 = arith.constant {name = "constant.87"} dense<0.000000e+00> : tensor<5xf32>
  %44 = stablehlo.constant dense<0.000000e+00> : tensor<5x5xf32>
  %45 = stablehlo.add %43, %44 {name = "add.89"} : tensor<5x5xf32>
  %46 = stablehlo.maximum %9, %45 {name = "maximum.92"} : tensor<5x5xf32>
  %47 = stablehlo.reshape %46 {name = "reshape.94"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %result = stablehlo.select %7, %8, %47 {name = "select.95"} : tensor<5x1x5xi1>, tensor<5x1x5xf32>

  check.expect_almost_eq_const(%result, dense<[[[0., 0., 0., 0., 0.]],
                                               [[3.79097, 4.9929, 0.9083, 0., 0.]],
                                               [[2.8042, 3.7808, 0.5600, 0., 0.]],
                                               [[0., 0., 0., 0., 0.]],
                                               [[0.8795, 1.2182, 0.1342, 0., 0.]]]> :
      tensor<5x1x5xf32>) : tensor<5x1x5xf32>
  return
}
