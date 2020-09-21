// RUN: (iree-run-mlir -iree-hal-target-backends=llvm-ir -input-value="1x5xf32=[[1,1,1,1,1]]" %s) | IreeFileCheck %s

// CHECK: EXEC @main
// CHECK: 2x5xf32=[1 1 1 1 1][0 0 0 0 0]

func @main(%0: tensor<1x5xf32>) -> tensor<2x5xf32> attributes {iree.module.export} {
  %1 = mhlo.constant dense<0.000000e+00> : tensor<1x5xf32> 
  %2 = "mhlo.concatenate"(%0, %1) {dimension = 0 : i64} : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  return %2 : tensor<2x5xf32>
}

