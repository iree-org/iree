// RUN: iree-opt -split-input-file -iree-flow-outline-large-constants %s | IreeFileCheck %s

// CHECK: flow.variable @[[LARGE_VARIABLE:.+]] dense<1.200000e+00> : tensor<512x128xf32>
func @fn1() -> (tensor<2xf32>, tensor<512x128xf32>) {
  // CHECK-DAG: %[[SMALL_VALUE:.+]] = constant dense<{{.+}}> : tensor<2xf32>
  %cst_0 = constant dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
  // CHECK-DAG: %[[LARGE_VALUE:.+]] = flow.variable.load @[[LARGE_VARIABLE]] : tensor<512x128xf32>
  %cst_1 = constant dense<1.2> : tensor<512x128xf32>
  return %cst_0, %cst_1 : tensor<2xf32>, tensor<512x128xf32>
}
