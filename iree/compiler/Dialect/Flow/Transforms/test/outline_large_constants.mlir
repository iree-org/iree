// RUN: iree-opt -split-input-file -iree-flow-outline-large-constants='min-storage-size=9' %s | IreeFileCheck %s

// CHECK: util.global private @[[LARGE_VARIABLE:.+]] {noinline} = dense<{{.+}}> : tensor<8xf32>
func @fn1() -> (tensor<2xf32>, tensor<512x128xf32>, tensor<8xf32>) {
  // CHECK-DAG: %[[SMALL_VALUE:.+]] = arith.constant dense<{{.+}}> : tensor<2xf32>
  %cst_0 = arith.constant dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
  // CHECK-DAG: %[[SPLATG_VALUE:.+]] = arith.constant dense<{{.+}}> : tensor<512x128xf32>
  %cst_1 = arith.constant dense<1.2> : tensor<512x128xf32>
  // CHECK-DAG: %[[LARGE_VALUE:.+]] = util.global.load @[[LARGE_VARIABLE]] : tensor<8xf32>
  %cst_2 = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]> : tensor<8xf32>
  return %cst_0, %cst_1, %cst_2 : tensor<2xf32>, tensor<512x128xf32>, tensor<8xf32>
}
