// RUN: iree-opt -split-input-file -iree-mhlo-to-linalg-on-tensors -canonicalize %s | IreeFileCheck %s

func @invalid_slice(%arg0: tensor<6xi32>) -> tensor<3xi32> {
  %0 = "mhlo.slice"(%arg0) {
    limit_indices = dense<5> : tensor<1xi64>,
    start_indices = dense<0> : tensor<1xi64>,
    strides = dense<2> : tensor<1xi64>
  } : (tensor<6xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}
// CHECK-LABEL: func @invalid_slice
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK:         %[[RES.+]] = tensor.extract_slice %[[ARG0]][0] [3] [2] : tensor<6xi32> to tensor<3xi32>
// CHECK:         return %[[RES]]
