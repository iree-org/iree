// RUN: iree-opt -split-input-file -iree-mhlo-to-linalg-on-tensors -canonicalize %s | IreeFileCheck %s

func @rfft_1d(%input: tensor<32xf32>) -> (tensor<17xf32>, tensor<17xf32>) {
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<1xi64>, fft_type = "RFFT"
  } : (tensor<32xf32>) -> tensor<17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  return %1, %2 : tensor<17xf32>, tensor<17xf32>
}
// CHECK:     func @rfft_1d
// CHECK-SAME:  %[[Arg0:[a-zA-Z0-9_]*]]
// CHECK-DAG:   %[[RealMatrix:.+]] = constant dense<"0x0000803F{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[ImagMatrix:.+]] = constant dense<"0x00000080{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[Zero:.+]] = constant 0.000000e+00 : f32
// CHECK:       %[[RealInit:.+]] = linalg.init_tensor [17] : tensor<17xf32>
// CHECK:       %[[RealFill:.+]] = linalg.fill(%[[RealInit]], %[[Zero]])
// CHECK:       %[[RealRes:.+]] = linalg.vecmat
// CHECK-SAME:    ins(%[[Arg0]], %[[RealMatrix]] : tensor<32xf32>, tensor<32x17xf32>)
// CHECK-SAME:    outs(%[[RealFill]] : tensor<17xf32>) -> tensor<17xf32>
// CHECK:        %[[ImagInit:.+]] = linalg.init_tensor [17] : tensor<17xf32>
// CHECK:        %[[ImagFill:.+]] = linalg.fill(%[[ImagInit]], %[[Zero]])
// CHECK:        %[[ImagRes:.+]] = linalg.vecmat
// CHECK-SAME:     ins(%[[Arg0]], %[[ImagMatrix]] : tensor<32xf32>, tensor<32x17xf32>)
// CHECK-SAME:     outs(%[[ImagFill]] : tensor<17xf32>) -> tensor<17xf32>
// CHECK:        return %[[RealRes]], %[[ImagRes]] : tensor<17xf32>, tensor<17xf32>

// -----

func @rfft_2d(%input: tensor<1x32xf32>) -> (tensor<1x17xf32>, tensor<1x17xf32>) {
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<1xi64>, fft_type = "RFFT"
  } : (tensor<1x32xf32>) -> tensor<1x17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  return %1, %2 : tensor<1x17xf32>, tensor<1x17xf32>
}
// CHECK:     func @rfft_2d
// CHECK-SAME:  %[[Arg0:[a-zA-Z0-9_]*]]
// CHECK-DAG:   %[[RealMatrix:.+]] = constant dense<"0x0000803F{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[ImagMatrix:.+]] = constant dense<"0x00000080{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[Zero:.+]] = constant 0.000000e+00 : f32
// CHECK:        %[[RealInit:.+]] = linalg.init_tensor [1, 17] : tensor<1x17xf32>
// CHECK:        %[[RealFill:.+]] = linalg.fill(%[[RealInit]], %[[Zero]])
// CHECK:        %[[RealRes:.+]] = linalg.matmul
// CHECK-SAME:     ins(%[[Arg0]], %[[RealMatrix]] : tensor<1x32xf32>, tensor<32x17xf32>)
// CHECK-SAME:     outs(%[[RealFill]] : tensor<1x17xf32>) -> tensor<1x17xf32>
// CHECK:        %[[ImagInit:.+]] = linalg.init_tensor [1, 17] : tensor<1x17xf32>
// CHECK:        %[[ImagFill:.+]] = linalg.fill(%[[ImagInit]], %[[Zero]])
// CHECK:        %[[ImagRes:.+]] = linalg.matmul
// CHECK-SAME:     ins(%[[Arg0]], %[[ImagMatrix]] : tensor<1x32xf32>, tensor<32x17xf32>)
// CHECK-SAME:     outs(%[[ImagFill]] : tensor<1x17xf32>) -> tensor<1x17xf32>
// CHECK:        return %[[RealRes]], %[[ImagRes]] : tensor<1x17xf32>, tensor<1x17xf32>
