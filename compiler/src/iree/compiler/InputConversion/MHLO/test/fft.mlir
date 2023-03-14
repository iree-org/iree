// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors --canonicalize %s | FileCheck %s

func.func @rfft_1d(%input: tensor<32xf32>) -> (tensor<17xf32>, tensor<17xf32>) {
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>
  } : (tensor<32xf32>) -> tensor<17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  return %1, %2 : tensor<17xf32>, tensor<17xf32>
}
// CHECK:     func.func @rfft_1d
// CHECK-SAME:  %[[Arg0:[a-zA-Z0-9_]*]]
// CHECK-DAG:   %[[RealMatrix:.+]] = arith.constant dense<"0x0000803F{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[ImagMatrix:.+]] = arith.constant dense<"0x00000080{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[Zero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[RealInit:.+]] = tensor.empty() : tensor<17xf32>
// CHECK:       %[[RealFill:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[Zero]] :
// CHECK-SAME:    outs(%[[RealInit]] :
// CHECK:       %[[RealRes:.+]] = linalg.vecmat
// CHECK-SAME:    ins(%[[Arg0]], %[[RealMatrix]] : tensor<32xf32>, tensor<32x17xf32>)
// CHECK-SAME:    outs(%[[RealFill]] : tensor<17xf32>) -> tensor<17xf32>
// CHECK:        %[[ImagInit:.+]] = tensor.empty() : tensor<17xf32>
// CHECK:        %[[ImagFill:.+]] = linalg.fill
// CHECK-SAME:     ins(%[[Zero]] :
// CHECK-SAME:     outs(%[[ImagInit]] :
// CHECK:        %[[ImagRes:.+]] = linalg.vecmat
// CHECK-SAME:     ins(%[[Arg0]], %[[ImagMatrix]] : tensor<32xf32>, tensor<32x17xf32>)
// CHECK-SAME:     outs(%[[ImagFill]] : tensor<17xf32>) -> tensor<17xf32>
// CHECK:        %[[ComplexRes:.*]] = linalg.generic
// CHECK:        %[[ReRes:.*]] = linalg.generic
// CHECK-SAME:     ins(%[[ComplexRes]]
// CHECK:        %[[ImRes:.*]] = linalg.generic
// CHECK-SAME:     ins(%[[ComplexRes]]
// CHECK:        return %[[ReRes]], %[[ImRes]] : tensor<17xf32>, tensor<17xf32>

// -----

func.func @rfft_2d(%input: tensor<1x32xf32>) -> (tensor<1x17xf32>, tensor<1x17xf32>) {
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>
  } : (tensor<1x32xf32>) -> tensor<1x17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  return %1, %2 : tensor<1x17xf32>, tensor<1x17xf32>
}
// CHECK:     func.func @rfft_2d
// CHECK-SAME:  %[[Arg0:[a-zA-Z0-9_]*]]
// CHECK-DAG:   %[[RealMatrix:.+]] = arith.constant dense<"0x0000803F{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[ImagMatrix:.+]] = arith.constant dense<"0x00000080{{.*}}"> : tensor<32x17xf32>
// CHECK-DAG:   %[[Zero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[RealInit:.+]] = tensor.empty() : tensor<1x17xf32>
// CHECK:        %[[RealFill:.+]] = linalg.fill
// CHECK-SAME:     ins(%[[Zero]] :
// CHECK-SAME:     outs(%[[RealInit]] :
// CHECK:        %[[RealRes:.+]] = linalg.matmul
// CHECK-SAME:     ins(%[[Arg0]], %[[RealMatrix]] : tensor<1x32xf32>, tensor<32x17xf32>)
// CHECK-SAME:     outs(%[[RealFill]] : tensor<1x17xf32>) -> tensor<1x17xf32>
// CHECK:        %[[ImagInit:.+]] = tensor.empty() : tensor<1x17xf32>
// CHECK:        %[[ImagFill:.+]] = linalg.fill
// CHECK-SAME:     ins(%[[Zero]] :
// CHECK-SAME:     outs(%[[ImagInit]] :
// CHECK:        %[[ImagRes:.+]] = linalg.matmul
// CHECK-SAME:     ins(%[[Arg0]], %[[ImagMatrix]] : tensor<1x32xf32>, tensor<32x17xf32>)
// CHECK-SAME:     outs(%[[ImagFill]] : tensor<1x17xf32>) -> tensor<1x17xf32>
// CHECK:        %[[ComplexRes:.*]] = linalg.generic
// CHECK:        %[[ReRes:.*]] = linalg.generic
// CHECK-SAME:     ins(%[[ComplexRes]]
// CHECK:        %[[ImRes:.*]] = linalg.generic
// CHECK-SAME:     ins(%[[ComplexRes]]
// CHECK:        return %[[ReRes]], %[[ImRes]] : tensor<1x17xf32>, tensor<1x17xf32>
