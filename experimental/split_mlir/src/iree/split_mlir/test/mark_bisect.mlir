// RUN: iree-opt \
// RUN:   --split-input-file \
// RUN:   --iree-plugin=split_mlir \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-mark-bisect{functions=two_ops,too_few_ops,multiple_ops}))" %s \
// RUN: | FileCheck %s

// Each operation is marked as separate range.
// CHECK-LABEL: func.func @two_ops
func.func @two_ops(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//       CHECK: mhlo.constant
//   CHECK-DAG: outline_range_first
//   CHECK-DAG: outline_range_last
  %cts1 = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
//  CHECK-NEXT: mhlo.add
//   CHECK-DAG: outline_range_first
//   CHECK-DAG: outline_range_last
  %res = mhlo.add %arg0, %cts1 : tensor<2xf32>
  return %res : tensor<2xf32>
}

// -----

// Degenerate case with too few ops should not mark enything.
// CHECK-LABEL: func.func @too_few_ops
func.func @too_few_ops(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//      CHECK: mhlo.constant
//  CHECK-NOT: outline_range_first
//  CHECK-NOT: outline_range_last
  %cts1 = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
// CHECK-NEXT: return
//  CHECK-NOT: outline_range_first
//  CHECK-NOT: outline_range_last
  return %cts1 : tensor<2xf32>
}

// -----

// Multiple ops per range.
// CHECK-LABEL: func.func @multiple_ops
func.func @multiple_ops(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//       CHECK: outline_range_first
//  CHECK-SAME: dense<1.000000e+00>
  %cts1 = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
//  CHECK-NEXT: outline_range_last
//  CHECK-SAME: dense<2.000000e+00>
  %cts2 = mhlo.constant dense<2.000000e+00> : tensor<2xf32>
//  CHECK-NEXT: outline_range_first
//  CHECK-SAME: dense<3.000000e+00>
  %cts3 = mhlo.constant dense<3.000000e+00> : tensor<2xf32>
//  CHECK-NEXT: outline_range_last
//  CHECK-SAME: dense<4.000000e+00>
  %cts4 = mhlo.constant dense<4.000000e+00> : tensor<2xf32>
//  CHECK-NEXT: return
  return %cts1 : tensor<2xf32>
}

// -----

// Non-listed functions should not be marked.
// CHECK-LABEL: func.func @function_not_to_mark
func.func @function_not_to_mark(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//   CHECK-NOT: outline_range_first
//   CHECK-NOT: outline_range_last
  %cts1 = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
  %res = mhlo.add %arg0, %cts1 : tensor<2xf32>
//       CHECK: return
  return %res : tensor<2xf32>
}
