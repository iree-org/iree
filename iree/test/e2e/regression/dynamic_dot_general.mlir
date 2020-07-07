// RUN: iree-run-mlir %s -iree-hal-target-backends=vmla -input-value="2x2xf32=[[1.0, 0.0], [0.0, 1.0]]" -input-value="2x3xf32=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]" -input-value="2x2x2xf32=[[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]" -input-value="2x2x3xf32=[[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]" | IreeFileCheck %s

// TODO(silvasean): Extent xla_ops directory test infra to support
// testing dynamic shapes.

// CHECK-LABEL: EXEC @basic_dot
func @basic_dot(
  %lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
  %unused0: tensor<?x?x?xf32>, %unused1: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.dot_general"(%lhs, %rhs) {dot_dimension_numbers={
    lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
    lhs_contracting_dimensions = dense<1> : tensor<1xi64>,
    rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
    rhs_contracting_dimensions = dense<0> : tensor<1xi64>
  }} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK: 2x3xf32=[1 2 3][4 5 6]

// CHECK-LABEL: EXEC @batch_dimension
func @batch_dimension(
  %unused0: tensor<?x?xf32>, %unused1: tensor<?x?xf32>,
  %lhs: tensor<?x?x?xf32>, %rhs: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.dot_general"(%lhs, %rhs) {dot_dimension_numbers={
    lhs_batching_dimensions = dense<[0]> : tensor<1xi64>,
    lhs_contracting_dimensions = dense<[2]> : tensor<1xi64>,
    rhs_batching_dimensions = dense<[0]> : tensor<1xi64>,
    rhs_contracting_dimensions = dense<[1]> : tensor<1xi64>
  }} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK: 2x2x3xf32=[
// CHECK-SAME:   [1.5 2.5 3.5][4.5 5.5 6.5]
// CHECK-SAME: ][
// CHECK-SAME:   [2 4 6][8 10 12]
// CHECK-SAME: ]


// TODO(silvasean): Add more tests when we have better test infra.
// This is currently too verbose / unreadable. We should test:
// - multiple contracting dimensions
// - multiple batch dimensions
// - multiple free dimensions
// - intermingled batch, free, and contracting dimensions
