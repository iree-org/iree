// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.bounds

// The dot lowering drops the encoding from its result type; without the
// pass the return type no longer matches the function type.
// CHECK-LABEL: @bounds_dot
// CHECK: util.assume.int %{{.+}}<umax = 16> : index
// CHECK: linalg.matmul
// CHECK: return %{{.+}} : tensor<?x2xf32>
func.func @bounds_dot(%a: tensor<?x8xf32, #stablehlo.bounds<16, ?>>, %b: tensor<8x2xf32>)
    -> tensor<?x2xf32, #stablehlo.bounds<16, ?>> {
  %r = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0]
    : (tensor<?x8xf32, #stablehlo.bounds<16, ?>>, tensor<8x2xf32>) -> tensor<?x2xf32, #stablehlo.bounds<16, ?>>
  return %r : tensor<?x2xf32, #stablehlo.bounds<16, ?>>
}

// -----

// A batched, non-square matmul with bounds on the batch and the row dims.
// CHECK-LABEL: @bounds_batch_matmul
// CHECK-DAG: util.assume.int %{{.+}}<umax = 4> : index
// CHECK-DAG: util.assume.int %{{.+}}<umax = 32> : index
// CHECK: linalg.batch_matmul
// CHECK: return %{{.+}} : tensor<?x?x3xf32>
func.func @bounds_batch_matmul(%a: tensor<?x?x5xf32, #stablehlo.bounds<4, 32, ?>>,
                               %b: tensor<?x5x3xf32, #stablehlo.bounds<4, ?, ?>>)
    -> tensor<?x?x3xf32, #stablehlo.bounds<4, 32, ?>> {
  %r = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    : (tensor<?x?x5xf32, #stablehlo.bounds<4, 32, ?>>, tensor<?x5x3xf32, #stablehlo.bounds<4, ?, ?>>)
    -> tensor<?x?x3xf32, #stablehlo.bounds<4, 32, ?>>
  return %r : tensor<?x?x3xf32, #stablehlo.bounds<4, 32, ?>>
}
