// RUN: iree-opt --iree-abi-transformation-pipeline \
// RUN:          --iree-flow-transformation-pipeline \
// RUN:          --iree-flow-dispatch-use-transform-dialect-jit \
// RUN:          --iree-flow-dispatch-use-transform-dialect-debug-emit-remarks \
// RUN:          --verify-diagnostics --split-input-file %s | FileCheck %s

// Check that the transform dialect-based dispatch region formation successfully
// kicks in. We request it to emit remarks when a match happens. Only do the
// basic matching of the actual region here, it is tested more extensively in
// Flow transforms.

// CHECK: flow.executable private @[[EXEC_1_NAME:.+]] {
// CHECK: builtin.module
// CHECK: func.func @[[FUNC_1_NAME:.+]](
func.func @multiple_reductions(%arg0: tensor<8x479xf32>, %arg1: tensor<32x32xf32>) -> (tensor<8xf32>, tensor<32xf32>) {
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store

  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  // expected-remark @below {{dispatch matched reduction}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg0 : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>

// CHECK: flow.executable private @[[EXEC_2_NAME:.+]] {
// CHECK: builtin.module
// CHECK: func.func @[[FUNC_2_NAME:.+]](
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store

  %empty2 = tensor.empty() : tensor<32xf32>
  %fill2 = linalg.fill ins(%cst : f32) outs(%empty2 : tensor<32xf32>) -> tensor<32xf32>
  // expected-remark @below {{dispatch matched reduction}}
  %result2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg1 : tensor<32x32xf32>)
    outs(%fill2 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<32xf32>

  return %result, %result2 : tensor<8xf32>, tensor<32xf32>
}
// CHECK: @multiple_reductions(
// CHECK: flow.dispatch @[[EXEC_1_NAME]]::@[[FUNC_1_NAME]]
// CHECK: flow.dispatch @[[EXEC_2_NAME]]::@[[FUNC_2_NAME]]

// -----

// Transform dialect-based dispatch region formation is not expected to handle
// this, but we need to check that the fallback happens.

// CHECK: @foo(
// CHECK: flow.tensor.splat
func.func @foo() -> tensor<8xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  return %fill : tensor<8xf32>
}
