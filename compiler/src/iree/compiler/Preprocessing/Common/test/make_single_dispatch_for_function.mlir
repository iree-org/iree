// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-preprocessing-make-single-dispatch-for-function))" --split-input-file %s | FileCheck %s

func.func @simple_test() -> tensor<10x20xf32> {
  %0 = tensor.empty() : tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}
// CHECK-LABEL: func @simple_test() -> tensor<10x20xf32>
//  CHECK-NEXT:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     flow.return %[[EMPTY]]
//       CHECK:   return %[[DISPATCH]]

// -----

func.func @simple_test_with_cfg(%arg0 : i1) -> tensor<10x20xf32> {
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    %0 = tensor.empty() : tensor<10x20xf32>
    return %0 : tensor<10x20xf32>
  ^bb2:
    %1 = arith.constant dense<1.0> : tensor<10x20xf32>
    return %1 : tensor<10x20xf32>
}
// CHECK-LABEL: func @simple_test_with_cfg
//  CHECK-SAME:     %[[ARG0:.+]]: i1
//  CHECK-NEXT:   %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<10x20xf32>) {
//       CHECK:       cf.cond_br %[[ARG0]], ^[[BB1:[a-zA-Z0-9]+]], ^[[BB2:[a-zA-Z0-9]+]]
//       CHECK:     ^[[BB1]]:
//       CHECK:       %[[EMPTY:.+]] = tensor.empty
//       CHECK:       flow.return %[[EMPTY]]
//       CHECK:     ^[[BB2]]:
//       CHECK:       %[[CST:.+]] = arith.constant
//       CHECK:       flow.return %[[CST]]
//       CHECK:   }
//       CHECK:   return %[[DISPATCH]]
