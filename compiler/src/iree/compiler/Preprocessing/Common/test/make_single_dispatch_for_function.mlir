// RUN: iree-opt --iree-preprocessing--make-single-dispatch-for-function --split-input-file %s | FileCheck %s

func.func @simple_test() -> tensor<10x20xf32> {
  %0 = tensor.empty() : tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}
// CHECK-LABEL: func @simple_test() -> tensor<10x20xf32>
//  CHECK-NEXT:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     flow.return %[[EMPTY]]
//       CHECK:   return %[[DISPATCH]]
