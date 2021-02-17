// RUN: iree-run-mlir -export-all -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=dylib-llvm-aot %s | IreeFileCheck %s)

flow.executable @ex0 {
  flow.dispatch.entry @dispatch0 attributes {workload = 4 : index}
  module {
    func @dispatch0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: EXEC @staticShapedFn
func @staticShapedFn() -> tensor<4xf32> {
  %input = iree.unfoldable_constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %workload = constant 4 : index
  %0 = flow.dispatch @ex0::@dispatch0[%workload](%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK: 4xf32=-2 4 -6 8
