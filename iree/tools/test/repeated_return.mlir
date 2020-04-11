// RUN: (iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=many_tensor) | IreeFileCheck %s

// (only checking exit codes).
// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vmla --entry_function=many_tensor --input_file=${TEST_TMPDIR?}/bc.module

// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @many_tensor
func @many_tensor() -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
                        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) attributes { iree.module.export } {
  %res = iree.unfoldable_constant
      dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %res, %res, %res, %res, %res, %res :
        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x2xf32>
}
// CHECK-COUNT-6: 2x2xf32=[1 2][3 4]
