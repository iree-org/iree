// RUN: (iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=multi_input --inputs="2xi32=[1 2], 2xi32=[3 4]") | IreeFileCheck %s

// (only checking exit codes).
// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vmla --entry_function=multi_input --inputs="2xi32=[1 2], 2xi32=[3 4]" --input_file=${TEST_TMPDIR?}/bc.module

// RUN: (iree-run-mlir --iree-hal-target-backends=vmla --input-value="2xi32=[1 2]" --input-value="2xi32=[3 4]" %s) | IreeFileCheck %s

// CHECK-LABEL: EXEC @multi_input
func @multi_input(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) attributes { iree.module.export } {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
// CHECK: 2xi32=1 2
// CHECK: 2xi32=3 4
