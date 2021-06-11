// RUN: iree-translate --iree-hal-target-backends=vmvx -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=multi_input --function_input="2xi32=[1 2]" --function_input="2xi32=[3 4]" | IreeFileCheck %s
// RUN: iree-run-mlir --iree-hal-target-backends=vmvx --function-input='2xi32=[1 2]' --function-input='2xi32=[3 4]' %s | IreeFileCheck %s
// RUN: iree-translate --iree-hal-target-backends=vmvx -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vmvx --entry_function=multi_input --function_input="2xi32=[1 2]" --function_input="2xi32=[3 4]" | IreeFileCheck --check-prefix=BENCHMARK %s

// BENCHMARK-LABEL: BM_multi_input
// CHECK-LABEL: EXEC @multi_input
func @multi_input(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) attributes { iree.module.export } {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
// CHECK: 2xi32=1 2
// CHECK: 2xi32=3 4
