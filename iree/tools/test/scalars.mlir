// RUN: (iree-translate --iree-hal-target-backends=vmvx -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=scalar --function_input=42) | IreeFileCheck %s
// RUN: iree-translate --iree-hal-target-backends=vmvx -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vmvx --entry_function=scalar --function_input=42 | IreeFileCheck --check-prefix=BENCHMARK %s
// RUN: (iree-run-mlir --iree-hal-target-backends=vmvx --function-input=42 %s) | IreeFileCheck %s

// BENCHMARK-LABEL: BM_scalar
// CHECK-LABEL: EXEC @scalar
func @scalar(%arg0 : i32) -> i32 {
  return %arg0 : i32
}
// CHECK: i32=42
