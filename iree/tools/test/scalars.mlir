// RUN: (iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=scalar --function_input=i32=42) | IreeFileCheck %s
// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vmla --entry_function=scalar --function_input=i32=42 | IreeFileCheck --check-prefix=BENCHMARK %s
// RUN: (iree-run-mlir --iree-hal-target-backends=vmla --function-input=i32=42 %s) | IreeFileCheck %s

// BENCHMARK-LABEL: BM_scalar
// CHECK-LABEL: EXEC @scalar
func @scalar(%arg0 : i32) -> i32 attributes { iree.module.export } {
  return %arg0 : i32
}
// CHECK: i32=42
