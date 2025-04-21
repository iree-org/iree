// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --module=- --function=scalar --input=42) | \
// RUN:  FileCheck %s
// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-benchmark-module --device=local-task --module=- --function=scalar --input=42) | \
// RUN:  FileCheck --check-prefix=BENCHMARK %s
// RUN: (iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx %s --input=42) | \
// RUN:  FileCheck %s

// BENCHMARK-LABEL: BM_scalar
// CHECK-LABEL: EXEC @scalar
func.func @scalar(%arg0 : i32) -> i32 {
  return %arg0 : i32
}
// CHECK: i32=42
