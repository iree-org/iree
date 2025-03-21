// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx %s | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu %s | FileCheck %s

// CHECK-LABEL: EXEC @scalar
func.func @scalar() -> i32 {
  %result = util.unfoldable_constant 42 : i32
  return %result : i32
}
// CHECK: i32=42
