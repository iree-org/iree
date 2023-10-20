// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-assign-import-ordinals)))" --split-input-file %s | FileCheck %s

// Tests that duplicate keys get the same ordinal and that ordinals are assigned
// in alphabetical order.

hal.executable private @executable {
  // CHECK: hal.executable.variant
  // CHECK-SAME: hal.executable.imports = {{.+}}["bar", true], ["foo", false]{{.+}}
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">) {
    builtin.module {
      // CHECK: llvm.mlir.global internal constant @__import_ordinal_foo_a(1 : i32)
      llvm.mlir.global internal @__import_ordinal_foo_a() {
        hal.executable.import.key = "foo"
      } : i32
      // CHECK: llvm.mlir.global internal constant @__import_ordinal_foo_b(1 : i32)
      llvm.mlir.global internal @__import_ordinal_foo_b() {
        hal.executable.import.key = "foo"
      } : i32
      // CHECK: llvm.mlir.global internal constant @__import_ordinal_bar(0 : i32)
      llvm.mlir.global internal @__import_ordinal_bar() {
        hal.executable.import.key = "bar",
        hal.executable.import.weak
      } : i32
    }
  }
}
