// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-assign-constant-ordinals)))" --split-input-file %s | FileCheck %s

hal.executable private @executable {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "bar" {
      %c1 = arith.constant 1 : i32
      hal.return %c1 : i32
    }
    builtin.module {
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_foo_a(0 : i32)
      llvm.mlir.global internal @__constant_ordinal_foo_a() {addr_space = 0 : i32, hal.executable.constant.key = "foo", sym_visibility = "private"} : i32
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_foo_b(0 : i32)
      llvm.mlir.global internal @__constant_ordinal_foo_b() {addr_space = 0 : i32, hal.executable.constant.key = "foo", sym_visibility = "private"} : i32
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_bar(1 : i32)
      llvm.mlir.global internal @__constant_ordinal_bar() {addr_space = 0 : i32, hal.executable.constant.key = "bar", sym_visibility = "private"} : i32
    }
  }
}
