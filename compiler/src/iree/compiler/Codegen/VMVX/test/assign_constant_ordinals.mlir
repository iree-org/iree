// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-vmvx-assign-constant-ordinals)))" --split-input-file %s | FileCheck %s

hal.executable private @executable {
  hal.executable.variant public @variant target(#hal.executable.target<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "bar" {
      %c1 = arith.constant 1 : i32
      hal.return %c1 : i32
    }
    builtin.module {
      vm.module @inner {
        // CHECK: vm.global.i32 private @__constant_ordinal_foo_a = 0 : i32
        vm.global.i32 private mutable @__constant_ordinal_foo_a {hal.executable.constant.key = "foo"} : i32
        // CHECK: vm.global.i32 private @__constant_ordinal_foo_b = 0 : i32
        vm.global.i32 private mutable @__constant_ordinal_foo_b {hal.executable.constant.key = "foo"} : i32
        // CHECK: vm.global.i32 private @__constant_ordinal_bar = 1 : i32
        vm.global.i32 private mutable @__constant_ordinal_bar {hal.executable.constant.key = "bar"} : i32
      }
    }
  }
}
