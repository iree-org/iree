// RUN: iree-compile %s -o ignored.mlir \
// RUN:     --iree-hal-target-backends=vmvx \
// RUN:     --iree-hal-dump-executable-benchmarks-to=- | \
// RUN: iree-compile - --output-format=vm-asm | \
// RUN: FileCheck %s

// Tests that it's possible to round-trip executable benchmarks produced by the
// compiler back to the compiler individually. This test relies on us piping
// stdout and that there's only a single executable (otherwise we'd need to look
// at files and that's harder cross-platform).

func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}

// We expect one executable and one exported function with the reflection attrs.
// CHECK: vm.rodata private @abs_dispatch_0_vmvx_bytecode_fb
// CHECK: vm.func private @abs_dispatch_0_vmvx_bytecode_fb_abs_dispatch_0{{.+}}(%arg0: i32)
// CHECK-SAME: iree.reflection = {iree.benchmark = "dispatch"}
// CHECK: vm.call @hal.command_buffer.dispatch
