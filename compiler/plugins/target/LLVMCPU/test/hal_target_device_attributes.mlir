// This test aims to check default HAL properties for LLVM CPU target, and
// whether CLI options modify the values correctly.

// TODO: Expand the test for more CLI configurations, e.g. different target triples

// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=x86_64-linux-gnu %s \
// RUN: | FileCheck %s --check-prefix=CHECK-X86-DEFAULT
//
// CHECK-X86-DEFAULT: module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
// CHECK-X86-DEFAULT-NEXT: util.global private @__device_0 = #hal.device.target<"local",
// CHECK-X86-DEFAULT-SAME: [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = ""
// CHECK-X86-DEFAULT-SAME: max_stack_allocation_size = 32768 : i64
// CHECK-X86-DEFAULT-SAME: native_vector_size = 16 : i64
// CHECK-X86-DEFAULT-SAME: target_triple = "x86_64-unknown-unknown-eabi-elf"
// CHECK-X86-DEFAULT-SAME: }>]> : !hal.device

// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=x86_64-linux-gnu %s \
// RUN:              --iree-llvmcpu-stack-allocation-limit=65536 \
// RUN: | FileCheck %s --check-prefix=CHECK-STACK-VALUE
//
// CHECK-STACK-VALUE: module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
// CHECK-STACK-VALUE-NEXT: util.global private @__device_0 = #hal.device.target<"local",
// CHECK-STACK-VALUE-SAME: [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = ""
//
// CHECK-STACK-VALUE-SAME: max_stack_allocation_size = 65536 : i64
//
// CHECK-STACK-VALUE-SAME: }>]> : !hal.device

// RUN: not iree-compile --compile-to=preprocessing --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=x86_64-linux-gnu %s \
// RUN:                  --iree-llvmcpu-stack-allocation-limit=64266 \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-INCORRECT-OPT-STACK-VALUE
//
// CHECK-INCORRECT-OPT-STACK-VALUE: for the --iree-llvmcpu-stack-allocation-limit option: '64266' value not a power-of-two

module {
  util.func public @foo(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    util.return %arg0 : tensor<?xf32>
  }
}
