// RUN: iree-compile --compile-to=input %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=INPUT-PHASE
// INPUT-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=abi %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=ABI-PHASE
// ABI-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=flow %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=FLOW-PHASE
// FLOW-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=stream %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=STREAM-PHASE
// STREAM-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=executable-sources --iree-hal-target-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=EXECUTABLE-SOURCES-PHASE
// EXECUTABLE-SOURCES-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=executable-targets --iree-hal-target-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=EXECUTABLE-TARGETS-PHASE
// EXECUTABLE-TARGETS-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=hal --iree-hal-target-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=HAL-PHASE
// HAL-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=vm --iree-hal-target-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=VM-PHASE
// VM-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
