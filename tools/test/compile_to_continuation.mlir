// RUN: iree-compile --compile-to=input %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=INPUT-PHASE
// INPUT-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=abi %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=ABI-PHASE
// ABI-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=flow --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=FLOW-PHASE
// FLOW-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=flow %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=FLOW-PHASE-NO-DEVICE
// FLOW-PHASE-NO-DEVICE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=stream --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=STREAM-PHASE
// STREAM-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=stream %s | \
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=STREAM-PHASE-NO-DEVICE
// STREAM-PHASE-NO-DEVICE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=executable-sources --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=EXECUTABLE-SOURCES-PHASE
// EXECUTABLE-SOURCES-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=executable-targets --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=EXECUTABLE-TARGETS-PHASE
// EXECUTABLE-TARGETS-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=hal --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=HAL-PHASE
// HAL-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=vm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=VM-PHASE
// VM-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=input %s | \
// RUN: iree-compile --compile-from=input --output-format=vm-asm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=FROM-ABI-PHASE
// FROM-INPUT-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=abi %s | \
// RUN: iree-compile --compile-from=abi --output-format=vm-asm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx - | \
// RUN: FileCheck %s --check-prefix=FROM-ABI-PHASE
// FROM-ABI-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=flow --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --compile-from=flow --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=FROM-FLOW-PHASE
// FROM-FLOW-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=stream --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --compile-from=stream --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=FROM-STREAM-PHASE
// FROM-STREAM-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=executable-sources --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --compile-from=executable-sources --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=FROM-EXECUTABLE-SOURCES-PHASE
// FROM-EXECUTABLE-SOURCES-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=executable-targets --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --compile-from=executable-targets --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=FROM-EXECUTABLE-TARGETS-PHASE
// FROM-EXECUTABLE-TARGETS-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=hal --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --compile-from=hal --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=FROM-HAL-PHASE
// FROM-HAL-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

// RUN: iree-compile --compile-to=vm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: iree-compile --compile-from=vm --output-format=vm-asm - | \
// RUN: FileCheck %s --check-prefix=FROM-VM-PHASE
// FROM-VM-PHASE: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>

func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
