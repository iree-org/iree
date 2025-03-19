// RUN: iree-opt --split-input-file --iree-hal-resolve-device-aliases --iree-hal-local-target-device-backends=llvm-cpu %s --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// CHECK: util.global private @device
// CHECK-SAME: #hal.device.target<"local"
// CHECK-SAME: extra_config = 4 : index
util.global private @device = #hal.device.alias<"local", {
  extra_config = 4 : index
}> : !hal.device

// -----

// CHECK: util.global private @device_ordinal
// CHECK-SAME: #hal.device.target<"local"
// CHECK-SAME: ordinal = 123 : index
util.global private @device_ordinal = #hal.device.alias<"local"[123]> : !hal.device

// -----

// CHECK: util.global private @device_select
// CHECK-SAME: #hal.device.select<[
// CHECK-SAME:  #hal.device.target<"local", {ordinal = 0 : index}
// CHECK-SAME:  #hal.device.target<"local", {ordinal = 1 : index}
util.global private @device_select = #hal.device.select<[
  #hal.device.alias<"local"[0]> : !hal.device,
  #hal.device.alias<"local"[1]> : !hal.device
]> : !hal.device

// -----

// expected-error@+1 {{unregistered device alias "__unregistered__"}}
util.global private @device_unregistered = #hal.device.alias<"__unregistered__"> : !hal.device

// -----

// expected-error@+1 {{unregistered device alias "__unregistered__"}}
util.global private @device_select_unregistered = #hal.device.select<[
  #hal.device.alias<"local"> : !hal.device,
  #hal.device.alias<"__unregistered__"> : !hal.device
]> : !hal.device
