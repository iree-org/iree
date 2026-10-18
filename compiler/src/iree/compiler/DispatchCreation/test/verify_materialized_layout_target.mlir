// RUN: iree-opt --split-input-file --iree-dispatch-creation-verify-materialized-layout-target %s | FileCheck %s

// The module is still compiled for the target its layouts were materialized
// for.
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
// CHECK: #[[TARGET:[a-z0-9_]+]] = #hal.executable.target<"llvm-cpu"
// CHECK: #[[DEVICE:[a-z0-9_]+]] = #hal.device.target<"local", [#[[TARGET]]]>
// CHECK: module attributes {iree.encoding.materialized_layout_target = #[[TARGET]], stream.affinity.default
// CHECK:   util.global private @device = #[[DEVICE]]
module attributes {iree.encoding.materialized_layout_target = #target, stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#target]> : !hal.device
}

// -----

// Without materialized layouts, any set of targets is accepted.
// CHECK-NOT: materialized_layout_target
// CHECK: module attributes {stream.affinity.default
// CHECK:   util.global private @device
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64">, #hal.executable.target<"vmvx", "vmvx-bytecode-fb">]> : !hal.device
}
