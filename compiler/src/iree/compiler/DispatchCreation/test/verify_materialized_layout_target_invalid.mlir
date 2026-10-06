// RUN: iree-opt --split-input-file --iree-dispatch-creation-verify-materialized-layout-target --verify-diagnostics %s
// The verification runs at the start of dispatch creation, also without data
// tiling.
// RUN: iree-opt --split-input-file --iree-dispatch-creation-pipeline --verify-diagnostics %s

#original = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#changed = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {target_triple = "x86_64-unknown-unknown-eabi-elf"}>
// expected-error @+1 {{cannot retarget a module with layouts materialized for}}
module attributes {iree.encoding.materialized_layout_target = #original, stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#changed]> : !hal.device
}

// -----

#original = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
// expected-error @+1 {{expected a single executable target for a module with materialized layouts, but found 2}}
module attributes {iree.encoding.materialized_layout_target = #original, stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#original, #hal.executable.target<"vmvx", "vmvx-bytecode-fb">]> : !hal.device
}

// -----

// expected-error @+1 {{expected a single executable target for a module with materialized layouts, but found 0}}
module attributes {iree.encoding.materialized_layout_target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64">} {
}

// -----

// expected-error @+1 {{expected 'iree.encoding.materialized_layout_target' to be a #hal.executable.target}}
module attributes {iree.encoding.materialized_layout_target = "llvm-cpu"} {
}
