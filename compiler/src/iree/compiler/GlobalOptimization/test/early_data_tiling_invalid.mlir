// RUN: iree-opt --iree-global-opt-early-data-tiling --verify-diagnostics %s

// A module with materialized layouts cannot be compiled for another target.
#original = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#changed = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {target_triple = "x86_64-unknown-unknown-eabi-elf"}>
// expected-error @+1 {{cannot retarget a module with layouts materialized for}}
module attributes {iree.encoding.materialized_layout_target = #original, stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#changed]> : !hal.device
}
