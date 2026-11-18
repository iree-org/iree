// RUN: iree-opt --split-input-file --iree-global-opt-early-data-tiling %s | FileCheck %s --check-prefix=EARLY --implicit-check-not=iree.encoding.materialized_layout_target --implicit-check-not=linalg.pack --implicit-check-not=iree_encoding.set_encoding
// RUN: iree-opt --split-input-file --iree-global-opt-early-data-tiling --iree-dispatch-creation-assign-data-tiling-encodings %s | FileCheck %s --check-prefix=LATE

// Modules whose target does not support host encoding materialization are left
// unchanged by the early pass and still get encodings at dispatch time.
// A backend other than llvm-cpu.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]> : !hal.device
  // EARLY-LABEL: util.func public @vmvx(
  // EARLY:         linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>)
  // LATE-LABEL:  util.func public @vmvx(
  // LATE:          iree_encoding.set_encoding
  // LATE:          linalg.matmul
  // LATE-SAME:       tensor<64x128xf32, #{{.*}}encoding
  // LATE:          iree_encoding.unset_encoding
  util.func public @vmvx(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}

// -----

// A heterogeneous module with a supported and an unsupported target.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf", cpu_features = "+neon", native_vector_size = 16 : i64, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>, #hal.executable.target<"vmvx", "vmvx-bytecode-fb">]> : !hal.device
  // EARLY-LABEL: util.func public @multiple_targets(
  // EARLY:         linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>)
  // LATE-LABEL:  util.func public @multiple_targets(
  // LATE:          iree_encoding.set_encoding
  // LATE:          linalg.matmul
  // LATE-SAME:       tensor<64x128xf32, #{{.*}}encoding
  // LATE:          iree_encoding.unset_encoding
  util.func public @multiple_targets(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}

// -----

// Inner-tiled layouts are not materialized on the host.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf", enable_inner_tiled = true, cpu_features = "+neon", native_vector_size = 16 : i64, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>]> : !hal.device
  // EARLY-LABEL: util.func public @inner_tiled(
  // EARLY:         linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>)
  // LATE-LABEL:  util.func public @inner_tiled(
  // LATE:          iree_encoding.set_encoding
  // LATE:          linalg.matmul
  // LATE-SAME:       tensor<64x128xf32, #{{.*}}encoding
  // LATE:          iree_encoding.unset_encoding
  util.func public @inner_tiled(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}

// -----

// A CPU target without the CPU encoding resolver.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf", cpu_features = "+neon", native_vector_size = 16 : i64}>]> : !hal.device
  // EARLY-LABEL: util.func public @missing_resolver(
  // EARLY:         linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>)
  // LATE-LABEL:  util.func public @missing_resolver(
  // LATE:          iree_encoding.set_encoding
  // LATE:          linalg.matmul
  // LATE-SAME:       tensor<64x128xf32, #{{.*}}encoding
  // LATE:          iree_encoding.unset_encoding
  util.func public @missing_resolver(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}

// -----

// A CPU architecture other than x86_64 and AArch64.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c", native_vector_size = 16 : i64, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>]> : !hal.device
  // EARLY-LABEL: util.func public @unsupported_arch(
  // EARLY:         linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>)
  // LATE-LABEL:  util.func public @unsupported_arch(
  // LATE:          iree_encoding.set_encoding
  // LATE:          linalg.matmul
  // LATE-SAME:       tensor<64x128xf32, #{{.*}}encoding
  // LATE:          iree_encoding.unset_encoding
  util.func public @unsupported_arch(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}
