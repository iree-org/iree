// RUN: iree-opt --split-input-file --iree-hal-resolve-device-aliases --iree-hal-verify-devices %s --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// Tests that modules without tensors don't need devices.

module @module {
  // CHECK: util.func private @func
  util.func private @func() -> ()
}

// -----

// TODO(multi-device): find a way to verify that devices exist if they need to.
// Currently the check is disabled as it's difficult to tell if a device will be
// needed by the time we get to the HAL layer: plugins may absorb things, etc.
// NO-expected-errorx@+1 {{no HAL devices defined in the module}}
module @module {
  util.func private @func() -> () {
    arith.constant dense<1.0> : tensor<4xf32>
    util.return
  }
}

// -----

module @module {
  // expected-error@+1 {{unregistered target device "__unregistered__"}}
  util.global private @device = #hal.device.target<"__unregistered__"> : !hal.device
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.affinity<@device>
  }
}

// -----

module @module {
  // expected-error@+1 {{unregistered target device "__unregistered__"}}
  util.global private @device = #hal.device.select<[
    #hal.device.target<"local"> : !hal.device,
    #hal.device.target<"__unregistered__"> : !hal.device
  ]> : !hal.device
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.affinity<@device>
  }
}

// -----

// Valid input with proper attributes.

// CHECK: module @module
module @module {
  util.global private @device = #hal.device.target<"local", {ordinal = 0 : index}> : !hal.device
  util.global private @optional = #hal.device.fallback<@device> : !hal.device
  util.global private @ordinal = #hal.device.ordinal<0> : !hal.device
  util.global private @selected = #hal.device.select<[
    #hal.device.target<"local", {ordinal = 0 : index}> : !hal.device,
    #hal.device.target<"local", {ordinal = 1 : index}> : !hal.device
  ]> : !hal.device
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.affinity<@device>
  }
}

// -----

// Modules without anything that needs an environment are OK.

// CHECK: module @module
module @module {
  hal.executable private @exe {
    hal.executable.variant public @embedded_elf_arm_64 target(#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {}>) {}
  }
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

module @native_storage_without_half_compute {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#target]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp16|fp32|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

module @half_compute_without_int16 {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#target]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int16|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

module @int16_without_half_compute {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#target]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp16|fp32|int16|int32, storage = b8|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

module @half_compute_without_half_storage {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#target]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp16|fp32|int16|int32, storage = b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

module @half_support_without_byte_storage {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#target]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp16|fp32|int16|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

// CHECK-LABEL: module @native_half_requires_no_word_padding
// CHECK: #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64},
module @native_half_requires_no_word_padding {
  util.global private @device = #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#target]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

// Explicit profiles may use their required unit or a stricter unit.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
// CHECK-LABEL: module @explicit_units
// CHECK-DAG: stream.min_access_granularity = 4 : i64
// CHECK-DAG: stream.min_access_granularity = 8 : i64
module @explicit_units {
  util.global private @exact = #hal.device.target<"vulkan", {stream.min_access_granularity = 4 : i64, stream.min_fill_granularity = 4 : i64}, [#target]> : !hal.device
  util.global private @larger = #hal.device.target<"vulkan", {stream.min_access_granularity = 8 : i64, stream.min_fill_granularity = 8 : i64}, [#target]> : !hal.device
  util.func private @func()
}

// -----

module @vulkan_fill_requirement {
  // expected-error@+1 {{device target requires stream.min_fill_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.target<"vulkan"> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

module @metal_fill_requirement {
  // expected-error@+1 {{device target requires stream.min_fill_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.target<"metal"> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

module @alias_cannot_weaken_storage_requirement {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.alias<"vulkan", {stream.min_access_granularity = 1 : i64}> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

module @alias_cannot_weaken_fill_requirement {
  // expected-error@+1 {{device target requires stream.min_fill_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.alias<"metal", {stream.min_fill_granularity = 1 : i64}> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

// CHECK-LABEL: module @alias_can_strengthen_requirements
// CHECK: stream.min_access_granularity = 8 : i64, stream.min_fill_granularity = 8 : i64
module @alias_can_strengthen_requirements {
  util.global private @device = #hal.device.alias<"vulkan", {stream.min_access_granularity = 8 : i64, stream.min_fill_granularity = 8 : i64}> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

#native = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp16|fp32|int16|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#word = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

module @all_select_members_native_first {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.select<[#hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#native]> : !hal.device, #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#word]> : !hal.device]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}

// -----

#native = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp16|fp32|int16|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#word = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>

module @all_select_members_word_first {
  // expected-error@+1 {{device target requires stream.min_access_granularity of at least 4 bytes; got 1}}
  util.global private @device = #hal.device.select<[#hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#word]> : !hal.device, #hal.device.target<"vulkan", {stream.min_fill_granularity = 4 : i64}, [#native]> : !hal.device]> : !hal.device
  util.func private @func() attributes {stream.affinity = #hal.device.affinity<@device>}
}
