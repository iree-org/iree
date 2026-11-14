// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-spirv-emulate-f16-storage,canonicalize,cse))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-spirv-emulate-f16-storage,iree-spirv-emulate-f16-storage,canonicalize,cse))' %s | FileCheck %s

// The second RUN line runs the emulation twice: emulating already emulated
// IR must change nothing.

// Neither f16 arithmetic nor 16-bit storage: f16 is computed in f32 and
// stored in 32-bit words.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
func.func @half_storage_b32(%i: index, %j: index) attributes {hal.executable.target = #target} {
  %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %value = memref.load %in[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %sum = arith.addf %value, %value : f16
  %signed = math.copysign %sum, %value : f16
  memref.store %signed, %out[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  return
}
// CHECK-LABEL: func.func @half_storage_b32
// CHECK: hal.interface.binding.subspan {{.+}} binding(0) : memref<8xi32,
// CHECK: memref.load
// CHECK: arith.addf {{.+}} : f32
// CHECK: arith.bitcast {{.+}} : f32 to i32
// CHECK: arith.andi {{.+}} : i16
// CHECK: arith.ori {{.+}} : i16
// CHECK: memref.atomic_rmw andi {{.+}} : (i32, memref<8xi32,
// CHECK: memref.atomic_rmw ori {{.+}} : (i32, memref<8xi32,

// -----

// 16-bit storage without f16 arithmetic still needs f32 computation, and
// f16 storage is emulated in words with it.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32|b16, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
func.func @half_storage_b16(%i: index, %j: index) attributes {hal.executable.target = #target} {
  %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %value = memref.load %in[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %sum = arith.addf %value, %value : f16
  %signed = math.copysign %sum, %value : f16
  memref.store %signed, %out[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  return
}
// CHECK-LABEL: func.func @half_storage_b16
// CHECK: hal.interface.binding.subspan {{.+}} binding(0) : memref<8xi32,
// CHECK: memref.load
// CHECK: arith.addf {{.+}} : f32
// CHECK: arith.bitcast {{.+}} : f32 to i32
// CHECK: arith.andi {{.+}} : i16
// CHECK: arith.ori {{.+}} : i16
// CHECK: memref.atomic_rmw andi {{.+}} : (i32, memref<8xi32,
// CHECK: memref.atomic_rmw ori {{.+}} : (i32, memref<8xi32,

// -----

// f16 arithmetic without 16-bit storage: f16 is stored in 32-bit words.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|fp16|int32|int16, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
func.func @half_storage_fp16_b32(%i: index, %j: index) attributes {hal.executable.target = #target} {
  %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %value = memref.load %in[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %sum = arith.addf %value, %value : f16
  %signed = math.copysign %sum, %value : f16
  memref.store %signed, %out[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  return
}
// CHECK-LABEL: func.func @half_storage_fp16_b32
// CHECK: hal.interface.binding.subspan {{.+}} binding(0) : memref<8xi32,
// CHECK: memref.load
// CHECK: arith.addf {{.+}} : f32
// CHECK: arith.bitcast {{.+}} : f32 to i32
// CHECK: arith.andi {{.+}} : i16
// CHECK: arith.ori {{.+}} : i16
// CHECK: memref.atomic_rmw andi {{.+}} : (i32, memref<8xi32,
// CHECK: memref.atomic_rmw ori {{.+}} : (i32, memref<8xi32,

// -----

// f16 arithmetic and 16-bit storage: nothing is emulated.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|fp16|int32|int16, storage = b32|b16, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
func.func @half_storage_native(%i: index, %j: index) attributes {hal.executable.target = #target} {
  %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %value = memref.load %in[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %sum = arith.addf %value, %value : f16
  %signed = math.copysign %sum, %value : f16
  memref.store %signed, %out[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  return
}
// CHECK-LABEL: func.func @half_storage_native
// CHECK: hal.interface.binding.subspan {{.+}} binding(0) : memref<3x5xf16,
// CHECK: memref.load
// CHECK: arith.addf {{.+}} : f16
// CHECK: math.copysign {{.+}} : f16
// CHECK: memref.store {{.+}} : memref<3x5xf16,

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
// A bitcast round trip must preserve every payload bit, including signalling
// NaNs. This path needs neither numeric conversion nor half-pack intrinsics.
// CHECK-LABEL: func.func @half_payload_bits(
// CHECK-SAME: %[[BITS:.+]]: i16)
// CHECK-NOT: arith.
// CHECK-NOT: spirv.GL.
// CHECK: return %[[BITS]] : i16
func.func @half_payload_bits(%bits: i16) -> i16 attributes {hal.executable.target = #target} {
  %half = arith.bitcast %bits : i16 to f16
  %result = arith.bitcast %half : f16 to i16
  return %result : i16
}

// -----

// CHECK-LABEL: func.func @no_target
// CHECK: arith.addf {{.+}} : f16
func.func @no_target(%a: f16, %b: f16) -> f16 {
  %result = arith.addf %a, %b : f16
  return %result : f16
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
// A function without half values needs no emulation pipeline. In particular,
// it must not acquire unrelated math restrictions from another float profile.
// CHECK-LABEL: func.func @no_half_values(
// CHECK: math.fma {{.+}} : f8E5M2
func.func @no_half_values(%a: f8E5M2, %b: f8E5M2, %c: f8E5M2) -> f8E5M2 attributes {hal.executable.target = #target} {
  %result = math.fma %a, %b, %c : f8E5M2
  return %result : f8E5M2
}

// -----

// Calls and function signatures must agree after half storage conversion.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
func.func private @helper(%a: f16) -> f16 attributes {hal.executable.target = #target} {
  %sum = arith.addf %a, %a : f16
  return %sum : f16
}
func.func @caller(%a: f16) -> f16 attributes {hal.executable.target = #target} {
  %r = func.call @helper(%a) : (f16) -> f16
  return %r : f16
}

// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: %[[ARG:.+]]: i16) -> i16
// CHECK: arith.addf {{.+}} : f32
// CHECK: return {{.+}} : i16
// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %[[ARG:.+]]: i16) -> i16
// CHECK: %[[RESULT:.+]] = call @helper(%[[ARG]]) : (i16) -> i16
// CHECK: return %[[RESULT]] : i16

// -----

// f16 arithmetic and 16-bit storage without 16-bit integers: f16 values
// still need no emulation.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|fp16|int32, storage = b32|b16, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
func.func @half_storage_native_without_int16(%i: index, %j: index) attributes {hal.executable.target = #target} {
  %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %value = memref.load %in[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  %sum = arith.addf %value, %value : f16
  %signed = math.copysign %sum, %value : f16
  memref.store %signed, %out[%i, %j] : memref<3x5xf16, #spirv.storage_class<StorageBuffer>>
  return
}
// CHECK-LABEL: func.func @half_storage_native_without_int16
// CHECK: hal.interface.binding.subspan {{.+}} binding(0) : memref<3x5xf16,
// CHECK: memref.load
// CHECK: arith.addf {{.+}} : f16
// CHECK: math.copysign {{.+}} : f16
// CHECK: memref.store {{.+}} : memref<3x5xf16,
