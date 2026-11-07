// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-vectorize-load-store),iree-codegen-flatten-memref-subspan,func.func(iree-spirv-emulate-f16-storage,iree-lower-affine,canonicalize,cse),iree-spirv-convert-gpu-target,iree-convert-to-spirv))))' %s | FileCheck %s

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
hal.executable private @half_storage_b32 {
  hal.executable.variant @vulkan target(#target) {
    hal.executable.export public @kernel layout(#layout) <workgroup_size = [1: index, 1: index, 1: index]>
    builtin.module {
      func.func @kernel() attributes {hal.executable.target = #target} {
        %i = arith.constant 0 : index
        %zero = arith.constant 0.0 : f16
        %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %value = vector.transfer_read %in[%i], %zero {in_bounds = [true]} : memref<16xf16, #spirv.storage_class<StorageBuffer>>, vector<4xf16>
        %sum = arith.addf %value, %value : vector<4xf16>
        vector.transfer_write %sum, %out[%i] {in_bounds = [true]} : vector<4xf16>, memref<16xf16, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32|b16, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
hal.executable private @half_storage_b16 {
  hal.executable.variant @vulkan target(#target) {
    hal.executable.export public @kernel layout(#layout) <workgroup_size = [1: index, 1: index, 1: index]>
    builtin.module {
      func.func @kernel() attributes {hal.executable.target = #target} {
        %i = arith.constant 0 : index
        %zero = arith.constant 0.0 : f16
        %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %value = vector.transfer_read %in[%i], %zero {in_bounds = [true]} : memref<16xf16, #spirv.storage_class<StorageBuffer>>, vector<4xf16>
        %sum = arith.addf %value, %value : vector<4xf16>
        vector.transfer_write %sum, %out[%i] {in_bounds = [true]} : vector<4xf16>, memref<16xf16, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|fp16|int32|int16, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
hal.executable private @half_storage_fp16_b32 {
  hal.executable.variant @vulkan target(#target) {
    hal.executable.export public @kernel layout(#layout) <workgroup_size = [1: index, 1: index, 1: index]>
    builtin.module {
      func.func @kernel() attributes {hal.executable.target = #target} {
        %i = arith.constant 0 : index
        %zero = arith.constant 0.0 : f16
        %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %value = vector.transfer_read %in[%i], %zero {in_bounds = [true]} : memref<16xf16, #spirv.storage_class<StorageBuffer>>, vector<4xf16>
        %sum = arith.addf %value, %value : vector<4xf16>
        vector.transfer_write %sum, %out[%i] {in_bounds = [true]} : vector<4xf16>, memref<16xf16, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|fp16|int32|int16, storage = b32|b16, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
hal.executable private @half_storage_native {
  hal.executable.variant @vulkan target(#target) {
    hal.executable.export public @kernel layout(#layout) <workgroup_size = [1: index, 1: index, 1: index]>
    builtin.module {
      func.func @kernel() attributes {hal.executable.target = #target} {
        %i = arith.constant 0 : index
        %zero = arith.constant 0.0 : f16
        %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %value = vector.transfer_read %in[%i], %zero {in_bounds = [true]} : memref<16xf16, #spirv.storage_class<StorageBuffer>>, vector<4xf16>
        %sum = arith.addf %value, %value : vector<4xf16>
        vector.transfer_write %sum, %out[%i] {in_bounds = [true]} : vector<4xf16>, memref<16xf16, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
}

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
hal.executable private @half_storage_offset_two {
  hal.executable.variant @vulkan target(#target) {
    hal.executable.export public @kernel layout(#layout) <workgroup_size = [1: index, 1: index, 1: index]>
    builtin.module {
      func.func @kernel() attributes {hal.executable.target = #target} {
        %i = arith.constant 0 : index
        %offset = arith.constant 2 : index
        %zero = arith.constant 0.0 : f16
        %in = hal.interface.binding.subspan layout(#layout) binding(0) offset(%offset) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %value = vector.transfer_read %in[%i], %zero {in_bounds = [true]} : memref<16xf16, #spirv.storage_class<StorageBuffer>>, vector<4xf16>
        %sum = arith.addf %value, %value : vector<4xf16>
        vector.transfer_write %sum, %out[%i] {in_bounds = [true]} : vector<4xf16>, memref<16xf16, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @half_storage_b32
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<8 x i32, stride=4>
// CHECK: spirv.FAdd
// CHECK-LABEL: hal.executable private @half_storage_b16
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<8 x i32, stride=4>
// CHECK: spirv.FAdd
// CHECK-LABEL: hal.executable private @half_storage_fp16_b32
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<8 x i32, stride=4>
// CHECK: spirv.FAdd
// CHECK-LABEL: hal.executable private @half_storage_native
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<4 x vector<4xf16>, stride=8>
// CHECK: spirv.FAdd
// CHECK-LABEL: hal.executable private @half_storage_offset_two
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<9 x i32, stride=4>
// CHECK: spirv.FAdd

// -----

// Three masked lanes occupy exactly two words; the fourth lane must not store.
#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
hal.executable private @half_storage_masked_tail {
  hal.executable.variant @vulkan target(#target) {
    hal.executable.export public @kernel layout(#layout) <workgroup_size = [1: index, 1: index, 1: index]>
    builtin.module {
      func.func @kernel() attributes {hal.executable.target = #target} {
        %i = arith.constant 0 : index
        %three = arith.constant 3 : index
        %mask = vector.create_mask %three : vector<4xi1>
        %zero = arith.constant 0.0 : f16
        %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<3xf16, #spirv.storage_class<StorageBuffer>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<3xf16, #spirv.storage_class<StorageBuffer>>
        %value = vector.transfer_read %in[%i], %zero, %mask {in_bounds = [true]} : memref<3xf16, #spirv.storage_class<StorageBuffer>>, vector<4xf16>
        %sum = arith.addf %value, %value : vector<4xf16>
        vector.transfer_write %sum, %out[%i], %mask {in_bounds = [true]} : vector<4xf16>, memref<3xf16, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
}


// CHECK-LABEL: hal.executable private @half_storage_masked_tail
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<2 x i32, stride=4>
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<2 x i32, stride=4>
// CHECK-NOT: !spirv.ptr<vector<
// CHECK: spirv.AtomicAnd
// CHECK: spirv.AtomicOr
// CHECK: spirv.AtomicAnd
// CHECK: spirv.AtomicOr
// CHECK: spirv.AtomicAnd
// CHECK: spirv.AtomicOr
// CHECK-NOT: spirv.Atomic
// CHECK: spirv.Return

// -----

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp32|fp16|int32, storage = b32|b16, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
hal.executable private @half_storage_native_without_int16 {
  hal.executable.variant @vulkan target(#target) {
    hal.executable.export public @kernel layout(#layout) <workgroup_size = [1: index, 1: index, 1: index]>
    builtin.module {
      func.func @kernel() attributes {hal.executable.target = #target} {
        %i = arith.constant 0 : index
        %zero = arith.constant 0.0 : f16
        %in = hal.interface.binding.subspan layout(#layout) binding(0) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) : memref<16xf16, #spirv.storage_class<StorageBuffer>>
        %value = vector.transfer_read %in[%i], %zero {in_bounds = [true]} : memref<16xf16, #spirv.storage_class<StorageBuffer>>, vector<4xf16>
        %sum = arith.addf %value, %value : vector<4xf16>
        vector.transfer_write %sum, %out[%i] {in_bounds = [true]} : vector<4xf16>, memref<16xf16, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
}


// CHECK-LABEL: hal.executable private @half_storage_native_without_int16
// CHECK: spirv.GlobalVariable {{.+}}!spirv.array<4 x vector<4xf16>, stride=8>
// CHECK: spirv.FAdd
