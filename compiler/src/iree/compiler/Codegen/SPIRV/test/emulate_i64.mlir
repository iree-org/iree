// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-spirv-emulate-i64))' %s | \
// RUN:   FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @buffer_types() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<8xi32, #spirv.storage_class<StorageBuffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  %3 = memref.load %0[%c0] : memref<8xi32, #spirv.storage_class<StorageBuffer>>
  %4 = memref.load %1[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  %5 = arith.addi %4, %c1_i64 : i64
  memref.store %5, %2[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  return
}

// Check that without the Int64 capability emulation produces expected i32 ops.
//
// CHECK-LABEL: func.func @buffer_types
//       CHECK:   [[REF_I64_0:%.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_1:%.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI64:%.+]]      = memref.load [[REF_I64_0]][{{%.+}}] : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   {{%.+}}           = arith.addui_extended {{%.+}}, {{%.+}} : i32, i1
//       CHECK:   memref.store {{%.+}}, [[REF_I64_1]][{{%.+}}] : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @emulate_1d_vector() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c95232 = arith.constant 95232 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c36864 = arith.constant 36864 : index
  %c1523712 = arith.constant 1523712 : index
  %c96 = arith.constant 96 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>{%c96}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c1523712) : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>{%c36864}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>{%c36864}
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %thread_id_x = gpu.thread_id  x
  %3 = arith.muli %workgroup_id_x, %c32 : index
  %4 = arith.addi %thread_id_x, %3 : index
  %5 = memref.load %0[%4] : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>
  %6 = arith.extsi %5 : vector<4xi32> to vector<4xi64>
  %7 = arith.extui %5 : vector<4xi32> to vector<4xi64>
  %8 = arith.muli %6, %7 : vector<4xi64>
  %9 = arith.addi %6, %8 : vector<4xi64>
  %10 = arith.trunci %9 : vector<4xi64> to vector<4xi32>
  %11 = arith.muli %workgroup_id_y, %c96 : index
  %12 = arith.addi %4, %11 : index
  %13 = arith.addi %12, %c95232 : index
  memref.store %10, %2[%13] : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>
  return
}

// Check that i64 emulation handles 1-D vector ops and does not introduce
// 2-D vectors.
//
// CHECK-LABEL: func.func @emulate_1d_vector
//       CHECK:   [[LOAD:%.+]]     = memref.load {{%.+}}[{{%.+}}] : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   {{%.+}}, {{%.+}} = arith.mului_extended {{%.+}}, {{%.+}} : vector<4xi32>
//       CHECK:   {{%.+}}, {{%.+}} = arith.addui_extended {{%.+}}, {{%.+}} : vector<4xi32>, vector<4xi1>
//       CHECK:   memref.store {{%.+}}, {{%.+}}[{{%.+}}] : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int64|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @no_emulation() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<8xi32, #spirv.storage_class<StorageBuffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  %3 = memref.load %0[%c0] : memref<8xi32, #spirv.storage_class<StorageBuffer>>
  %4 = memref.load %1[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  %5 = arith.addi %4, %c1_i64 : i64
  memref.store %5, %2[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
  return
}

// Check that with the Int64 capability we do not emulate i64 ops.
//
// CHECK-LABEL: func.func @no_emulation
//       CHECK:   [[CST1:%.+]]      = arith.constant 1 : i64
//       CHECK:   [[REF_I32:%.+]]   = hal.interface.binding.subspan layout({{.+}}) binding(0) : memref<8xi32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_0:%.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_1:%.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI32:%.+]]      = memref.load [[REF_I32]][{{%.+}}] : memref<8xi32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI64:%.+]]      = memref.load [[REF_I64_0]][{{%.+}}] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   {{%.+}}           = arith.addi {{%.+}} : i64
//       CHECK:   memref.store {{%.+}}, [[REF_I64_1]][{{%.+}}] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   return
