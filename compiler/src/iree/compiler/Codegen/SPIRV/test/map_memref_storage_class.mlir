// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-spirv-map-memref-storage-class))' --allow-unregistered-dialect %s | FileCheck %s

#target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<
    arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
      compute = fp32|int32, storage = b32, subgroup = shuffle|arithmetic,
      dot = none, mma = [], subgroup_size_choices = [64],
      max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
      max_workgroup_memory_bytes = 16384>>}>

func.func @vulkan_client_api() attributes {hal.executable.target = #target} {
  %0 = "dialect.memref_producer"() : () -> (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>)
  "dialect.memref_consumer"(%0) : (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>) -> ()

  %1 = "dialect.memref_producer"() : () -> (memref<?x8xf32, #hal.descriptor_type<storage_buffer>>)
  "dialect.memref_consumer"(%1) : (memref<?x8xf32, #hal.descriptor_type<storage_buffer>>) -> ()

  %2 = "dialect.memref_producer"() : () -> (memref<?x8xf32>)
  "dialect.memref_consumer"(%2) : (memref<?x8xf32>) -> ()

  %3 = "dialect.memref_producer"() : () -> (memref<?x8xf32, 3>)
  "dialect.memref_consumer"(%3) : (memref<?x8xf32, 3>) -> ()

  return
}

// CHECK-LABEL: func.func @vulkan_client_api()
//       CHECK:   %[[P0:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<Uniform>>
//       CHECK:   "dialect.memref_consumer"(%[[P0]]) : (memref<?x8xf32, #spirv.storage_class<Uniform>>) -> ()

//       CHECK:   %[[P1:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   "dialect.memref_consumer"(%[[P1]]) : (memref<?x8xf32, #spirv.storage_class<StorageBuffer>>) -> ()

//       CHECK:   %[[P2:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   "dialect.memref_consumer"(%[[P2]]) : (memref<?x8xf32, #spirv.storage_class<StorageBuffer>>) -> ()

//       CHECK:   %[[P3:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<Workgroup>>
//       CHECK:   "dialect.memref_consumer"(%[[P3]]) : (memref<?x8xf32, #spirv.storage_class<Workgroup>>) -> ()

// -----

#target = #hal.executable.target<"opencl-spirv", "opencl-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<
    arch = "", features = "spirv:v1.3,cap:Kernel", wgp = <
      compute = fp32|int32, storage = b32, subgroup = shuffle|arithmetic,
      dot = none, mma = [], subgroup_size_choices = [64],
      max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
      max_workgroup_memory_bytes = 16384>>}>

func.func @opencl_client_api() attributes {hal.executable.target = #target} {
  %0 = "dialect.memref_producer"() : () -> (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>)
  "dialect.memref_consumer"(%0) : (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>) -> ()

  %1 = "dialect.memref_producer"() : () -> (memref<?x8xf32, #hal.descriptor_type<storage_buffer>>)
  "dialect.memref_consumer"(%1) : (memref<?x8xf32, #hal.descriptor_type<storage_buffer>>) -> ()

  %2 = "dialect.memref_producer"() : () -> (memref<?x8xf32>)
  "dialect.memref_consumer"(%2) : (memref<?x8xf32>) -> ()

  %3 = "dialect.memref_producer"() : () -> (memref<?x8xf32, 3>)
  "dialect.memref_consumer"(%3) : (memref<?x8xf32, 3>) -> ()

  return
}

// CHECK-LABEL: func.func @opencl_client_api()
//       CHECK:   %[[P0:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<Uniform>>
//       CHECK:   "dialect.memref_consumer"(%[[P0]]) : (memref<?x8xf32, #spirv.storage_class<Uniform>>) -> ()

//       CHECK:   %[[P1:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:   "dialect.memref_consumer"(%[[P1]]) : (memref<?x8xf32, #spirv.storage_class<CrossWorkgroup>>) -> ()

//       CHECK:   %[[P2:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:   "dialect.memref_consumer"(%[[P2]]) : (memref<?x8xf32, #spirv.storage_class<CrossWorkgroup>>) -> ()

//       CHECK:   %[[P3:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, #spirv.storage_class<Workgroup>>
//       CHECK:   "dialect.memref_consumer"(%[[P3]]) : (memref<?x8xf32, #spirv.storage_class<Workgroup>>) -> ()
