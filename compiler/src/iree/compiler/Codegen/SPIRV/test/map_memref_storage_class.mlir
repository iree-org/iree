// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-map-memref-storage-class)))))' --allow-unregistered-dialect %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @vulkan_client_api {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @vulkan_client_api layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @vulkan_client_api() {
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
    }
  }
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @opencl_client_api {
  hal.executable.variant @opencl target(<"opencl-spirv", "opencl-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @opencl_client_api layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @opencl_client_api() {
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
    }
  }
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
