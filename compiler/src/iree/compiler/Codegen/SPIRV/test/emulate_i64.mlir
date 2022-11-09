// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-spirv-emulate-i64))))' %s | \
// RUN:   FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @buffer_types {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>}> {
    hal.executable.export @buffer_types layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @buffer_types() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i64
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8xi32, #spirv.storage_class<StorageBuffer>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<8xi64, #spirv.storage_class<StorageBuffer>>

        %3 = memref.load %0[%c0] : memref<8xi32, #spirv.storage_class<StorageBuffer>>
        %4 = memref.load %1[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
        %5 = arith.addi %4, %c1 : i64
        memref.store %5, %2[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>

        return
      }
    }
  }
}

// Check that without the Int64 capability emulation produces expected i32 ops.
//
// CHECK-LABEL: func.func @buffer_types
//       CHECK:   [[CST1:%.+]]      = arith.constant dense<[1, 0]> : vector<2xi32>
//       CHECK:   [[REF_I32:%.+]]   = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8xi32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_0:%.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_1:%.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI32:%.+]]      = memref.load [[REF_I32]][{{%.+}}] : memref<8xi32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI64:%.+]]      = memref.load [[REF_I64_0]][{{%.+}}] : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   {{%.+}}           = arith.addui_carry {{%.+}}, {{%.+}} : i32, i1
//       CHECK:   memref.store {{%.+}}, [[REF_I64_1]][{{%.+}}] : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   return

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @no_emulation {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, Int64], []>, #spirv.resource_limits<>>}> {
    hal.executable.export @no_emulation layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @no_emulation() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i64
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8xi32, #spirv.storage_class<StorageBuffer>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<8xi64, #spirv.storage_class<StorageBuffer>>

        %3 = memref.load %0[%c0] : memref<8xi32, #spirv.storage_class<StorageBuffer>>
        %4 = memref.load %1[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
        %5 = arith.addi %4, %c1 : i64
        memref.store %5, %2[%c0] : memref<8xi64, #spirv.storage_class<StorageBuffer>>

        return
      }
    }
  }
}

// Check that with the Int64 capability we do not emulate i64 ops.
//
// CHECK-LABEL: func.func @no_emulation
//       CHECK:   [[CST1:%.+]]      = arith.constant 1 : i64
//       CHECK:   [[REF_I32:%.+]]   = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8xi32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_0:%.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_1:%.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI32:%.+]]      = memref.load [[REF_I32]][{{%.+}}] : memref<8xi32, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI64:%.+]]      = memref.load [[REF_I64_0]][{{%.+}}] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   {{%.+}}           = arith.addi {{%.+}} : i64
//       CHECK:   memref.store {{%.+}}, [[REF_I64_1]][{{%.+}}] : memref<8xi64, #spirv.storage_class<StorageBuffer>>
//       CHECK:   return
