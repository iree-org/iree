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
//       CHECK:   [[REF_I64_0:%.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[REF_I64_1:%.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   [[VI64:%.+]]      = memref.load [[REF_I64_0]][{{%.+}}] : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   {{%.+}}           = arith.addui_extended {{%.+}}, {{%.+}} : i32, i1
//       CHECK:   memref.store {{%.+}}, [[REF_I64_1]][{{%.+}}] : memref<8xvector<2xi32>, #spirv.storage_class<StorageBuffer>>
//       CHECK:   return

// -----

hal.executable private @emulate_1d_vector {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, #spirv.resource_limits<>>}> {
    hal.executable.export public @emulate_1d_vector ordinal(0)
      layout(#hal.pipeline.layout<push_constants = 0,
                                  sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @emulate_1d_vector() {
        %c95232 = arith.constant 95232 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %c36864 = arith.constant 36864 : index
        %c1523712 = arith.constant 1523712 : index
        %c96 = arith.constant 96 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>{%c96}
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1523712) : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>{%c36864}
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>{%c36864}
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %3 = gpu.thread_id  x
        %4 = arith.muli %workgroup_id_x, %c32 : index
        %5 = arith.addi %3, %4 : index
        %6 = memref.load %0[%5] : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>
        %7 = arith.extsi %6 : vector<4xi32> to vector<4xi64>
        %8 = arith.extui %6 : vector<4xi32> to vector<4xi64>
        %9 = arith.muli %7, %8 : vector<4xi64>
        %10 = arith.addi %7, %9 : vector<4xi64>
        %11 = arith.trunci %10 : vector<4xi64> to vector<4xi32>
        %12 = arith.muli %workgroup_id_y, %c96 : index
        %13 = arith.addi %5, %12 : index
        %14 = arith.addi %13, %c95232 : index
        memref.store %11, %2[%14] : memref<?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>
        return
      }
    }
  }
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
