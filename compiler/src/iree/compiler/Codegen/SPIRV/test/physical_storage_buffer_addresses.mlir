// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-spirv{index-bits=64}))))' \
// RUN:   %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ], flags = Indirect>
]>
hal.executable private @interface_binding {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb-ptr", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Int64, Shader, PhysicalStorageBufferAddresses],
                                                            [SPV_KHR_physical_storage_buffer]>, #spirv.resource_limits<>>,
      hal.bindings.indirect}>) {
    hal.executable.export @interface_binding layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @interface_binding() -> f32 {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8x5xf32, #spirv.storage_class<PhysicalStorageBuffer>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<5xf32, #spirv.storage_class<PhysicalStorageBuffer>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x5xf32, #spirv.storage_class<PhysicalStorageBuffer>>

        %3 = memref.load %0[%c0, %c0] : memref<8x5xf32, #spirv.storage_class<PhysicalStorageBuffer>>
        %4 = memref.load %1[%c0] : memref<5xf32, #spirv.storage_class<PhysicalStorageBuffer>>
        %5 = memref.load %2[%c0, %c0] : memref<4x5xf32, #spirv.storage_class<PhysicalStorageBuffer>>

        %6 = arith.addf %3, %4 : f32
        %8 = arith.addf %6, %5 : f32

        return %8 : f32
      }
    }
  }
}

// Explicitly check address calculations.
// Many of these are redundant and get optimized out after `--cse`.

// CHECK-LABEL: spirv.module PhysicalStorageBuffer64
//       CHECK:   spirv.GlobalVariable [[GLOBAL:@.+]] bind(3, 0) :
//  CHECK-SAME:     !spirv.ptr<!spirv.struct<(!spirv.ptr<i32, PhysicalStorageBuffer> [0], !spirv.ptr<i32, PhysicalStorageBuffer> [8], !spirv.ptr<i32, PhysicalStorageBuffer> [16])>, StorageBuffer>
//       CHECK:   spirv.func
//       CHECK:   %[[addr0:.+]] = spirv.mlir.addressof [[GLOBAL]]
//  CHECK-NEXT:   %[[cst0:.+]] = spirv.Constant 0 : i32
//  CHECK-NEXT:   %[[s0b0:.+]] = spirv.AccessChain %[[addr0]][%[[cst0]]]
//  CHECK-NEXT:   %[[ld0:.+]]  = spirv.Load "StorageBuffer" %[[s0b0]]
//  CHECK-NEXT:   %[[int0:.+]] = spirv.ConvertPtrToU %[[ld0]] : !spirv.ptr<i32, PhysicalStorageBuffer> to i64
//  CHECK-NEXT:   %[[ptr0:.+]] = spirv.ConvertUToPtr %[[int0]] : i64 to !spirv.ptr<!spirv.struct<(!spirv.array<40 x f32, stride=4> [0])>, PhysicalStorageBuffer>
//  CHECK-NEXT:   %[[addr1:.+]] = spirv.mlir.addressof [[GLOBAL]]
//  CHECK-NEXT:   %[[cst1:.+]] = spirv.Constant 1 : i32
//  CHECK-NEXT:   %[[s0b1:.+]] = spirv.AccessChain %[[addr1]][%[[cst1]]]
//  CHECK-NEXT:   %[[ld1:.+]]  = spirv.Load "StorageBuffer" %[[s0b1]]
//  CHECK-NEXT:   %[[int1:.+]] = spirv.ConvertPtrToU %[[ld1]] : !spirv.ptr<i32, PhysicalStorageBuffer> to i64
//  CHECK-NEXT:   %[[ptr1:.+]] = spirv.ConvertUToPtr %[[int1]] : i64 to !spirv.ptr<!spirv.struct<(!spirv.array<5 x f32, stride=4> [0])>, PhysicalStorageBuffer>
//       CHECK:   %[[ptr2:.+]] = spirv.ConvertUToPtr %{{.+}} : i64 to !spirv.ptr<!spirv.struct<(!spirv.array<20 x f32, stride=4> [0])>, PhysicalStorageBuffer>
//
//       CHECK:   %[[loc0:.+]] = spirv.AccessChain %[[ptr0]]
//  CHECK-NEXT:   spirv.Load "PhysicalStorageBuffer" %[[loc0]]
//       CHECK:   %[[loc1:.+]] = spirv.AccessChain %[[ptr1]]
//  CHECK-NEXT:   spirv.Load "PhysicalStorageBuffer" %[[loc1]]
//       CHECK:   %[[loc2:.+]] = spirv.AccessChain %[[ptr2]]
//  CHECK-NEXT:   spirv.Load "PhysicalStorageBuffer" %[[loc2]]


