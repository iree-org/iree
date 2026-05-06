// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-spirv{index-bits=64}))))' \
// RUN:   %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @interface_binding {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-bda-v1">) {
    hal.executable.export public @interface_binding layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module attributes {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.5,
        [Int64, Shader, PhysicalStorageBufferAddresses],
        [SPV_KHR_physical_storage_buffer]>, #spirv.resource_limits<>>
    } {
      func.func @interface_binding() -> f32 {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<8x5xf32, #spirv.storage_class<PhysicalStorageBuffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<5xf32, #spirv.storage_class<PhysicalStorageBuffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<4x5xf32, #spirv.storage_class<PhysicalStorageBuffer>>

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
//   CHECK-NOT:   bind(3, 0)
//       CHECK:   spirv.GlobalVariable [[ROOT:@.+]] : !spirv.ptr<!spirv.struct<(!spirv.array<8 x i32, stride=4> [0])>, PushConstant>
//       CHECK:   spirv.func
//       CHECK:   %[[root_addr0:.+]] = spirv.mlir.addressof [[ROOT]]
//       CHECK:   %[[low_addr0:.+]] = spirv.AccessChain %[[root_addr0]][%{{.+}}, %{{.+}}]
//       CHECK:   %[[low32_0:.+]] = spirv.Load "PushConstant" %[[low_addr0]]
//       CHECK:   %[[root_addr1:.+]] = spirv.mlir.addressof [[ROOT]]
//       CHECK:   %[[high_addr0:.+]] = spirv.AccessChain %[[root_addr1]][%{{.+}}, %{{.+}}]
//       CHECK:   %[[high32_0:.+]] = spirv.Load "PushConstant" %[[high_addr0]]
//       CHECK:   %[[low64_0:.+]] = spirv.UConvert %[[low32_0]] : i32 to i64
//       CHECK:   %[[high64_0:.+]] = spirv.UConvert %[[high32_0]] : i32 to i64
//       CHECK:   %[[shifted0:.+]] = spirv.ShiftLeftLogical %[[high64_0]], %{{.+}} : i64, i64
//       CHECK:   %[[table_addr0:.+]] = spirv.BitwiseOr %[[low64_0]], %[[shifted0]] : i64
//       CHECK:   %[[table0:.+]] = spirv.ConvertUToPtr %[[table_addr0]] : i64 to !spirv.ptr<!spirv.struct<(!spirv.ptr<i32, PhysicalStorageBuffer> [0])>, PhysicalStorageBuffer>
//       CHECK:   %[[slot0:.+]] = spirv.AccessChain %[[table0]][%{{.+}}]
//       CHECK:   %[[ld0:.+]] = spirv.Load "PhysicalStorageBuffer" %[[slot0]]
//       CHECK:   %[[int0:.+]] = spirv.ConvertPtrToU %[[ld0]] : !spirv.ptr<i32, PhysicalStorageBuffer> to i64
//       CHECK:   %[[ptr0:.+]] = spirv.ConvertUToPtr %[[int0]] : i64 to !spirv.ptr<!spirv.struct<(!spirv.array<40 x f32, stride=4> [0])>, PhysicalStorageBuffer>
//       CHECK:   %[[table1:.+]] = spirv.ConvertUToPtr %{{.+}} : i64 to !spirv.ptr<!spirv.struct<(!spirv.ptr<i32, PhysicalStorageBuffer> [0], !spirv.ptr<i32, PhysicalStorageBuffer> [8])>, PhysicalStorageBuffer>
//       CHECK:   %[[slot1:.+]] = spirv.AccessChain %[[table1]][%{{.+}}]
//       CHECK:   %[[ld1:.+]] = spirv.Load "PhysicalStorageBuffer" %[[slot1]]
//       CHECK:   %[[int1:.+]] = spirv.ConvertPtrToU %[[ld1]] : !spirv.ptr<i32, PhysicalStorageBuffer> to i64
//       CHECK:   %[[ptr1:.+]] = spirv.ConvertUToPtr %[[int1]] : i64 to !spirv.ptr<!spirv.struct<(!spirv.array<5 x f32, stride=4> [0])>, PhysicalStorageBuffer>
//       CHECK:   %[[table2:.+]] = spirv.ConvertUToPtr %{{.+}} : i64 to !spirv.ptr<!spirv.struct<(!spirv.ptr<i32, PhysicalStorageBuffer> [0], !spirv.ptr<i32, PhysicalStorageBuffer> [8], !spirv.ptr<i32, PhysicalStorageBuffer> [16])>, PhysicalStorageBuffer>
//       CHECK:   %[[slot2:.+]] = spirv.AccessChain %[[table2]][%{{.+}}]
//       CHECK:   %[[ld2:.+]] = spirv.Load "PhysicalStorageBuffer" %[[slot2]]
//       CHECK:   %[[int2:.+]] = spirv.ConvertPtrToU %[[ld2]] : !spirv.ptr<i32, PhysicalStorageBuffer> to i64
//       CHECK:   %[[ptr2:.+]] = spirv.ConvertUToPtr %[[int2]] : i64 to !spirv.ptr<!spirv.struct<(!spirv.array<20 x f32, stride=4> [0])>, PhysicalStorageBuffer>
//
//       CHECK:   %[[loc0:.+]] = spirv.AccessChain %[[ptr0]]
//  CHECK-NEXT:   spirv.Load "PhysicalStorageBuffer" %[[loc0]]
//       CHECK:   %[[loc1:.+]] = spirv.AccessChain %[[ptr1]]
//  CHECK-NEXT:   spirv.Load "PhysicalStorageBuffer" %[[loc1]]
//       CHECK:   %[[loc2:.+]] = spirv.AccessChain %[[ptr2]]
//  CHECK-NEXT:   spirv.Load "PhysicalStorageBuffer" %[[loc2]]
