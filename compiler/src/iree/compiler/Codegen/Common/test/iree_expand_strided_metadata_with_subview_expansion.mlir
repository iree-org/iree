// RUN: iree-opt %s --iree-codegen-expand-strided-metadata="allow-subview-expansion=true" --split-input-file | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @expand_subview_into_resource_cast(%offset: index) -> memref<256x4096xf16, #amdgpu.address_space<fat_raw_buffer>> {
  %c4096_i14 = arith.constant 4096 : i14
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%1) flags(ReadOnly)
    : memref<4096x4096xf16, #gpu.address_space<global>>
  %subview = memref.subview %2[%offset, 0] [256, 4096] [1, 1]
    : memref<4096x4096xf16, #gpu.address_space<global>> to memref<256x4096xf16, strided<[4096, 1], offset: ?>, #gpu.address_space<global>>
  %38 = amdgpu.fat_raw_buffer_cast %subview cacheSwizzleStride(%c4096_i14) resetOffset
    : memref<256x4096xf16, strided<[4096, 1], offset: ?>, #gpu.address_space<global>> to memref<256x4096xf16, #amdgpu.address_space<fat_raw_buffer>>
  return %38 : memref<256x4096xf16, #amdgpu.address_space<fat_raw_buffer>>
}

// CHECK-LABEL: func.func @expand_subview_into_resource_cast
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[REINTERPRET:.+]] = memref.reinterpret_cast %[[SUBSPAN]]
//       CHECK:   amdgpu.fat_raw_buffer_cast %[[REINTERPRET]]
