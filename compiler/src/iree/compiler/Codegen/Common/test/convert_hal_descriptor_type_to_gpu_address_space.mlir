// RUN: iree-opt -iree-codegen-convert-hal-descriptor-type-to-gpu-address-space -split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK-LABEL: func.func @to_global()
// CHECK: hal.interface.binding.subspan
// CHECK-SAME: #gpu.address_space<global>
// CHECK-NEXT: memref.store
// CHECK-SAME: #gpu.address_space<global>
func.func @to_global() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %mem = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c256)
    : memref<256xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  memref.store %c0_i32, %mem[%c0] : memref<256xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: func.func @to_amdgpu_buffer()
// CHECK: hal.interface.binding.subspan
// CHECK-SAME: #gpu.address_space<global>
// CHECK-NEXT: amdgpu.fat_raw_buffer_cast
// CHECK-SAME: resetOffset
// CHECK-NEXT: memref.cast
// CHECK-SAME: memref<256xi32, strided<[1]>, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: to memref<256xi32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-NEXT: memref.subview
// CHECK-NEXT: memref.store
// CHECK-SAME: #amdgpu.address_space<fat_raw_buffer>
func.func @to_amdgpu_buffer() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %tid = gpu.thread_id x
  %mem = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c256)
    : memref<256xi32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
  %part = memref.subview %mem[%tid] [4] [1]
    : memref<256xi32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
      to memref<4xi32, strided<[1], offset: ?>,  #amdgpu.address_space<fat_raw_buffer>>
  memref.store %c0_i32, %part[%c0] : memref<4xi32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
  return
}
