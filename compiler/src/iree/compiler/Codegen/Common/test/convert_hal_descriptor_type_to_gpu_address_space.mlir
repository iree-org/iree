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
