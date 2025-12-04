// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-propagate-reshapes-by-expansion))" %s | FileCheck %s

#pipeline_layout_bufferized = #hal.pipeline.layout<bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly">]>
func.func @fold_bitcast_into_buffer_load() -> tensor<4x8x128xi8> {
  %c0 = arith.constant 0 : index
  %subspan = hal.interface.binding.subspan layout(#pipeline_layout_bufferized) binding(0) alignment(64) offset(%c0)
      : memref<4x8x256xf4E2M1FN, strided<[2048, 256, 1]>>
  %buffer = amdgpu.fat_raw_buffer_cast %subspan resetOffset
      : memref<4x8x256xf4E2M1FN, strided<[2048, 256, 1]>> to memref<4x8x256xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>
  %tensor = iree_codegen.load_from_buffer %buffer : memref<4x8x256xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x8x256xf4E2M1FN>
  %bitcast = iree_tensor_ext.bitcast %tensor : tensor<4x8x256xf4E2M1FN> -> tensor<4x8x128xi8>
  return %bitcast : tensor<4x8x128xi8>
}
// CHECK-LABEL: func @fold_bitcast_into_buffer_load()
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       memref<4x8x128xi8, strided<[1024, 128, 1]>>
//       CHECK:   %[[BUFFER:.+]] = amdgpu.fat_raw_buffer_cast %[[SUBSPAN]] resetOffset
//  CHECK-SAME:       memref<4x8x128xi8, strided<[1024, 128, 1]>> to memref<4x8x128xi8, #amdgpu.address_space<fat_raw_buffer>>
//       CHECK:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//  CHECK-SAME:       memref<4x8x128xi8, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x8x128xi8>
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   return %[[LOAD]]
