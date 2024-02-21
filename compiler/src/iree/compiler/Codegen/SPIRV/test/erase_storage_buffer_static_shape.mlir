// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-spirv-erase-storage-buffer-static-shape))" %s | FileCheck %s

func.func @storage_buffer_load_store(%offset: index, %i0: index, %i1: index) {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%offset) flags(ReadOnly) : memref<256xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%offset) : memref<256xf32, #hal.descriptor_type<storage_buffer>>
  %val = memref.load %0[%i0] : memref<256xf32, #hal.descriptor_type<storage_buffer>>
  memref.store %val, %1[%i1] : memref<256xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: func.func @storage_buffer_load_store
//  CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index)
//       CHECK:   %[[C256:.+]] = arith.constant 256 : index
//       CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[OFFSET]]) flags(ReadOnly) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%[[C256]]}
//       CHECK:   %[[SPAN1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[OFFSET]]) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%[[C256]]}
//       CHECK:   %[[LD:.+]] = memref.load %[[SPAN0]][%[[I0]]]
//       CHECK:   memref.store %[[LD]], %[[SPAN1]][%[[I1]]]

// -----

// Test that we don't rewrite memref for uniform buffers.

func.func @uniform_buffer_load(%offset: index, %i0: index) -> f32 {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) alignment(64) offset(%offset) flags(ReadOnly) : memref<256xf32, #hal.descriptor_type<uniform_buffer>>
  %val = memref.load %0[%i0] : memref<256xf32, #hal.descriptor_type<uniform_buffer>>
  return %val : f32
}

// CHECK-LABEL: func.func @uniform_buffer_load
//       CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) alignment(64) offset(%{{.+}}) flags(ReadOnly) : memref<256xf32, #hal.descriptor_type<uniform_buffer>>
//       CHECK:   memref.load %[[SPAN0]]

// -----

// Test that we don't rewrite memref without HAL descriptor types.

func.func @uniform_buffer_load(%offset: index, %i0: index) -> f32 {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) alignment(64) offset(%offset) flags(ReadOnly) : memref<256xf32>
  %val = memref.load %0[%i0] : memref<256xf32>
  return %val : f32
}

// CHECK-LABEL: func.func @uniform_buffer_load
//       CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) alignment(64) offset(%{{.+}}) flags(ReadOnly) : memref<256xf32>
//       CHECK:   memref.load %[[SPAN0]]

// -----

func.func @storage_buffer_transfer_read_write(%offset: index, %i0: index, %i1: index) {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%offset) flags(ReadOnly) : memref<256xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%offset) : memref<256xf32, #hal.descriptor_type<storage_buffer>>
  %f0 = arith.constant 0.0 : f32
  %val = vector.transfer_read %0[%i0], %f0 {in_bounds = [true]} : memref<256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
  vector.transfer_write %val, %1[%i1] {in_bounds = [true]} : vector<4xf32>, memref<256xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: func.func @storage_buffer_transfer_read_write(%arg0: index, %arg1: index, %arg2: index) {
//       CHECK:   vector.transfer_read {{.+}} : memref<?xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.+}} : vector<4xf32>, memref<?xf32, #hal.descriptor_type<storage_buffer>>

// -----

func.func @storage_buffer_subview(%offset : index, %i0: index, %i1: index) -> f32 {
  %c0 = arith.constant 0 : index
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<128xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %subspan[%i0][16][1] : memref<128xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %value = memref.load %subview[%c0] : memref<16xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  return %value : f32
}

// CHECK-LABEL: func.func @storage_buffer_subview
//       CHECK:   memref.subview %{{.+}}[%{{.+}}] [16] [1] : memref<?xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>

// -----

func.func @storage_buffer_cast(%offset: index) -> memref<?xf32, #hal.descriptor_type<storage_buffer>> {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%offset) : memref<16xf32, #hal.descriptor_type<storage_buffer>>
  %1 = memref.cast %0 : memref<16xf32, #hal.descriptor_type<storage_buffer>> to memref<?xf32, #hal.descriptor_type<storage_buffer>>
  return %1 : memref<?xf32, #hal.descriptor_type<storage_buffer>>
}

// CHECK-LABEL: func.func @storage_buffer_cast
//       CHECK:   %[[C16:.+]] = arith.constant 16 : index
//       CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%{{.+}}) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%[[C16]]}
//       CHECK:   return %[[SPAN0]]
