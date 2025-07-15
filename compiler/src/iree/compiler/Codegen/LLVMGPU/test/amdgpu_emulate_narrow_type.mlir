// RUN: iree-opt --iree-amdgpu-emulate-narrow-type --split-input-file %s | FileCheck %s

func.func @memref_memory_space_cast_i4(%arg0: memref<32x128xi4>) -> memref<32x128xi4, #amdgpu.address_space<fat_raw_buffer>> {
  %cast = amdgpu.fat_raw_buffer_cast %arg0 resetOffset : memref<32x128xi4> to memref<32x128xi4, #amdgpu.address_space<fat_raw_buffer>>
  return %cast : memref<32x128xi4, #amdgpu.address_space<fat_raw_buffer>>
}

// CHECK-LABEL:   func.func @memref_memory_space_cast_i4(
//  CHECK-SAME:   %[[ARG0:.*]]: memref<2048xi8>
//       CHECK:     %[[CAST:.*]] = amdgpu.fat_raw_buffer_cast %[[ARG0]] resetOffset
//  CHECK-SAME:       : memref<2048xi8> to memref<2048xi8, #amdgpu.address_space<fat_raw_buffer>>
//       CHECK:     return %[[CAST]]
