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

// -----

// Test combining memref.dim resolution with narrow type emulation and vector ops.
// This tests a previously failing case:
// 1. memref.alloc provides a buffer with dynamic dims
// 2. amdgpu.fat_raw_buffer_cast converts to fat buffer addressing
// 3. memref.dim queries the dimension (must be resolved before emulation)
// 4. vector.load/store operates on the narrow type (must be emulated to i8)
func.func @dim_resolution_with_vector_emulation(%size: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc(%size) : memref<?x128xi4>
  %cast = amdgpu.fat_raw_buffer_cast %alloc resetOffset
      : memref<?x128xi4> to memref<?x128xi4, #amdgpu.address_space<fat_raw_buffer>>
  %dim = memref.dim %cast, %c0 : memref<?x128xi4, #amdgpu.address_space<fat_raw_buffer>>
  // Use the dimension in a loop bound (realistic use case)
  scf.for %i = %c0 to %dim step %c1 {
    // Load narrow type vector - this must be emulated
    %vec = vector.load %cast[%i, %c0] : memref<?x128xi4, #amdgpu.address_space<fat_raw_buffer>>, vector<8xi4>
    vector.store %vec, %cast[%i, %c0] : memref<?x128xi4, #amdgpu.address_space<fat_raw_buffer>>, vector<8xi4>
  }
  return
}

// CHECK-LABEL: func.func @dim_resolution_with_vector_emulation(
//  CHECK-SAME:     %[[SIZE:.*]]: index
// Verify the loop uses the resolved dimension (the function argument)
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %[[SIZE]]
// Verify vector operations are emulated to i8 (8xi4 -> 4xi8)
//       CHECK:     vector.load %{{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
//       CHECK:     vector.store %{{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
