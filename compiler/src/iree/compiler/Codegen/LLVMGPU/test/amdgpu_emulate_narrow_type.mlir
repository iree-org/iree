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

// -----

// Test that gather_to_lds with sub-byte element types (i4) gets converted
// to use byte-sized elements (i8), with indices packed and transfer type
// adjusted to preserve the same number of transferred bits.
func.func @gather_to_lds_i4_2d(
    %src: memref<128x32xi4, #amdgpu.address_space<fat_raw_buffer>>,
    %dst: memref<64xi4, #gpu.address_space<workgroup>>,
    %i0: index, %i1: index, %j0: index) {
  amdgpu.gather_to_lds %src[%i0, %i1], %dst[%j0]
      : vector<8xi4>,
        memref<128x32xi4, #amdgpu.address_space<fat_raw_buffer>>,
        memref<64xi4, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func.func @gather_to_lds_i4_2d(
//  CHECK-SAME:     %[[SRC:.*]]: memref<2048xi8, #amdgpu.address_space<fat_raw_buffer>>
//  CHECK-SAME:     %[[DST:.*]]: memref<32xi8, #gpu.address_space<workgroup>>
//  CHECK-SAME:     %[[I0:.*]]: index, %[[I1:.*]]: index, %[[J0:.*]]: index
//       CHECK:     amdgpu.gather_to_lds %[[SRC]][{{.*}}], %[[DST]][{{.*}}]
//  CHECK-SAME:       : vector<4xi8>

// -----

// Test gather_to_lds with async attribute is preserved after conversion.
func.func @gather_to_lds_i4_async(
    %src: memref<256xi4, #amdgpu.address_space<fat_raw_buffer>>,
    %dst: memref<64xi4, #gpu.address_space<workgroup>>,
    %idx: index, %jdx: index) {
  amdgpu.gather_to_lds async %src[%idx], %dst[%jdx]
      : vector<8xi4>,
        memref<256xi4, #amdgpu.address_space<fat_raw_buffer>>,
        memref<64xi4, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func.func @gather_to_lds_i4_async(
//  CHECK-SAME:     %[[SRC:.*]]: memref<128xi8, #amdgpu.address_space<fat_raw_buffer>>
//  CHECK-SAME:     %[[DST:.*]]: memref<32xi8, #gpu.address_space<workgroup>>
//       CHECK:     amdgpu.gather_to_lds async
//  CHECK-SAME:       : vector<4xi8>

// -----

// Test gather_to_lds with f4E2M1FN sub-byte type gets converted
// (vector<8xf4E2M1FN> = 32 bits -> vector<4xi8>).
func.func @gather_to_lds_f4E2M1FN_2d(
    %src: memref<128x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>,
    %dst: memref<64xf4E2M1FN, #gpu.address_space<workgroup>>,
    %i0: index, %i1: index, %j0: index) {
  amdgpu.gather_to_lds %src[%i0, %i1], %dst[%j0]
      : vector<8xf4E2M1FN>,
        memref<128x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>,
        memref<64xf4E2M1FN, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func.func @gather_to_lds_f4E2M1FN_2d(
//  CHECK-SAME:     %[[SRC:.*]]: memref<2048xi8, #amdgpu.address_space<fat_raw_buffer>>
//  CHECK-SAME:     %[[DST:.*]]: memref<32xi8, #gpu.address_space<workgroup>>
//  CHECK-SAME:     %[[I0:.*]]: index, %[[I1:.*]]: index, %[[J0:.*]]: index
//       CHECK:     amdgpu.gather_to_lds %[[SRC]][{{.*}}], %[[DST]][{{.*}}]
//  CHECK-SAME:       : vector<4xi8>

// -----

// Test gather_to_lds with i2 sub-byte type gets converted (vector<16xi2> =
// 32 bits -> vector<4xi8>).
func.func @gather_to_lds_i2_2d(
    %src: memref<128x64xi2, #amdgpu.address_space<fat_raw_buffer>>,
    %dst: memref<128xi2, #gpu.address_space<workgroup>>,
    %i0: index, %i1: index, %j0: index) {
  amdgpu.gather_to_lds %src[%i0, %i1], %dst[%j0]
      : vector<16xi2>,
        memref<128x64xi2, #amdgpu.address_space<fat_raw_buffer>>,
        memref<128xi2, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func.func @gather_to_lds_i2_2d(
//  CHECK-SAME:     %[[SRC:.*]]: memref<2048xi8, #amdgpu.address_space<fat_raw_buffer>>
//  CHECK-SAME:     %[[DST:.*]]: memref<32xi8, #gpu.address_space<workgroup>>
//       CHECK:     amdgpu.gather_to_lds %[[SRC]][{{.*}}], %[[DST]][{{.*}}]
//  CHECK-SAME:       : vector<4xi8>
