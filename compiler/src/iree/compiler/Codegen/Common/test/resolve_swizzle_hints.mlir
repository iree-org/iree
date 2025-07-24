// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-resolve-swizzle-hints, canonicalize, cse))" \
// RUN:   --split-input-file --mlir-print-local-scope %s | FileCheck %s

func.func @swizzle_load(%src: memref<?xf32>) -> vector<4xf32> {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>

  // 68 = (1 x 64, 4) -> (1, 8) = 72
  %offset = arith.constant 68 : index
  %1 = vector.load %0[%offset] : memref<?xf32>, vector<4xf32>
  return %1: vector<4xf32>
}

// CHECK-LABEL: func @swizzle_load
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>
//       CHECK:   %[[SWOFF:.+]] = arith.constant 72 : index
//       CHECK:   %[[VECTOR:.+]] = vector.load %[[SRC]][%[[SWOFF]]]
//       CHECK:   return %[[VECTOR]]

// -----

func.func @swizzle_store(%dst: memref<?xf32>, %src: vector<4xf32>) {
  %0 = iree_codegen.swizzle_hint %dst[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>

  // 124 = (1 x 64, 60) -> (1, 64 % 64) = 64
  %offset = arith.constant 124 : index
  vector.store %src, %0[%offset] : memref<?xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func @swizzle_store
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: memref<?xf32>
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: vector<4xf32>
//       CHECK:   %[[SWOFF:.+]] = arith.constant 64 : index
//       CHECK:   vector.store %[[SRC]], %[[DST]][%[[SWOFF]]]

// -----

func.func @swizzle_both(%src: memref<?xf32>) {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>
  %c4 = arith.constant 4 : index
  %c44 = arith.constant 44 : index
  %c444 = arith.constant 444 : index
  %c4444 = arith.constant 4444 : index
  %1 = vector.load %0[%c4] : memref<?xf32>, vector<4xf32>
  %2 = vector.load %0[%c44] : memref<?xf32>, vector<4xf32>
  %3 = vector.load %0[%c444] : memref<?xf32>, vector<4xf32>
  %4 = vector.load %0[%c4444] : memref<?xf32>, vector<4xf32>
  vector.store %4, %0[%c4] : memref<?xf32>, vector<4xf32>
  vector.store %3, %0[%c44] : memref<?xf32>, vector<4xf32>
  vector.store %2, %0[%c444] : memref<?xf32>, vector<4xf32>
  vector.store %1, %0[%c4444] : memref<?xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func @swizzle_both
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>
//   CHECK-DAG:   %[[O0:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[O1:.+]] = arith.constant 44 : index
//   CHECK-DAG:   %[[O2:.+]] = arith.constant 404 : index
//   CHECK-DAG:   %[[O3:.+]] = arith.constant 4464 : index
//       CHECK:   %[[V0:.+]] = vector.load %[[SRC]][%[[O0]]]
//       CHECK:   %[[V1:.+]] = vector.load %[[SRC]][%[[O1]]]
//       CHECK:   %[[V2:.+]] = vector.load %[[SRC]][%[[O2]]]
//       CHECK:   %[[V3:.+]] = vector.load %[[SRC]][%[[O3]]]
//       CHECK:   vector.store %[[V3]], %[[SRC]][%[[O0]]]
//       CHECK:   vector.store %[[V2]], %[[SRC]][%[[O1]]]
//       CHECK:   vector.store %[[V1]], %[[SRC]][%[[O2]]]
//       CHECK:   vector.store %[[V0]], %[[SRC]][%[[O3]]]

// -----

func.func @drop_swizzle_non_access_user(%src: memref<?xf32>) -> (memref<?xf32>, vector<4xf32>) {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>
  %offset = arith.constant 68 : index
  %1 = vector.load %0[%offset] : memref<?xf32>, vector<4xf32>
  return %0, %1: memref<?xf32>, vector<4xf32>
}

// CHECK-LABEL: func @drop_swizzle_non_access_user
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>

// Make sure the offset remains the same
//       CHECK:   %[[SWOFF:.+]] = arith.constant 68 : index
//       CHECK:   %[[VECTOR:.+]] = vector.load %[[SRC]][%[[SWOFF]]]
//       CHECK:   return %[[SRC]], %[[VECTOR]]

// -----

func.func @swizzle_unroll_load(%src: memref<?xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>
  %offset = arith.constant 60 : index
  %1 = vector.load %0[%offset] : memref<?xf32>, vector<8xf32>
  %2 = vector.extract_strided_slice %1 {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
  %3 = vector.extract_strided_slice %1 {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
  return %2, %3 : vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL: func @swizzle_unroll_load
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>
//   CHECK-DAG:   %[[SWOFF0:.+]] = arith.constant 60 : index
//   CHECK-DAG:   %[[SWOFF1:.+]] = arith.constant 68 : index
//   CHECK-DAG:   %[[V0:.+]] = vector.load %[[SRC]][%[[SWOFF0]]]
//   CHECK-DAG:   %[[V1:.+]] = vector.load %[[SRC]][%[[SWOFF1]]]
//       CHECK:   return %[[V0]], %[[V1]]

// -----

func.func @swizzle_unroll_store(%dst: memref<?xf32>, %src0: vector<4xf32>, %src1: vector<4xf32>) {
  %0 = iree_codegen.swizzle_hint %dst[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>
  %offset = arith.constant 60 : index
  %cst = arith.constant dense<0.0> : vector<8xf32>
  %1 = vector.insert_strided_slice %src0, %cst {offsets = [0], strides = [1]} : vector<4xf32> into vector<8xf32>
  %2 = vector.insert_strided_slice %src1, %1 {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
  vector.store %2, %0[%offset] : memref<?xf32>, vector<8xf32>
  return
}

// CHECK-LABEL: func @swizzle_unroll_store
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: memref<?xf32>
//  CHECK-SAME:   %[[SRC0:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK-SAME:   %[[SRC1:[A-Za-z0-9]+]]: vector<4xf32>
//   CHECK-DAG:   %[[SWOFF0:.+]] = arith.constant 60 : index
//   CHECK-DAG:   %[[SWOFF1:.+]] = arith.constant 68 : index
//   CHECK-DAG:   vector.store %[[SRC0]], %[[DST]][%[[SWOFF0]]]
//   CHECK-DAG:   vector.store %[[SRC1]], %[[DST]][%[[SWOFF1]]]

// -----

func.func @swizzle_dynamic(%src: memref<?xf32>, %vec: vector<4xf32>, %offset: index) -> vector<4xf32> {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>
  %1 = vector.load %0[%offset] : memref<?xf32>, vector<4xf32>
  vector.store %vec, %0[%offset] : memref<?xf32>, vector<4xf32>
  return %1: vector<4xf32>
}

// CHECK-LABEL: func @swizzle_dynamic
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>
//  CHECK-SAME:   %[[VEC:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK-SAME:   %[[OFFSET:[A-Za-z0-9]+]]: index
//   CHECK-DAG:   %[[ROW_WIDTH:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[GROUP_COUNT:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[GROUP_WIDTH:.+]] = arith.constant 4 : index
//       CHECK:   %[[I:.+]] = arith.divui %[[OFFSET]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[JELEM:.+]] = arith.remui %[[OFFSET]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[J:.+]] = arith.divui %[[JELEM]], %[[GROUP_WIDTH]] : index
//       CHECK:   %[[ADD:.+]] = arith.addi %[[I]], %[[J]] : index
//       CHECK:   %[[ROTATEJ:.+]] = arith.remui %[[ADD]], %[[GROUP_COUNT]] : index
//       CHECK:   %[[ROTATEJELEM:.+]] = arith.muli %[[ROTATEJ]], %[[GROUP_WIDTH]] : index
//       CHECK:   %[[IELEM:.+]] = arith.muli %[[I]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[SWOFF:.+]] = arith.addi %[[ROTATEJELEM]], %[[IELEM]] : index

// Make sure both the load and store get the same calculation.
//       CHECK:   %[[VECTOR:.+]] = vector.load %[[SRC]][%[[SWOFF]]]
//       CHECK:   vector.store %[[VEC]], %[[SRC]][%[[SWOFF]]]
//       CHECK:   return %[[VECTOR]]

// -----

func.func @swizzle_adjust_add_offset(%src: memref<?xf32>, %vec: vector<4xf32>, %offset_base: index) -> vector<4xf32> {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>
  %c16 = arith.constant 16 : index
  %c1040 = arith.constant 1040 : index
  %load_offset = arith.addi %offset_base, %c16 overflow<nsw> : index
  %1 = vector.load %0[%load_offset] : memref<?xf32>, vector<4xf32>
  %store_offset = arith.addi %offset_base, %c1040 overflow<nsw> : index
  vector.store %vec, %0[%store_offset] : memref<?xf32>, vector<4xf32>
  return %1: vector<4xf32>
}

// CHECK-LABEL: func @swizzle_adjust_add_offset
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>
//  CHECK-SAME:   %[[VEC:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK-SAME:   %[[OFFSET:[A-Za-z0-9]+]]: index
//   CHECK-DAG:   %[[ROW_WIDTH:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[GROUP_COUNT:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[GROUP_WIDTH:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C1040:.+]] = arith.constant 1040 : index
//       CHECK:   %[[APPLY_BASE:.+]] = arith.addi %[[OFFSET]], %[[GROUP_COUNT]] overflow<nsw> : index
//       CHECK:   %[[I:.+]] = arith.divui %[[APPLY_BASE]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[JELEM:.+]] = arith.remui %[[APPLY_BASE]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[J:.+]] = arith.divui %[[JELEM]], %[[GROUP_WIDTH]] : index
//       CHECK:   %[[ADD:.+]] = arith.addi %[[I]], %[[J]] : index
//       CHECK:   %[[ROTATEJ:.+]] = arith.remui %[[ADD]], %[[GROUP_COUNT]] : index
//       CHECK:   %[[ROTATEJELEM:.+]] = arith.muli %[[ROTATEJ]], %[[GROUP_WIDTH]] : index
//       CHECK:   %[[IELEM:.+]] = arith.muli %[[I]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[SWOFF:.+]] = arith.addi %[[ROTATEJELEM]], %[[IELEM]] : index

//       CHECK:   %[[VECTOR:.+]] = vector.load %[[SRC]][%[[SWOFF]]]

//       CHECK:   %[[STORE_BASE:.+]] = arith.addi %[[OFFSET]], %[[C1040]] overflow<nsw> : index
//       CHECK:   %[[OFFSET_DIFF:.+]] = arith.subi %[[SWOFF]], %[[APPLY_BASE]] : index
//       CHECK:   %[[STORE_SWOFF:.+]] = arith.addi %[[STORE_BASE]], %[[OFFSET_DIFF]] : index
//       CHECK:   vector.store %[[VEC]], %[[SRC]][%[[STORE_SWOFF]]]
//       CHECK:   return %[[VECTOR]]

// -----

func.func @swizzle_gather_to_lds(%src: memref<?xf32>, %offset: index) {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 4>] : memref<?xf32>
  %lds = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %0[%offset], %lds[%c0] : vector<4xf32>, memref<?xf32>, memref<256xf32, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func @swizzle_gather_to_lds
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>
//  CHECK-SAME:   %[[OFFSET:[A-Za-z0-9]+]]: index
//   CHECK-DAG:   %[[ROW_WIDTH:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[GROUP_COUNT:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[GROUP_WIDTH:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[DSTOFFSET:.+]] = arith.constant 0 : index
//       CHECK:   %[[LDS:.+]] = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[I:.+]] = arith.divui %[[OFFSET]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[JELEM:.+]] = arith.remui %[[OFFSET]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[J:.+]] = arith.divui %[[JELEM]], %[[GROUP_WIDTH]] : index
//       CHECK:   %[[ADD:.+]] = arith.addi %[[I]], %[[J]] : index
//       CHECK:   %[[ROTATEJ:.+]] = arith.remui %[[ADD]], %[[GROUP_COUNT]] : index
//       CHECK:  %[[ROTATEJELEM:.+]] = arith.muli %[[ROTATEJ]], %[[GROUP_WIDTH]] : index
//       CHECK:   %[[IELEM:.+]] = arith.muli %[[I]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[SWOFF:.+]] = arith.addi %[[ROTATEJELEM]], %[[IELEM]] : index
//       CHECK:   amdgpu.gather_to_lds %[[SRC]][%[[SWOFF]]], %[[LDS]][%[[DSTOFFSET]]]

// -----
func.func @swizzle_gather_to_lds_scalar(%src: memref<?xf32>, %offset: index) {
  %0 = iree_codegen.swizzle_hint %src[#iree_codegen.rotate_rows<64, 1>] : memref<?xf32>
  %lds = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %0[%offset], %lds[%c0] : f32, memref<?xf32>, memref<256xf32, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func @swizzle_gather_to_lds_scalar
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?xf32>
//  CHECK-SAME:   %[[OFFSET:[A-Za-z0-9]+]]: index
//   CHECK-DAG:   %[[ROW_WIDTH:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[DSTOFFSET:.+]] = arith.constant 0 : index
//       CHECK:   %[[LDS:.+]] = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[I:.+]] = arith.divui %[[OFFSET]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[JELEM:.+]] = arith.remui %[[OFFSET]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[J:.+]] = arith.addi %[[I]], %[[JELEM]] : index
//       CHECK:   %[[ROTATEJ:.+]] = arith.remui %[[J]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[IELEM:.+]] = arith.muli %[[I]], %[[ROW_WIDTH]] : index
//       CHECK:   %[[SWOFF:.+]] = arith.addi %[[ROTATEJ]], %[[IELEM]] : index
//       CHECK:   amdgpu.gather_to_lds %[[SRC]][%[[SWOFF]]], %[[LDS]][%[[DSTOFFSET]]]
