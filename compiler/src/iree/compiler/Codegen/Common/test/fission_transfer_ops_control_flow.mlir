// RUN: iree-opt --split-input-file -pass-pipeline="builtin.module(func.func(iree-codegen-fission-transfer-ops-in-control-flow),cse,canonicalize)" %s | FileCheck %s

// CHECK-LABEL: @fission_global_read_to_private_write
// CHECK-SAME: %[[ARG0:.*]]: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: i1
// CHECK-SAME: %[[ARG3:.*]]: vector<1x1x1x8xbf16>
// CHECK-SAME: %[[ARG4:.*]]: memref<1x1x1x8xbf16, #gpu.address_space<private>>
func.func @fission_global_read_to_private_write(%arg0: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %arg1: index, %arg2: i1, %arg3: vector<1x1x1x8xbf16>, %arg4: memref<1x1x1x8xbf16, #gpu.address_space<private>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : bf16
  scf.for %arg5 = %c0 to %arg1 step %c1 {
    %read = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
    %select = arith.select %arg2, %read, %arg3 : vector<1x1x1x8xbf16>
    vector.transfer_write %select, %arg4[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, #gpu.address_space<private>>
  }
  return
}
// CHECK: %[[ALLOCA:.*]] = memref.alloca(%[[ARG1]])
// CHECK: scf.for %[[ITER:.*]] = %c0 to %[[ARG1]] step %c1 {
// CHECK:   %[[read:.*]] = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]}
// CHECK:   vector.transfer_write %[[read]], %[[ALLOCA]][%[[ITER]], %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]}
// CHECK: }
// CHECK: scf.for %[[ITER:.*]] = %c0 to %[[ARG1]] step %c1 {
// CHECK:   %[[read:.*]] = vector.transfer_read %[[ALLOCA]][%[[ITER]], %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]}
// CHECK:   %[[select:.*]] = arith.select %[[ARG2]], %[[read]], %[[ARG3]]
// CHECK:   vector.transfer_write %[[select]], %arg4[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]}
// CHECK: }

// -----

// CHECK-LABEL: @fission_global_read_to_workgroup_write
// CHECK-SAME: %[[ARG0:.*]]: index
// CHECK-SAME: %[[ARG1:.*]]: memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG2:.*]]: memref<1x4xf32, #gpu.address_space<workgroup>>
func.func @fission_global_read_to_workgroup_write(%arg0: index, %arg1: memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>, %arg2: memref<1x4xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  scf.for %arg5 = %arg0 to %c16 step %c128 {
    %58 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1x4xf32>
    vector.transfer_write %58, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, memref<1x4xf32, #gpu.address_space<workgroup>>
  }
  return
}
// CHECK: %[[SUB:.*]] = arith.subi %c16, %[[ARG0]]
// CHECK: %[[DIV:.*]] = arith.ceildivui %[[SUB]], %c128
// CHECK: %[[ALLOCA:.*]] = memref.alloca(%[[DIV]])
// CHECK: scf.for %[[ITER:.*]] = %[[ARG0]] to %c16 step %c128 {
// CHECK:   %[[READ:.*]] = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]}
// CHECK:   %[[SUB:.*]] = arith.subi %[[ITER]], %[[ARG0]]
// CHECK:   %[[DIV:.*]] = arith.divui %[[SUB]], %c128
// CHECK:   vector.transfer_write %[[READ]], %[[ALLOCA]][%[[DIV]], %c0, %c0] {in_bounds = [true, true]}
// CHECK: }
// CHECK: scf.for %[[ITER:.*]] = %[[ARG0]] to %c16 step %c128 {
// CHECK:   %[[SUB:.*]] = arith.subi %[[ITER]], %[[ARG0]]
// CHECK:   %[[DIV:.*]] = arith.divui %[[SUB]], %c128
// CHECK:   %[[READ:.*]] = vector.transfer_read %[[ALLOCA]][%[[DIV]], %c0, %c0], %cst {in_bounds = [true, true]}
// CHECK:   vector.transfer_write %[[READ]], %arg2[%c0, %c0] {in_bounds = [true, true]}
// CHECK: }

// -----

// CHECK-LABEL: @no_fission_global_read_to_global_write
// CHECK-SAME: %[[ARG0:.*]]: memref<1x?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: memref<1x?x?xf32, #gpu.address_space<global>>
// CHECK-SAME: %[[ARG2:.*]]: index
func.func @no_fission_global_read_to_global_write(%arg0: memref<1x?x?xf32, #amdgpu.address_space<fat_raw_buffer>>, %arg1: memref<1x?x?xf32, #gpu.address_space<global>>, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  scf.for %arg3 = %c0 to %arg2 step %c1 {
    %read = vector.transfer_read %arg0[%arg3, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x?x?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x4xf32>
    vector.transfer_write %read, %arg1[%arg3, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x?x?xf32, #gpu.address_space<global>>
  }
  return
}
// CHECK: scf.for %[[ITER:.*]] = %c0 to %[[ARG2]] step %c1 {
// CHECK:   %[[READ:.*]] = vector.transfer_read
// CHECK:   vector.transfer_write %[[READ]], %arg1[%[[ITER]], %c0, %c0] {in_bounds = [true, true, true]}
// CHECK: }
// CHECK-NOT: scf.for
